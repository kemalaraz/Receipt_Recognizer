import os
import re
import logging
from typing import List

import torch
import numpy as np
from torch import nn
import torch.utils.data
import torch.nn.functional as F

from utils import FileHandler
from utils import ImageUtils
from utils import copyStateDict
from downloaders import download_model
from datasets import AlignCollate, RawDataset
from analyzer import Metrics, ImageManipulate
from models import Craft, RefineNet, Tesseract, ReNBAtt
from transforms import ImageTransform, PostTransform, AttnLabelConverter, CTCLabelConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("receiptrecognizer.module")

# TODO: Add functionality for working as seperate models one can actually load recognition model
# without detection model but now first the class should be initialized by first loading the detection
# model then other models can be loaded.

class ReceiptRecognizer(nn.Module):

    __models_path__  = os.path.join(os.path.expanduser("~"), "receipt_models")

    __device__ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    __gdrive_paths__ = {"craft_mlt_25k.pth": {
                                                "model": Craft,
                                                "model_id": "1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ"
                                            },
                        "craft_refiner_CTW1500.pth": {
                                                        "model": RefineNet,
                                                        "model_id": "1XSaFwBkOaFOdtk4Ane3DFyJGPRw6v5bO"
                                                    },
                        "TPS-ResNet-BiLSTM-Attn.pth": {
                                                        "model": ReNBAtt,
                                                        "model_id": "15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW"
                                                    }
                        }


    __character__ = "0123456789abcdefghijklmnopqrstuvwxyz"

    __model_types__ = ["detection", "recognition"]

    __numclass__ = None

    def __init__(self, detector:torch.nn.Module, recognizer:torch.nn.Module = None):
        super().__init__()
        self.detector = detector
        self.refiner = None
        self.recognizer = None

    @classmethod
    def build(cls, model:torch.nn.Module):
        return cls(model)

    @classmethod
    def load(cls, path:str, model_name:str, model_type:str = "detection") -> torch.nn.Module:
        assert model_type in ReceiptRecognizer.__model_types__, ("Given model type is not in supported model types",
                                                                f"Supported model types: {ReceiptRecognizer.__model_types__}")
        if model_type == "detection":
            model = ReceiptRecognizer.__gdrive_paths__[model_name]["model"]()
        elif model_type == "recognition":
            model = ReceiptRecognizer.__gdrive_paths__[model_name]["model"](ReceiptRecognizer.__numclass__)
        model.load_state_dict(copyStateDict(torch.load(path, map_location = ReceiptRecognizer.__device__)))
        return model

    @classmethod
    def load_from_drive(cls, model_name:str, model_type:str = "detection") -> torch.nn.Module:
        model_path = download_model(ReceiptRecognizer.__models_path__, file_name = model_name,
                                    file_id = ReceiptRecognizer.__gdrive_paths__[model_name]["model_id"])
        logger.info(f"Model downloaded to {model_path}")
        return cls.load(model_path, model_name, model_type)

    @classmethod
    def load_pretrained_model(cls, model_name:str, model_path:str = None, local_path:str = None, model_type:str = "detection") -> torch.nn.Module:
        """
        Loads the pretrained model from either by downloading from google drive or finding the reserved model path or
        by taking a local path.

        Args:
            model_name (str): The basename of the model.
            model_path (str): The reserved path of the model
            local_path (str, optional): Models path in the local computer. Defaults to None.

        Returns:
            torch.nn.Module: PyTorch model.
        """
        if not model_path: model_path = os.path.join(ReceiptRecognizer.__models_path__, model_name)
        if not local_path and not os.path.exists(model_path):
            model = cls.load_from_drive(model_name)
        elif local_path:
            if not os.path.exists(local_path):
                logger.info(f"Given model path does not exists -> {local_path}\nDownloading from drive...")
                model = cls.load_from_drive(model_name)
            model = cls.load(local_path, model_name, model_type)
        else:
            model = cls.load(model_path, model_name, model_type)

        return model

    @classmethod
    def from_pretrained(cls, model_name:str, local_path:str = None):
        assert isinstance(model_name, str), "Model name is required as string"
        assert model_name in ReceiptRecognizer.__gdrive_paths__.keys(), ("Model name not found"
                                    f"Models we have {str(ReceiptRecognizer.__gdrive_paths__.keys())}")

        if not os.path.exists(ReceiptRecognizer.__models_path__):
            os.makedirs(ReceiptRecognizer.__models_path__, exist_ok = True)

        model_path = os.path.join(ReceiptRecognizer.__models_path__, model_name)

        return cls.build(cls.load_pretrained_model(model_name, model_path, local_path))


    @torch.no_grad()
    @torch.jit.unused
    def detection(self, image:np.ndarray, canvas_size:int = 1280, mag_ratio:float = 1.5, text_threshold:float = 0.7,
                link_threshold:float = 0.4, low_text:float = 0.4, poly:bool = False, refine:bool = True, **kwargs):

        img_resized, target_ratio, size_heatmap = ImageTransform.resize_aspect_ratio(image, canvas_size,
                                                        interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = ImageTransform.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1) # [h, w, c] to [c, h, w]
        x = x.unsqueeze(0) # [c, h, w] to [b, c, h, w]
        x.to(ReceiptRecognizer.__device__)

        # forward pass
        y, feature = self.detector(x)

        # make score and link map
        heatmap = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # # refine link
        if refine:
            if not self.refiner:
                model_name = kwargs.get("refiner_name", "craft_refiner_CTW1500.pth")
                self.refiner = self.load_pretrained_model(model_name, os.path.join(ReceiptRecognizer.__models_path__, "craft_refiner_CTW1500.pth"))
                self.refiner.eval()
                y_refiner = self.refiner.forward(y, feature)
                score_link = y_refiner[0,:,:,0].cpu().data.numpy()
            else:
                y_refiner = self.refiner(y, feature)
                score_link = y_refiner[0,:,:,0].cpu().data.numpy()

        # Post-processing
        boxes, polys = PostTransform.getDetBoxes(heatmap, score_link, text_threshold, link_threshold, low_text, poly)

        # coordinate adjustment
        boxes = PostTransform.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = PostTransform.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        # render results (optional)
        render_img = heatmap.copy()
        render_img = np.hstack((render_img, score_link))
        ret_heatmap = ImageUtils.cvt2HeatmapImg(render_img)

        return boxes, polys, ret_heatmap

    @torch.no_grad()
    @torch.jit.unused
    def recognition(self, prediction_style:str = "Attn", imgH:int = 32, imgW:int = 100, pad:bool = False, rgb:bool = True,
                    image_folder:str = "../test_instances/results_detection", batch_size:int = 8, batch_max_length:int = 25):

        if not ReceiptRecognizer.__numclass__:
            if 'CTC' in prediction_style:
                self.converter = CTCLabelConverter(ReceiptRecognizer.__character__)
            else:
                self.converter = AttnLabelConverter(ReceiptRecognizer.__character__)

        if not self.recognizer:
            ReceiptRecognizer.__numclass__ = len(self.converter.character)
            self.recognizer = self.load_pretrained_model("TPS-ResNet-BiLSTM-Attn.pth", model_type = "recognition")
        self.recognizer.eval()

        # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
        AlignCollate_demo = AlignCollate(imgH = imgH, imgW = imgW, keep_ratio_with_pad = pad)
        demo_data = RawDataset(root=image_folder)  # use RawDataset
        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=batch_size,
            shuffle=False,
            collate_fn=AlignCollate_demo, pin_memory=True)

        result = []
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(ReceiptRecognizer.__device__)
            # For max length prediction
            length_for_pred = torch.IntTensor([batch_max_length] * batch_size).to(ReceiptRecognizer.__device__)
            text_for_pred = torch.LongTensor(batch_size, batch_max_length + 1).fill_(0).to(ReceiptRecognizer.__device__)

            if 'CTC' in prediction_style:
                preds = self.recognizer(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = self.converter.decode(preds_index, preds_size)

            else:
                preds = self.recognizer(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in prediction_style:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                result.append(pred)
        return result

    def allign_char_results(self, tesseract_results:List, model_results:List):
        final_result = []
        for tes, model in zip(tesseract_results, model_results):
            if tes == "\x0c":
                dollar_matches = list(re.finditer("s\d*", model))
                if len(dollar_matches) > 0:
                    new_model = ""
                    for dollar_match in dollar_matches:
                        if dollar_match.span()[1] - dollar_match.span()[0] > 2:
                            model = list(model)
                            model[dollar_match.span()[0]] = "$"
                            new_model += "".join(model[:dollar_match.span()[1]-2]) + "." + "".join(model[dollar_match.span()[1]-2:])
                        else:
                            new_model = model
                        final_result.append(new_model)
                else:
                    final_result.append(model)
            else:
                tes = re.sub(r"[\n\x0c]","",tes)
                final_result.append(tes)
        return final_result

if __name__ == "__main__":
    import cv2
    model = ReceiptRecognizer.from_pretrained("craft_mlt_25k.pth")
    model.detector.eval()
    test_path = "../test_instances/raw_images"
    results_path = "../test_instances/results_detection"
    # all_results = []
    # ground_truths = FileHandler.load_gt("/home/kemalaraz/Downloads/labels_my-project-name_2021-06-15-12-40-33.csv")
    # image_ious = {}
    # for k, image_path in enumerate(os.listdir(test_path)):
    #     image_path = os.path.join(test_path, image_path)
    #     image = ImageUtils.loadImage(image_path)
    #     image_name = os.path.basename(os.path.splitext(image_path)[0])

    #     # Get image ious with closest bbox
    #     bboxes, polys, heatmap = model.detection(image)
    #     image_ious[image_name] = {}
    #     for keys, values in ground_truths[image_name].items():
    #         if keys == "size":
    #             continue
    #         for value in values:
    #             # nearest_polygon = Metrics.nearest_polygon(value, bboxes)
    #             if keys not in image_ious.keys():
    #                 image_ious[keys] = [Metrics.polygon_IOU(value, bboxes)]
    #             else:
    #                 image_ious[keys].append(Metrics.polygon_IOU(value, bboxes))
    #     print(image_ious)

    all_results = []
    for k, image_path in enumerate(os.listdir(test_path)): # TODO: Can be changed with imutils paths.listimages(path)
        image_path = os.path.join(test_path, image_path)
        image = ImageUtils.loadImage(image_path)

        # Get the outcome of the detection model
        bboxes, polys, heatmap = model.detection(image)

        # crop images according to bboxes
        cropped_images = ImageUtils.crop_image(image[:,:,::-1], bboxes)

        image_results_path = os.path.join(results_path, os.path.basename(image_path).split("-")[0])
        if not os.path.exists(image_results_path):
            os.makedirs(image_results_path, exist_ok = True)

        # Get tesseract results and write cropped images to a folder
        tesseract_results = []
        for e, cropped_image in enumerate(cropped_images):
            cv2.imwrite(os.path.join(image_results_path, f"{e}-det-"+ os.path.basename(image_path)), cropped_image)
            tesseract_results.append(Tesseract.predict(cropped_image))

        # Find the characters from the folders that cropped images were written
        craft_rec_results = model.recognition(image_folder = image_results_path)
        print(tesseract_results)
        print(craft_rec_results)
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = results_path + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, heatmap)

        #FileHandler.saveResult(image_path, image[:,:,::-1], bboxes)
        alligned_results = model.allign_char_results(tesseract_results, craft_rec_results)
        all_results.append(alligned_results)

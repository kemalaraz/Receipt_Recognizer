import os
import logging

import torch
import numpy as np
from torch import nn

from utils import copyStateDict
from models import Craft, RefineNet
from downloaders import download_model
from utils.image_utils import loadImage, cvt2HeatmapImg
from transforms import ImageTransform, PostTransform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("receiptrecognizer.module")

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
                                                    }
                        }

    def __init__(self, model:torch.nn.Module):
        super().__init__()
        self.model = model
        self.refiner = None

    @classmethod
    def build(cls, model:torch.nn.Module):
        return cls(model)

    @classmethod
    def load_pretrained_model(cls, model_name:str, model_path:str) -> torch.nn.Module:
        model = ReceiptRecognizer.__gdrive_paths__[model_name]["model"]()
        model.load_state_dict(copyStateDict(torch.load(model_path, map_location = ReceiptRecognizer.__device__)))
        return model

    @classmethod
    def load_from_drive(cls, model_name:str) -> torch.nn.Module:
        model_path = download_model(ReceiptRecognizer.__models_path__, file_name = model_name,
                                    file_id = ReceiptRecognizer.__gdrive_paths__[model_name]["model_id"])
        logger.info(f"Model downloaded to {model_path}")
        return cls.load_pretrained_model(model_name, model_path)

    @classmethod
    def load_from_local(cls, local_path:str, model_name:str) -> torch.nn.Module:
        if not os.path.exists(local_path):
            logger.info(f"Given model path does not exists -> {local_path}\nDownloading from drive...")
            return load_from_drive(model_name)
        else:
            return cls.load_pretrained_model(model_name, local_path)

    @classmethod
    def from_pretrained(cls, model_name:str, local_path:str = None):
        assert isinstance(model_name, str), "Model name is required as string"
        assert model_name in ReceiptRecognizer.__gdrive_paths__.keys(), ("Model name not found"
                                    f"Models we have {str(ReceiptRecognizer.__gdrive_paths__.keys())}")

        if not os.path.exists(ReceiptRecognizer.__models_path__):
            os.makedirs(ReceiptRecognizer.__models_path__, exist_ok = True)

        model_path = os.path.join(ReceiptRecognizer.__models_path__, model_name)
        if not local_path and not os.path.exists(model_path):
            return cls.build(cls.load_from_drive(model_name))
        elif local_path:
            return cls.build(cls.load_from_local(local_path, model_name))
        else:
            return cls.build(cls.load_pretrained_model(model_name, model_path))

    @torch.no_grad()
    @torch.jit.unused
    def predict(self, image:np.ndarray, canvas_size:int = 1280, mag_ratio:float = 1.5, text_threshold:float = 0.7,
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
        y, feature = self.model(x)

        # make score and link map
        heatmap = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # # refine link
        if refine:
            if not self.refiner:
                model_name = kwargs.get("refiner_name", "craft_refiner_CTW1500.pth")
                self.refiner = self.load_pretrained_model(model_name, os.path.join(ReceiptRecognizer.__models_path__, "craft_refiner_CTW1500.pth"))
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
        ret_heatmap = cvt2HeatmapImg(render_img)

        return boxes, polys, ret_heatmap


if __name__ == "__main__":
    import cv2
    model = ReceiptRecognizer.from_pretrained("craft_mlt_25k.pth")
    test_path = "../test_instances/raw_images"
    for k, image_path in enumerate(os.listdir(test_path)):
        image_path = os.path.join(test_path, image_path)
        image = loadImage(image_path)

        bboxes, polys, heatmap = model.predict(image)
        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        cv2.imshow("test", heatmap)
        cv2.waitKey(0)
        exit(0)
        # mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        # cv2.imwrite(mask_file, heatmap)

        # file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)


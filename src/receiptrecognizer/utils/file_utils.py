__all__ = ["FileHandler"]

import os
import cv2
import csv
import json
import numpy as np
from typing import Dict

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

class FileHandler:
    """A general class in order to group together file operations"""
    # borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py

    @staticmethod
    def get_files(img_dir):
        imgs, masks, xmls = list_files(img_dir)
        return imgs, masks, xmls

    @staticmethod
    def saveResult(img_file, img, boxes, dirname='../test_instances/results/', verticals=None, texts=None):
            """ save text detection result one by one
            Args:
                img_file (str): image file name
                img (array): raw image context
                boxes (array): array of result file
                    Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
            Return:
                None
            """
            img = np.array(img)

            # make result file list
            filename, file_ext = os.path.splitext(os.path.basename(img_file))

            # result directory
            res_file = dirname + "res_" + filename + '.txt'
            res_img_file = dirname + "res_" + filename + '.jpg'

            if not os.path.isdir(dirname):
                os.mkdir(dirname)

            with open(res_file, 'w') as f:
                for i, box in enumerate(boxes):
                    poly = np.array(box).astype(np.int32).reshape((-1))
                    strResult = ','.join([str(p) for p in poly]) + '\r\n'
                    f.write(strResult)

                    poly = poly.reshape(-1, 2)
                    cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
                    ptColor = (0, 255, 255)
                    if verticals is not None:
                        if verticals[i]:
                            ptColor = (255, 0, 0)

                    if texts is not None:
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                        cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)

            # Save result image
            cv2.imshow(res_img_file, img)
            cv2.waitKey(0)

    @staticmethod
    def load_gt(*args) -> Dict:
        """
        Loads ground truth bounding boxes for images depending on its extention.

        CSV FILE -> business_name,274,124,166,32,1003-receipt.jpg,750,1000

        Args:
            path (str): Path to the file

        Returns:                                                                                    X ,  Y
            Dict: {"image_name1":{"tag1":[np.array1, np.array2, ...], "tag2": [...] ... , "size": (int, int)}, ...}
        """
        # TODO: Add support for other formats such as YOLO, VOC XML etc.
        for path in args:
            gt_dict = {}
            if os.path.splitext(path)[-1] == ".csv":
                with open(path, "r") as file:
                    reader = csv.reader(file)
                    for e, row in enumerate(reader):
                        if len(row) > 8:
                            print(row)
                        image_name = os.path.splitext(row[-3])[0]
                        if image_name not in gt_dict.keys():
                            gt_dict[image_name] = {row[0]: [np.array([int(elem) for elem in row[1:-3]]).reshape(-1, 2)],
                                                                        "size": (int(row[-2]), int(row[-1]))}
                        else:
                            if row[0] in gt_dict[image_name].keys():
                                gt_dict[image_name][row[0]].append(np.array([int(elem) for elem in row[1:-3]]).reshape(-1, 2))
                            else:
                                gt_dict[image_name][row[0]] = [np.array([int(elem) for elem in row[1:-3]]).reshape(-1, 2)]
            if os.path.splitext(path)[-1] == ".json":
                pass
        return gt_dict
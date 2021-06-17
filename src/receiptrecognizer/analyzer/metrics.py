import numpy as np
from typing import List
from shapely.geometry import Polygon

class Metrics:

    @staticmethod
    def shape_shifter(arr):
        return [(arr[0,0], arr[0,1]), (arr[1,0], arr[0,1]), (arr[1,0], arr[1,1]), (arr[0,0], arr[1,1])]

    @staticmethod
    def polygon_IOU_old(arr1:np.array, arr2:np.array):
        """
        Calculate the intersection over union metric for two polygons.

        Args:
            polygon_1 ([type]): [description]
            polygon_2 ([type]): [description]
        """
         # areas = [Metrics.calc_area(arr) for arr in [arr1, arr2]]
        print(arr1)
        print(arr2)
        if isinstance(arr1, np.ndarray):
            arr1 = Metrics.shape_shifter(arr1)
        if isinstance(arr2, np.ndarray):
            arr2 = Metrics.shape_shifter(arr2)
        poly1 = Polygon(arr1)
        poly2 = Polygon(arr2)

        intersection = poly1.intersection(poly2).area

        return  intersection / (poly1.area + poly2.area - intersection)

    @staticmethod
    def polygon_IOU(target:np.array, bboxes):
        """
        Calculate the intersection over union metric for two polygons.

        Args:
            polygon_1 ([type]): [description]
            polygon_2 ([type]): [description]
        """

        if isinstance(target, np.ndarray):
            target = Metrics.shape_shifter(target)

        poly1 = Polygon(target)

        if isinstance(bboxes, list) and len(bboxes) == 4:
            poly2 = Polygon(bboxes)
            intersection = poly1.intersection(poly2).area
            return intersection / (poly1.area + poly2.area - intersection)

        iou_final = []
        for bbox in bboxes:
            if isinstance(bbox, np.ndarray) and bbox.shape[1] == 2:
                bbox = Metrics.shape_shifter(bbox)

            poly2 = Polygon(bbox)
            intersection = poly1.intersection(poly2).area

            iou = intersection / (poly1.area + poly2.area - intersection)
            iou_final.append(iou)

        return sorted(iou_final, reverse = True)[0]

    @staticmethod
    def nearest_polygon(target_arr:np.array, candidate_arrs:List[np.array]):
        """[summary]

        Args:
            target_polygon (np.array): [description]
            candidate_polygons (List[np.array]): [description]
        """
        if target_arr.shape[1] == 2:
            target_arr = Metrics.shape_shifter(target_arr)

        target_p_dist = Polygon(target_arr).centroid
        dist = 100000

        for cand in candidate_arrs:
            if cand.shape[1] == 2:
                cand = Metrics.shape_shifter(cand)
            cand_p_dist = Polygon(cand).centroid
            cand_distance = np.linalg.norm(target_p_dist - cand_p_dist)
            if cand_distance < dist:
                nearest = cand
                dist = cand_distance
        return nearest
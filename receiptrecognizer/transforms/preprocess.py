# -*- coding: utf-8 -*-
import cv2
import numpy as np

class ImageTransform:
    """A general class in order to group together image transformations"""
    ### mean and std are calculated according to the CRAFT paper https://arxiv.org/pdf/1904.01941.pdf ###
    __mean__ = (0.485, 0.456, 0.406)
    __std__ = (0.229, 0.224, 0.225)

    @staticmethod
    def normalizeMeanVariance(in_img, mean=ImageTransform.__mean__, variance=ImageTransform.__std__):
        # should be RGB order
        img = in_img.copy().astype(np.float32)

        img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
        img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
        return img

    @staticmethod
    def denormalizeMeanVariance(in_img, mean=ImageTransform.__mean__, variance=ImageTransform.__std__):
        # should be RGB order
        img = in_img.copy()
        img *= variance
        img += mean
        img *= 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    @staticmethod
    def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
        height, width, channel = img.shape

        # magnify image size
        target_size = mag_ratio * max(height, width)

        # set original image size
        if target_size > square_size:
            target_size = square_size

        ratio = target_size / max(height, width)

        target_h, target_w = int(height * ratio), int(width * ratio)
        proc = cv2.resize(img, (target_w, target_h), interpolation = interpolation)


        # make canvas and paste image
        target_h32, target_w32 = target_h, target_w
        if target_h % 32 != 0:
            target_h32 = target_h + (32 - target_h % 32)
        if target_w % 32 != 0:
            target_w32 = target_w + (32 - target_w % 32)
        resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
        resized[0:target_h, 0:target_w, :] = proc
        target_h, target_w = target_h32, target_w32

        size_heatmap = (int(target_w/2), int(target_h/2))

        return resized, ratio, size_heatmap
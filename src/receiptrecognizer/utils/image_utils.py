__all__ = ["ImageUtils"]

import cv2
import math
import imutils
import numpy as np
from skimage import io


class ImageUtils:

    @staticmethod
    def align_image(image:np.array, template:np.array, maxFeatures=500, keepPercent=0.2, debug=False
                    ) -> np.array:
        """
        If a good receipt template can be found use it.

        Args:
            image (np.array): Original image
            template (np.array): Template to be warped to
            maxFeatures (int, optional): . Defaults to 500.
            keepPercent (float, optional): . Defaults to 0.2.
            debug (bool, optional): . Defaults to False.

        Returns:
            [np.array]: allgined image
        """
        # convert both the input image and template to grayscale
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # use ORB to detect keypoints and extract (binary) local
        # invariant features
        orb = cv2.ORB_create(maxFeatures)
        (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
        (kpsB, descsB) = orb.detectAndCompute(templateGray, None)

        # match the features
        method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
        matcher = cv2.DescriptorMatcher_create(method)
        matches = matcher.match(descsA, descsB, None)

        # sort the matches by their distance (the smaller the distance,
        # the "more similar" the features are)
        matches = sorted(matches, key=lambda x:x.distance)
        # keep only the top matches
        keep = int(len(matches) * keepPercent)
        matches = matches[:keep]
        # check to see if we should visualize the matched keypoints
        if debug:
            matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
                matches, None)
            matchedVis = imutils.resize(matchedVis, width=1000)
            cv2.imshow("Matched Keypoints", matchedVis)
            cv2.waitKey(0)

        # allocate memory for the keypoints (x, y)-coordinates from the
        # top matches -- we'll use these coordinates to compute our
        # homography matrix
        ptsA = np.zeros((len(matches), 2), dtype="float")
        ptsB = np.zeros((len(matches), 2), dtype="float")
        # loop over the top matches
        for (i, m) in enumerate(matches):
            # indicate that the two keypoints in the respective images
            # map to each other
            ptsA[i] = kpsA[m.queryIdx].pt
            ptsB[i] = kpsB[m.trainIdx].pt

        # compute the homography matrix between the two sets of matched
        # points
        (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
        # use the homography matrix to align the images
        (h, w) = template.shape[:2]
        aligned = cv2.warpPerspective(image, H, (w, h))
        # return the aligned image
        return aligned


    @staticmethod
    def allign_bboxes(img_shape:int, bboxes:np.array, divide_imagey:int = 40, divide_imagex:int = 30):
        """
        Allign boxes with a left-to-right and top-to-bottom manner. Draws grids with respect to x and y axis and first

        Args:
            img_shape (int): The shape of the image (h, w)
            bboxes (np.array): The bounding boxes (x, y)
            divide_imagey (int, optional): How much lines will be drawn in y axis. Defaults to 40.
            divide_imagex (int, optional): How much lines will be drawn in x axis. Defaults to 30.

        Returns:
            [type]: [description]
        """
        top_lefts = bboxes[:, 0, :]
        bottom_lefts = bboxes[:, -1, :]
        top_rights = bboxes[:, 1, :]
        all_y_dims = [np.linalg.norm(top_left - bottom_left) for top_left, bottom_left in zip(top_lefts, bottom_lefts)]
        all_x_dims = [np.linalg.norm(top_left - top_right) for top_left, top_right in zip(top_lefts, top_rights)]
        mean_y, std_y = math.ceil(np.mean(all_y_dims)), math.ceil(np.std(all_y_dims))
        mean_x, std_x = math.ceil(np.mean(all_x_dims)), math.ceil(np.std(all_x_dims))

        sorted_bboxes = [x for _,x in sorted(zip(bboxes[:, 0, 1],bboxes))]

        low_y = 0
        alligned_bboxes = []

        for y in range(0, img_shape[0] + mean_y + std_y,  mean_y + std_y):
            y_alligned = [bbox for bbox in sorted_bboxes if bbox[0,1] >= low_y and bbox[0,1] <= y]
            for x in range(0, img_shape[1] + mean_x  + std_x, mean_x + std_x):
                if y_alligned:
                    x_alligned = sorted([bbox for bbox in y_alligned if bbox[0,0] >= low_x and bbox[0,0] <= x], key=lambda x: x[0,0])
                    if x_alligned:
                        alligned_bboxes.extend(sorted(x_alligned, key=lambda x: x[0,1]))
                low_x = x
            low_y = y
        return alligned_bboxes

    @staticmethod
    def loadImage(img_file):
        img = io.imread(img_file)           # RGB order
        if img.shape[0] == 2: img = img[0]
        if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 4:   img = img[:,:,:3]
        img = np.array(img)

        return img

    @staticmethod
    def cvt2HeatmapImg(img):
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        return img

    @staticmethod
    def crop_image(img, bboxes:np.array, divide_imagey:int = 40, divide_imagex:int = 30):
        """
        Crop image areas where bounding boxes exist.

        Args:
            img ([np.array]): BGR image
            bboxes (np.array): Bounding boxes where x, y
            divide_imagey (int, optional): How much lines will be drawn in y axis. Defaults to 40.
            divide_imagex (int, optional): How much lines will be drawn in x axis. Defaults to 30.
        """

        alligned_bboxes = ImageUtils.allign_bboxes(img.shape, bboxes, divide_imagey, divide_imagex)
        cropped_images = []
        for box in alligned_bboxes:
            poly = np.array(box).astype(np.int32)
            ## (1) Crop the bounding rect
            rect = cv2.boundingRect(poly)
            x,y,w,h = rect
            croped = img[y:y+h, x:x+w].copy()

            ## (2) make mask
            poly = poly - poly.min(axis=0)

            mask = np.zeros(croped.shape[:2], np.uint8)
            cv2.drawContours(mask, [poly], -1, (255, 255, 255), -1, cv2.LINE_AA)

            ## (3) do bit-op
            dst = cv2.bitwise_and(croped, croped, mask=mask)

            # ## (4) add the white background
            bg = np.ones_like(croped, np.uint8)*255
            cv2.bitwise_not(bg,bg, mask=mask)
            dst2 = bg+ dst

            cropped_images.append(dst2)
        return cropped_images

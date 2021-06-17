import cv2
import numpy as np

class ImageManipulate:

    __sobel__ = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]])

    __gradient__ = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]])

    __filters__ = {
                        "sobel_vertical": __sobel__,
                        "sobel_horizontal": __sobel__[:, [2, 1, 0]].T,
                        "gradient_vertical": __gradient__,
                        "gradient_horizontal": __gradient__[:, [2, 1, 0]].T
                  }

    @staticmethod
    def convolution(image:np.array, filter_name:str):
        assert filter_name in ImageManipulate.__filters__.keys(), ("Given filter not found in filters"
                                                        f"Supported filters are {list(ImageManipulate.__filters__.keys())}")
        img_conv = cv2.filter2D(image, -1, ImageManipulate.__filters__[filter_name])
        return img_conv

if __name__ == "__main__":
    img = cv2.imread("/home/kemalaraz/Desktop/Reciept_Recognizer/test_instances/raw_images/1023-receipt.jpg")
    img_horz = ImageManipulate.convolution(img, "sobel_horizontal")
    img_ver = ImageManipulate.convolution(img, "sobel_vertical")
    cv2.imshow("", img_ver+img_horz)
    cv2.waitKey(0)
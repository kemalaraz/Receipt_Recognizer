import pytesseract
import numpy as np

class Tesseract:

    @staticmethod
    def predict(img:np.array) -> str:
        return pytesseract.image_to_string(img)
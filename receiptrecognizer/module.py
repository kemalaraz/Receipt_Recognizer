import os

import torch

class ReceiptRecognizer:

    __gdrive_paths__ = {"model1": "path"}

    @classmethod
    def load_from_drive(cls, drive_path) -> torch.nn.Module:
        pass

    @classmethod
    def load_from_local(cls, local_path) -> torch.nn.Module:
        pass

    @classmethod
    def from_pretrained(cls, model_name:str, local_path:str = None):
        if not local_path and not os.path.exists(local_path):
            return cls.load_from_drive()
        else:
            return cls.load_from_local()
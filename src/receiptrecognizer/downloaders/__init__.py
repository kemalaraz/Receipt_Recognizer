import os
import logging

from .google_downloader import GoogleDriveDownloader

logger = logging.getLogger("receiptrecognizer.model_downloader")

#TODO: Add http downloader
def download_model(dest_path:str, file_name:str, file_id:str, **kwargs):
    logger.info(f"Downloding model to {dest_path} from google drive")
    return GoogleDriveDownloader.download(dest_path, file_name = file_name, file_id = file_id)

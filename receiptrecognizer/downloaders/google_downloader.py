import os
from google_drive_downloader import GoogleDriveDownloader as gdrive

class GoogleDriveDownloader:
    """Class for downloading models from google drive"""

    @staticmethod
    def download(dest_path: str, file_name: str = None, file_id: str = None,
            extract: bool = False, showsize: bool = True, **kwargs) -> str:
        assert isinstance(file_id, str), "file id must be string but found:{}".format(type(file_id))
        assert isinstance(file_name, str), "file name must be string but found:{}".format(type(file_name))
        if not os.path.exists(dest_path):
            os.makedirs(dest_path, exist_ok=True)
        file_path = os.path.join(dest_path, file_name)
        gdrive.download_file_from_google_drive(file_id, file_path, unzip=extract, showsize=showsize, **kwargs)
        return file_path
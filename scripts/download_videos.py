"""Download source videos from Google Drive."""

import gdown
from config import SOURCE_DIR, GOOGLE_DRIVE_FOLDER_ID


def download_all():
    """Download all videos from the shared Google Drive folder."""
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{GOOGLE_DRIVE_FOLDER_ID}"
    print(f"Downloading videos from: {url}")
    gdown.download_folder(url, output=str(SOURCE_DIR), quiet=False)
    print(f"Videos saved to: {SOURCE_DIR}")


if __name__ == "__main__":
    download_all()

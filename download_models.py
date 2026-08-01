import os
import gdown

FOLDER_URL = "https://drive.google.com/drive/folders/1rAonN5_zXpeWWmPDwZzZXnjTtcldlMFr?usp=drive_link"

def download_models():
    if os.path.exists("models") and len(os.listdir("models")) > 0:
        print("Models already downloaded.")
        return

    print("Downloading models...")
    gdown.download_folder(
        url=FOLDER_URL,
        output="models",
        quiet=False,
        use_cookies=False
    )
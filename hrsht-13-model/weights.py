# import gdown
# def download_weights():
#     gdown.download(id="1--7p9rRJy7WU4OmomkzM8i0veetZctTT",
#                    output="weights/modeldense1.h5",
#                    quiet=False)

import os
import gdown

FILE_ID = "1--7p9rRJy7WU4OmomkzM8i0veetZctTT"
DEST   = "weights/modeldense1.h5"

def download_weights():
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    if not os.path.isfile(DEST):
        print(f"Downloading weights to {DEST}…")
        gdown.download(id=FILE_ID, output=DEST, quiet=False)
    else:
        print(f"Weights already present at {DEST}")

if __name__ == "__main__":
    download_weights()

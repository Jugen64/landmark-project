from pathlib import Path
from collections import defaultdict
import random

from src.utils import data_utils

METADATA_CSV = Path("data/processed/metadata.csv")
SPLITS_DIR = Path("data/processed/splits")
TRAIN_IMAGE_FILE = SPLITS_DIR / "train_images.txt"
VAL_IMAGE_FILE = SPLITS_DIR / "val_images.txt"
TEST_IMAGE_FILE = SPLITS_DIR / "test_images.txt"

TRAIN_LANDMARK_FILE = SPLITS_DIR / "train_ids.txt"
VAL_LANDMARK_FILE = SPLITS_DIR / "val_ids.txt"
TEST_LANDMARK_FILE = SPLITS_DIR / "test_ids.txt"

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

RANDOM_SEED = 64
random.seed(RANDOM_SEED)

def write_split(path, landmarks):
    with open(path, "w") as f:
        for lid in sorted(landmarks):
            f.write(f"{lid}\n")

if __name__ == "__main__":

    landmark_to_images = data_utils.metadata_landmark_to_image_path(METADATA_CSV)

    country_to_landmarks = data_utils.metadata_country_to_landmarks(METADATA_CSV)

    train_landmarks, val_landmarks, test_landmarks = set(), set(), set()
    train_images, val_images, test_images = set(), set(), set()

    for country_id, landmarks in country_to_landmarks.items():
        landmarks = list(landmarks)
        random.shuffle(landmarks)

        n = len(landmarks)
        n_train = int(TRAIN_FRAC * n)
        n_val = int(VAL_FRAC * n)

        for landmark in landmarks[:n_train]:
            train_landmarks.add(landmark)
            train_images.update(landmark_to_images[landmark])

        for landmark in landmarks[n_train:n_train + n_val]:
            val_landmarks.add(landmark)
            val_images.update(landmark_to_images[landmark])

        for landmark in landmarks[n_train + n_val:]:
            test_landmarks.add(landmark)
            test_images.update(landmark_to_images[landmark])


    write_split(TRAIN_LANDMARK_FILE, train_landmarks)
    write_split(VAL_LANDMARK_FILE, val_landmarks)
    write_split(TEST_LANDMARK_FILE, test_landmarks)

    write_split(TRAIN_IMAGE_FILE, train_images)
    write_split(VAL_IMAGE_FILE, val_images)
    write_split(TEST_IMAGE_FILE, test_images)

    print("done!")


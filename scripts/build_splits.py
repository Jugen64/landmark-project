from pathlib import Path
from collections import defaultdict
import random

from src.utils import metadata_utils

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

RANDOM_SEED = 64
random.seed(RANDOM_SEED)


if __name__ == "__main__":
    print("Splitting data by landmark_id...")

    train_set, val_set, test_set = metadata_utils.partition_landmarks()

    print(f"train_set size: {len(train_set)}")
    print(f"val_set size: {len(val_set)}")
    print(f"test_set size: {len(test_set)}")

    metadata_utils.write_splits(train_set, val_set, test_set)

    print("Data has been split by landmark_id.")


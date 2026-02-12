from torch.utils.data import DataLoader
from projects.landmark_project.src.datasets import landmark_dataset
from src.utils.transforms import get_transforms, get_train_transform, get_eval_transform

from src.utils import metadata_utils
from pathlib import Path

train_transform = get_train_transform()
val_transform   = get_eval_transform()


DATASET_DIR = Path("~/Documents/Code/projects/datasets").expanduser()
IMAGE_DIR = DATASET_DIR / "gldv2_micro/images"

SPLIT_DIR = Path("~/Documents/Code/projects/landmark_project/data/processed/splits").expanduser()
TRAIN_IMAGES = SPLIT_DIR / "train_images.txt"

def build_dataloader(image_paths, labels, split, batch_size):
    dataset = landmark_dataset.LandmarkDataset(
        image_paths=image_paths,
        labels=labels,
        transform=get_transforms(split),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=4,
        pin_memory=False,
    )

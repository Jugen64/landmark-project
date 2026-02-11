from torch.utils.data import DataLoader
import torch
from src.datasets import landmark_image_dataset
from src.utils.transforms import get_transforms, get_train_transform, get_eval_transform
from src.dataloaders import landmark_dataloader
from src.models.landmark_classifier import LandmarkClassifier
from src.training.validate import run_validation

from src.utils import metadata_utils
from pathlib import Path

train_transform = get_train_transform()
val_transform   = get_eval_transform()


DATASET_DIR = Path("~/Documents/Code/projects/datasets").expanduser()
IMAGE_DIR = DATASET_DIR / "gldv2_micro/images"

SPLIT_DIR = Path("~/Documents/Code/projects/landmark_project/data/splits").expanduser()
TRAIN_IMAGES = SPLIT_DIR / "train_images.txt"
TRAIN_LABELS = SPLIT_DIR / "train_countries.txt"
VAL_IMAGES = SPLIT_DIR / "val_images.txt"
VAL_LABELS = SPLIT_DIR / "val_countries.txt"

if __name__ == "__main__":
    train_image_list = metadata_utils.split_to_list(TRAIN_IMAGES)
    train_label_list = metadata_utils.split_to_list(TRAIN_LABELS)

    val_data_list = metadata_utils.split_to_list(VAL_IMAGES)
    val_label_list = metadata_utils.split_to_list(VAL_LABELS)

    country_list = list(set(train_label_list))
    
    training_loader = landmark_dataloader.build_dataloader(train_image_list, train_label_list, "train", 100)
    val_loader = landmark_dataloader.build_dataloader(val_data_list, val_label_list, "train", 100)

    xb, yb = next(iter(training_loader))

    print(xb.shape)
    print(yb.shape)
    print(yb.dtype)
    print(yb.min(), yb.max())

    model = LandmarkClassifier(len(country_list))
    out = model(xb)
    print("model out:", out.shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(out, yb)
    print("loss:", loss.item())

    NUM_EPOCHS = 100

    for epoch in range(NUM_EPOCHS):
        model.train()

        for step, (xb, yb) in enumerate(training_loader):
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            print(f"step {step}, loss {loss.item():.4f}")
            if step == 200:
                break
        
        val_loss, val_acc = run_validation(model, val_loader, criterion)

        print(
            f"epoch {epoch:03d} | "
            f"train loss {loss.item():.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_acc:.4f}"
        )
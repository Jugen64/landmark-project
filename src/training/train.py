import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights

from PIL import Image
from collections import Counter
from pathlib import Path

from src.utils.transforms import get_transforms, get_train_transform, get_eval_transform
from projects.landmark_project.src.datasets.landmark_dataset import LandmarkDataset
from src.dataloaders import landmark_dataloader
from src.models.landmark_classifier import LandmarkClassifier
from src.training.validate import run_validation

from src.utils import metadata_utils

train_transform = get_train_transform()
val_transform   = get_eval_transform()

SUBSET_TEST = False 
VERBOSE = True
ADAM_LR = 1e-3
TRAINING_EPOCHS = 5
SUBSET_EPOCHS = 50

DATASET_DIR = Path("~/Documents/Code/projects/datasets").expanduser()
IMAGE_DIR = DATASET_DIR / "gldv2_micro/images"

SPLIT_DIR = Path("~/Documents/Code/projects/landmark_project/data/splits").expanduser()
TRAIN_IMAGES = SPLIT_DIR / "train_images.txt"
TRAIN_LABELS = SPLIT_DIR / "train_countries.txt"
VAL_IMAGES = SPLIT_DIR / "val_images.txt"
VAL_LABELS = SPLIT_DIR / "val_countries.txt"


def run_model(model, data_loader, criterion, optimizer, device, epochs):
    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        running_correct = 0
        total = 0

        for image_batch, label_batch in data_loader:

            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()

            batch_distributions = model(image_batch)
            loss = criterion(batch_predictions, label_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * image_batch.size(0)
            batch_predictions = batch_distributions.argmax(dim=1)

            running_correct += (batch_predictions == label_batch).sum().item()
            total += image_batch.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_correct / total

        print(f"Epoch {epoch+1}/{epochs} "
              f"| Loss: {epoch_loss:.4f} "
              f"| Acc: {epoch_acc:.4f}")

def get_datasets(transform):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = LandmarkDataset(
        image_path="data/splits/train/images.txt",
        label_path="data/splits/train/countries.txt",
        transform=transform
    )

    val_dataset = LandmarkDataset(
        image_path="data/splits/val/images.txt",
        label_path="data/splits/val/countries.txt",
        transform=transform
    )

    test_dataset = LandmarkDataset(
        image_path="data/splits/val/images.txt",
        label_path="data/splits/val/countries.txt",
        transform=transform
    )

    return train_dataset, val_dataset, test_dataset

def get_sampler(dataset):

    counts = Counter(dataset.labels)
    num_samples = len(dataset)
    num_classes = len(dataset.label_map)

    class_counts = torch.tensor(
        [counts[i] for i in range(num_classes)],
        dtype=torch.float
    )

    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in dataset.labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler, counts, num_samples, num_classes

def set_up_model(weights, device, num_classes):
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    #### vvvv ResNet18 vvvv ###
    weights = ResNet18_Weights.DEFAULT

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=weights.transforms().mean,
            std=weights.transforms().std
        )
    ])

    train_dataset, val_dataset, test_dataset = get_datasets(transform)
    train_sampler, train_counts, num_samples, num_classes = get_sampler(train_dataset)
    train_epochs = TRAINING_EPOCHS
    val_epochs = 1

    if VERBOSE:
        print(f"Total samples: {num_samples}")
        print(f"Class distribution: {train_counts}")

    if SUBSET_TEST:
        train_dataset = Subset(train_dataset, range(20))
        train_epochs = SUBSET_EPOCHS
        print("Training on 20-sample subset.")
    else:
        train_epochs = TRAINING_EPOCHS
        print("Training on full subset.")
    
    train_dataset_loader = DataLoader(
        train_dataset,
        batch_size=64,
        sampler=train_sampler,
        num_workers=4
    )

    val_dataset_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4
    )
    
    model, optimizer, criterion = set_up_model(weights, device, num_classes)

    run_model(model, train_dataset_loader, criterion, optimizer, device, train_epochs)
    run_model(model, val_dataset_loader, criterion, optimizer, device, val_epochs)

    


if __name__ == "__main__":
    main()

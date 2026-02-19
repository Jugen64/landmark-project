import torch
from torch.utils.data import Dataset
from PIL import Image
import csv

class LandmarkDataset(Dataset):
    def __init__(self, image_path, label_path, transform=None, subset=0):
        self.image_paths = []
        self.labels = []
        count = 0

        with open(image_path, 'r') as image_f:
            image_reader = csv.reader(image_f)
            for row in image_reader:
                image_name = row[0]
                self.image_paths.append(image_name)
                count += 1
                if count == subset:
                    print(f"breaking!")
                    break
                
        count = 0
        with open(label_path, 'r') as label_f:
            label_reader = csv.reader(label_f)
            for row in label_reader:
                label_name = row[0]
                self.labels.append(label_name)
                count += 1
                if count == subset:
                    break

        self.label_set = set(self.labels)
        self.label_map = {}

        i = 0
        for label in self.label_set:
            self.label_map[label] = i
            i += 1

        self.labels = [self.label_map[label] for label in self.labels]
        
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

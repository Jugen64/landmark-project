import torch
from torch.utils.data import Dataset
from PIL import Image

class LandmarkDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.class_list = sorted(set(labels))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_list)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        self.labels = torch.tensor(
            [self.class_to_idx[label] for label in labels],
            dtype=torch.long,
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        img = Image.open(self.image_paths[i]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.labels[i]
        return img, label

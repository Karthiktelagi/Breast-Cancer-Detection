import os
import torch
from torch.utils.data import Dataset
import cv2

class BreastDataset(Dataset):
    def __init__(self, folder):
        self.paths = []
        self.labels = []

        classes = ["normal", "benign", "malignant"]

        for label, cls in enumerate(classes):
            path = os.path.join(folder, cls)

            for img in os.listdir(path):
                self.paths.append(os.path.join(path, img))
                self.labels.append(label)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], 0)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0

        img = torch.tensor(img).unsqueeze(0).float()
        label = torch.tensor(self.labels[idx])

        return img, label
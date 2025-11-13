import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class LungeDataset(Dataset):
    def __init__(self, keypoints_dir, classes, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        self.classes = classes

        for idx, cls in enumerate(classes):
            cls_folder = os.path.join(keypoints_dir, cls)
            for file in os.listdir(cls_folder):
                if file.endswith(".npy"):
                    self.samples.append(os.path.join(cls_folder, file))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = np.load(self.samples[idx])  # Shape: (num_frames, 99)
        seq = torch.tensor(seq, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.transform:
            seq = self.transform(seq)
        return seq, label


def get_dataloaders(keypoints_dir="C:\\python\\formfit_pose\\main\\lunges\\keypoints_normalized", batch_size=4, test_size=0.2, shuffle=True):
    classes = sorted(os.listdir(keypoints_dir))  # ['back_straight', 'correct', 'legs_far']
    dataset = LungeDataset(keypoints_dir, classes)

    # Train/test split
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=test_size, shuffle=shuffle, stratify=dataset.labels)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, classes

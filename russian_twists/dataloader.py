import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class AugmentPose:
    """Data augmentation for pose sequences"""
    def __init__(self, noise_std=0.01, scale_range=(0.95, 1.05)):
        self.noise_std = noise_std
        self.scale_range = scale_range
    
    def __call__(self, seq):
        # Add random noise
        noise = torch.randn_like(seq) * self.noise_std
        seq = seq + noise
        
        # Random scaling
        scale = np.random.uniform(*self.scale_range)
        seq = seq * scale
        
        return seq

class PushupDataset(Dataset):
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


def get_dataloaders(keypoints_dir="keypoints", batch_size=4, test_size=0.2, shuffle=True, augment=True):
    classes = sorted(os.listdir(keypoints_dir))  # ['correct', 'legs_bent']
    
    # Create datasets with and without augmentation
    transform = AugmentPose() if augment else None
    train_dataset_full = PushupDataset(keypoints_dir, classes, transform=transform)
    val_dataset_full = PushupDataset(keypoints_dir, classes, transform=None)  # No augmentation for validation

    # Train/test split
    indices = list(range(len(train_dataset_full)))
    train_idx, val_idx = train_test_split(indices, test_size=test_size, shuffle=shuffle, stratify=train_dataset_full.labels)

    train_dataset = torch.utils.data.Subset(train_dataset_full, train_idx)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, classes

from image_dataset import ImageDataset
from net import Net
from pathlib import Path
from torch.utils.data import ConcatDataset

import torch
import torchvision.transforms as transforms
import numpy as np

def augment_dataset(X_path:Path, y_path:Path, augmentation_iter:int = 5) -> ConcatDataset:
    """
    Extends the normal dataset generating random transformed images.
    """
    # Define your transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=5, scale=(0.95, 1.05)),
    ])

    # Create a PyTorch dataset from the X and y arrays
    dataset = ConcatDataset(
        [ImageDataset(X_path, y_path, transform = transform)
         for _ in range(augmentation_iter) ])

    return dataset


# def visualize_augmentation():


data = augment_dataset(Path("../data/X_train.npy"), Path("../data/Y_train.npy"))
print(len(data))


    



import numpy as np
import torch
import requests
import io
from os import path
from typing import Tuple, List, Callable
from pathlib import Path
import os

from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    """
    Creates a DataSet from numpy arrays while keeping the data
    in the more efficient numpy arrays for as long as possible and only
    converting to torchtensors when needed (torch tensors are the objects used
    to pass the data through the neural network and apply weights).
    """

    def __init__(self, x: Path, y: Path, transform=None) -> None:
        # Target labels
        self.targets = ImageDataset.load_numpy_arr_from_npy(y)
        # Images
        self.imgs = ImageDataset.load_numpy_arr_from_npy(x)
        # Apply transform
        self.transform = transform

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        image = torch.tensor(self.imgs[idx] / 255).float()
        label = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    @staticmethod
    def load_numpy_arr_from_npy(path: Path) -> np.ndarray:
        """
        Loads a numpy array from local storage.
        Input:
        path: local path of file
        Outputs:
        dataset: numpy array with input features or labels
        """

        return np.load(path)



class AugImageDataset(ConcatDataset):
    """
    Used for lazy data augmentation with while using the ImageDataset
    """

    def __init__(self, x: Path, y: Path,
                 augmentation_iter:int = 5,
                 transform=None) -> None:

        # Define your transformations
        if not transform:
            transform = transforms.Compose([
                transforms.RandomAffine(degrees=5, scale=(0.95, 1.05)),
                transforms.Resize((128,128))
            ])

        # Create a PyTorch dataset from the X and y arrays
        super().__init__(
            [ImageDataset(x, y, transform = tr)
             for tr in [None] + [transform] * augmentation_iter ])

        self.transform = transform
        self.targets = np.concatenate(
            [dataset.targets for dataset in self.datasets])


def load_numpy_arr_from_url(url: str) -> np.ndarray:
    """
    Loads a numpy array from surfdrive.
    Input:
    url: Download link of dataset
    Outputs:
    dataset: numpy array with input features or labels
    """

    response = requests.get(url)
    response.raise_for_status()

    return np.load(io.BytesIO(response.content))


def gen_augmented_dataset(X_path:Path,
                          y_path:Path,
                          augmentation_iter:int = 5,
                          transform:Callable = None) -> ConcatDataset:
    """
    Generate a new dataset factor of augmentation_iter bigger than the provided data
    """
    # Define your transformations
    if not transform:
        transform = transforms.Compose([
            transforms.RandomAffine(degrees=5, scale=(0.95, 1.05)),
        ])

    # Create a PyTorch dataset from the X and y arrays
    dataset = ConcatDataset(
        [ImageDataset(X_path, y_path, transform = tr)
         for tr in [None] + [transform] * augmentation_iter ])

    return dataset


if __name__ == "__main__":
    train_dataset = AugImageDataset(
        Path("../data/X_train.npy"), Path("../data/Y_train.npy"))

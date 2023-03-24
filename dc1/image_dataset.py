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

    def __init__(self, x: Path, y: Path, transform:Callable = None) -> None:
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



class AugImageDataset(ImageDataset):
    """
    Used for lazy data augmentation with while using the ImageDataset
    """

    def __init__(self, x: Path, y: Path,
                 transform: List[Callable] = None,
                 augmentation_iter: int = 5,
                 device: str = 'cpu') -> None:
        # Create a PyTorch dataset from the X and y arrays
        super().__init__(x,y)
        if transform is None:
            self.transform = [None] + [transforms.Compose([
                transforms.RandomAffine(degrees=5, scale=(0.95, 1.05)),
            ])] * augmentation_iter

        print(len(self.transform))
         
        self.rawlen = len(self.targets)
        self.targets = np.tile(self.targets, augmentation_iter)
        self.device = device


    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        tidx = idx // self.rawlen
        idx = idx % self.rawlen
        transform = self.transform[tidx]
        label = self.targets[idx]

        image = torch.tensor(self.imgs[idx] / 255).float().to(self.device)
        if transform:
            image = transform(image)

        return image, label



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
    cwd = os.getcwd()
    if path.exists(path.join(cwd + "data/")):
        print("Data directory exists, files may be overwritten!")
    else:
        os.mkdir(path.join(cwd, "../data/"))
    ### Load labels
    train_y = load_numpy_arr_from_url(
        url="https://surfdrive.surf.nl/files/index.php/s/i6MvQ8nqoiQ9Tci/download"
    )
    np.save("../data/Y_train.npy", train_y)
    test_y = load_numpy_arr_from_url(
        url="https://surfdrive.surf.nl/files/index.php/s/wLXiOjVAW4AWlXY/download"
    )
    np.save("../data/Y_test.npy", test_y)
    ### Load data
    train_x = load_numpy_arr_from_url(
        url="https://surfdrive.surf.nl/files/index.php/s/4rwSf9SYO1ydGtK/download"
    )
    np.save("../data/X_train.npy", train_x)
    test_x = load_numpy_arr_from_url(
        url="https://surfdrive.surf.nl/files/index.php/s/dvY2LpvFo6dHef0/download"
    )
    np.save("../data/X_test.npy", test_x)

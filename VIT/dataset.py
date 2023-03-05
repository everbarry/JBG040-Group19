from typing import *
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class XrayDataset(Dataset):
    """ pytorch dataset class that returns an iterable object to use for passing to the model"""
    def __init__(self, x, y):
        # normalized target labels
        self.y = XrayDataset.load_np_arr_from_npy(y)
        num_labels = len(np.unique(self.y)) # number of classes
        # Create a one-hot encoded matrix
        self.y = np.eye(num_labels)[self.y]
        # images normalized to 0-1
        self.x = XrayDataset.load_np_arr_from_npy(x)/255

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        image = torch.from_numpy(self.x[idx]).float()
        label = self.y[idx]
        return image, label

    @staticmethod
    def load_np_arr_from_npy(path) -> np.ndarray:
        return np.load(path)

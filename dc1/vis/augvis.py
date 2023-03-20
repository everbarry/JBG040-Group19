from image_dataset import ImageDataset
from net import Net
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

import numpy as np
import matplotlib.pyplot as plt


def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs) -> None:
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)

    if row_title is not None:
        for row_idx in range(1, num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])
    else:
        for col_idx in range(1, num_cols):
            axs[0, col_idx].set(title=f"Transform {col_idx}")
            axs[0, col_idx].title.set_size(8)

    plt.tight_layout()


def visualize_augmentation(augmented:ConcatDataset, nrows=4):
    """
    Visualize the augmentatative transformations.
    """
    ncols = len(augmented.cummulative_sizes)
    samp = np.random.randint(0, augmented.cummulative_sizes[0], size = nrows)
    augsamp = [[augmented.datasets[offset][samp[i]][0][0]
               for offset in range(ncols)]
               for i in range(nrows)]
    plot(augsamp)

    plt.show()

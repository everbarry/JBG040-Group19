import numpy as np
import random
import torch
from dc1.image_dataset import ImageDataset
from typing import Generator, Tuple


class BatchSampler:
    """
    Implements an iterable which given a torch dataset and a batch_size
    will produce batches of data of that given size. The batches are
    returned as tuples in the form (images, labels).
    Can produce balanced batches, where each batch will have an equal
    amount of samples from each class in the dataset. If your dataset is heavily

    imbalanced, this might mean throwing away a lot of samples from
    over-represented classes!
    """

    def __init__(self, batch_size: int, dataset: ImageDataset, balanced: bool = False) -> None:
        self.batch_size = batch_size
        self.dataset = dataset
        self.balanced = balanced
        if self.balanced:
            # Counting the ocurrence of the class labels:
            unique, counts = np.unique(self.dataset.targets, return_counts=True)
            indexes = []
            # Sampling an equal amount from each class:
            for i in range(len(unique)):
                indexes.append(
                    np.random.choice(
                        np.where(self.dataset.targets == i)[0],
                        size=counts.min(),
                        replace=False,
                    )
                )
            # Setting the indexes we will sample from later:
            self.indexes = np.concatenate(indexes)
        else:
            # Setting the indexes we will sample from later (all indexes):
            self.indexes = [i for i in range(len(dataset))]

    def __len__(self) -> int:
        return (len(self.indexes) // self.batch_size) + 1

    def shuffle(self) -> None:
        random.shuffle(self.indexes)

    def __iter__(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        remaining = False
        self.shuffle()
        # Go over the datset in steps of 'self.batch_size':
        for i in range(0, len(self.indexes), self.batch_size):
            # If our current batch is larger than the remaining data, we quit:
            if i + self.batch_size > len(self.indexes):
                remaining = True
                break
            # If not, we yield a complete batch:
            else:
                # Getting a list of samples from the dataset, given the indexes we defined:
                X_batch = [
                    self.dataset[self.indexes[k]][0]
                    for k in range(i, i + self.batch_size)
                ]
                Y_batch = [
                    self.dataset[self.indexes[k]][1]
                    for k in range(i, i + self.batch_size)
                ]
                # Stacking all the samples and returning the target labels as a tensor:
                yield torch.stack(X_batch).float(), torch.tensor(Y_batch).long()
        # If there is still data left that was not a full batch:
        if remaining:
            # Return the last batch (smaller than batch_size):
            X_batch = [
                self.dataset[self.indexes[k]][0] for k in range(i, len(self.indexes))
            ]
            Y_batch = [
                self.dataset[self.indexes[k]][1] for k in range(i, len(self.indexes))
            ]
            yield torch.stack(X_batch).float(), torch.tensor(Y_batch).long()

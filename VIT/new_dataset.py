import numpy as np
from torch.utils.data import Dataset, ConcatDataset

X_train = np.load('../data/X_train.npy')
Y_train = np.load('../data/Y_train.npy')
X_test = np.load('../data/X_test.npy')
Y_test = np.load('../data/Y_test.npy')

unique_labels, counts = np.unique(Y_train, return_counts=True)
class_frequencies = counts / Y_train.size


class AugImageDataset()

def split_dataset_by_class(X, Y):
    class_datasets = {}

    unique_labels = np.unique(Y)

    for label in unique_labels:
        mask = (Y == label)
        class_datasets[label] = (X[mask], Y[mask])

    return class_datasets

train_datasets_by_class = split_dataset_by_class(X_train, Y_train)
test_datasets_by_class = split_dataset_by_class(X_test, Y_test)

for key, value in train_dataset_by_class.items():
    train_datasets_by_class[key] =  AugImageDataset(value, class_frequencies[key])

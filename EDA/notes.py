import numpy as np
import torch

train_y = np.load('../data/Y_train.npy')
test_y = np.load('../data/Y_test.npy')
y = np.concatenate((train_y, test_y))


unique_values, value_counts = np.unique(y, return_counts=True)
l = len(y)
r = []
for value, count in zip(unique_values, value_counts):
    print(f"{value}: {count/l:.2f}")
    r.append(count/l)

t = torch.tensor(r)
x = torch.tensor([0.1494, 0.1400, 0.1782, 0.3573, 0.0966, 0.0785])
print(l)

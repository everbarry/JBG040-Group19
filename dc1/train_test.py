from tqdm import tqdm

import torch
import numpy as np
from sklearn.metrics import confusion_matrix

from net import Net
from batch_sampler import BatchSampler
from typing import Callable, List


def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


def resume(model, filename):
    model.load_state_dict(torch.load(filename))


def train_model(
        model: Net,
        train_sampler: BatchSampler,
        optimizer: torch.optim.Optimizer,
        loss_function: Callable[..., torch.Tensor],
        device: str,
) -> List[torch.Tensor]:
    # Lets keep track of all the losses:
    losses = []
    # Put the model in train mode:
    model.train()
    # Feed all the batches one by one:
    correct = 0
    total = 0
    TP = 0
    FP = 0
    FN = 0

    for batch in tqdm(train_sampler):
        # Get a batch:
        x, y = batch
        # Making sure our samples are stored on the same device as our model:
        x, y = x.to(device), y.to(device)
        # Get predictions:
        predictions = model.forward(x)
        loss = loss_function(predictions, y)
        losses.append(loss)
        # We first need to make sure we reset our optimizer at the start.
        # We want to learn from each batch seperately,
        # not from the entire dataset at once.
        optimizer.zero_grad()
        # We now backpropagate our loss through our model:
        loss.backward()
        # We then make the optimizer take a step in the right direction.
        optimizer.step()
        _, predicted = torch.max(predictions, 1)
        correct += (predicted == y).sum().item()
        total += len(y)
        TP += ((predicted == 1) & (y == 1)).sum().item()
        FP += ((predicted == 1) & (y == 0)).sum().item()
        FN += ((predicted == 0) & (y == 1)).sum().item()
    print(f'correct: {correct}/{total}\nacc: {correct/total:.2f}')
    return losses, correct, total, TP, FP, FN


def test_model(
        model: Net,
        test_sampler: BatchSampler,
        loss_function: Callable[..., torch.Tensor],
        device: str,
) -> List[torch.Tensor]:
    # Setting the model to evaluation mode:
    model.eval()
    losses = []
    # We need to make sure we do not update our model based on the test data:
    correct = 0
    total = 0
    TP = 0
    FP = 0
    FN = 0
    with torch.no_grad():
        for (x, y) in tqdm(test_sampler):
            # Making sure our samples are stored on the same device as our model:
            x = x.to(device)
            y = y.to(device)
            prediction = model.forward(x)
            loss = loss_function(prediction, y)
            losses.append(loss)
            _, predicted = torch.max(prediction, 1)
            correct += (predicted == y).sum().item()
            total += len(y)
            TP += ((predicted == 1) & (y == 1)).sum().item()
            FP += ((predicted == 1) & (y== 0)).sum().item()
            FN += ((predicted == 0) & (y== 1)).sum().item()

    print(f'correct: {correct}/{total}\nacc: {correct / total:.2f}')
    return losses, correct, total, TP, FP, FN


# TODO: generate confusion matrix for classes.
def gen_confusion(
        model: Net,
        test_sampler: BatchSampler,
        device: str,
) -> List[torch.Tensor]:
    model.eval()
    predy = []
    truey = []

    with torch.no_grad():
        for (x, y) in tqdm(test_sampler):
            truey = np.append(truey, y)
            # Making sure our samples are stored on the same device as our model:
            x = x.to(device)
            y = y.to(device)
            prediction = model.forward(x)
            _, predicted = torch.max(prediction, 1)
            predy = np.append(predy, predicted.to('cpu'))

    result = confusion_matrix(truey, predy)
    print(result)
    return result


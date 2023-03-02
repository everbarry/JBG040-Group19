import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
from vis import *

def trainLoop(device, optimizer, criterion, train_loader, model, epochs):
    """ function that trains model on given device with given parameters."""
    for epoch in range(epochs):
        train_losses = []
        correct, total = 0,0

        for batch_idx, batch_sample in enumerate(tqdm(train_loader)):
            # get data from dataloader
            X, y = batch_sample
            X = X.to(device)
            y = y.to(device)
            y = y.long()
            y = y.to(device)
            # forward, backward pass, optimizer step
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()
            _, predicted = torch.max(output, 1)
            correct += (predicted == y).sum().item()
            total += len(y)
        train_loss = float(np.mean(train_losses))
        train_acc = 100 * correct / total
        print('\nTrain set: Epoch: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        epoch+1, train_loss, correct, total, train_acc))
    # save weights
    timestamp = str(datetime.now())[5:19]
    torch.save(model.state_dict(), f"models/Vit-{timestamp}.pt")
    return train_acc

def testLoop(device, model, criterion, optimizer, test_loader):
    """ function that runs the validation over the test set and returns results. """
    # to not train over the test set
    with torch.no_grad():
        test_losses = []
        correct = 0
        for batch_idx, batch_sample in enumerate(tqdm(test_loader)):
            # get data from dataloader
            X, y = batch_sample
            X = X.to(device)
            y = y.to(device)
            y = y.long()
            y = y.to(device)
            # get output and calculate loss
            optimizer.zero_grad()            # avoid gradient accumulation
            output = model(X)                # forward pass
            loss = criterion(output, y)      # compute loss
            test_losses.append(loss.item())  # storing loss

            _, predicted = torch.max(output, 1)
            correct += (predicted == y).sum().item()
        # return results
        test_loss = float(np.mean(test_losses))
        test_acc = 100 * correct / len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(test_acc, correct, len(test_loader.dataset), test_acc))
        return (correct,len(test_loader.dataset), test_acc, test_loss)


def visAttention(device, optimizer, model):
    x = np.load('../data/X_test.npy')/255
    with torch.no_grad():
       image = x[np.random.randint(0,1000)]
       image_tensor = torch.from_numpy(image).float()
       image_tensor = image_tensor.to(device)
       optimizer.zero_grad()
       output, attn_masks = model(image_tensor, getAttention=True)
       _, predicted = torch.max(output, 1)
       # attn_masks = attn_masks.detach().cpu().numpy()
       print(len(attn_masks), len(attn_masks[0]), attn_masks[1][1].max(), attn_masks[1][1].min(), attn_masks[1][1])

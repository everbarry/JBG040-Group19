import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
import wandb


def checkpoint(model, epoch, name):
    #save weights
    timestamp = datetime.now().strftime('%m:%d:%H:%M')
    r = 'ViT-{}-e{}-{}.pth'.format(name, epoch, timestamp)
    torch.save(model.state_dict(), f"models/{r}")
    print(f'checkpointed model: {r}')


def test_loop(device, test_loader, model, optimizer, criterion):
    #test loop
    model.eval()
    with torch.no_grad():
        test_losses = []
        y_true, y_pred = [], []
        correct = 0
        for batch_idx, batch_sample in enumerate(tqdm(test_loader)):
            #get data from dataloader
            X, y = batch_sample
            X = X.to(device)
            y = y.long()
            y = y.to(device)
            optimizer.zero_grad()       # avoid gradient accumulation
            output = model(X)           # forward pass
            loss = criterion(output, y) # compute loss
            test_losses.append(loss.item())
            _, predicted = torch.max(output, 1)
            correct += (predicted == y).sum().item()
            y_true.append(y.cpu().tolist())
            y_pred.append(predicted.cpu().tolist())
        #return results
        test_loss = float(np.mean(test_losses))
        test_acc = 100 * correct / len(test_loader.dataset)
        y_true = [element for sublist in y_true for element in sublist]
        y_pred = [element for sublist in y_pred for element in sublist]
        return test_acc, (y_true, y_pred)


def train_loop(epochs, train_loader, device, optimizer, model, criterion, test_loader, early_stop_thresh, fast=False):
    """train loop if fast=True doesnt run validation every epoch
    """
    #TODO implement fast training, dont log nothing just as fast as possible
    #TODO add timeout after 1hour 45mins
    model.train()
    best = ''
    for epoch in range(epochs):
        dct = {'accuracies': [], 'best_acc': -1, 'best_epoch': 0, 'train_losses': [], 'correct': 0, 'total': 0}
        for _, batch_sample in enumerate(tqdm(train_loader)):
            #get data from dataloader
            X, y = batch_sample
            X = X.to(device)
            y = y.long()
            y = y.to(device)
            optimizer.zero_grad()         # avoid gradient accumulation
            output = model(X)             # forward pass
            loss = criterion(output, y)   # calculate loss
            loss.backward()               # backpropagation
            dct['train_losses'].append(loss.item())
            optimizer.step()
            _, predicted = torch.max(output, 1)
            dct['correct'] += (predicted == y).sum().item()
            dct['total'] += len(y)
        train_loss = float(np.mean(dct['train_losses']))
        train_acc = 100 * dct['correct'] / dct['total']
        #checkpointing training progress every 10 epochs
        if epoch % 10 == 0 and epoch != 0:
            checkpoint(model, epoch, 'cp')
        #early-stopping if test_acc decreases
        test_acc, (y_true, y_pred) = test_loop(device, test_loader, model, optimizer, criterion)
        dct['accuracies'].append(test_acc)
        wandb.log({'train acc': train_acc, 'test acc': test_acc, 'loss': train_loss})
        if dct['accuracies'][-1] > dct['best_acc']:
            dct['best_acc'] = dct['accuracies'][-1]
            dct['best_epoch'] = epoch
            checkpoint(model, epoch, 'best')
            best = 'ViT-best-e{}-{}.pth'.format(epoch, datetime.now().strftime('%m:%d:%H:%M'))
        elif epoch - dct['best_epoch'] == early_stop_thresh:
            print('Terminating training, early stopping model')
            return best
        print(f'Epoch: {epoch}, Average loss: {train_loss:.4f}, Train Accuracy: {dct["correct"]}/{dct["total"]} ({train_acc:.2f})%, Test Acc: ({dct["best_acc"]:.2f})%')
        checkpoint(model, epochs, 'final')
        return best

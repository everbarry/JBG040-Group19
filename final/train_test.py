import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from datetime import datetime, timedelta
import wandb
import torch.optim as optim
from cnnnet import CNNet
from typing import Callable, List
from torch.utils.data import DataLoader
from vis import vis_cm, vis_roc

def checkpoint(model, epoch, name):
    #save weights
    timestamp = datetime.now().strftime('%m:%d:%H:%M')
    r = '{}-{}-e{}-{}.pth'.format(
        "CNN" if isinstance(model, CNNet) else "ViT" ,
        name,
        epoch,
        timestamp)
    torch.save(model.state_dict(), f"models/{r}")
    print(f'checkpointed model: {r}')


def test_loop(device, test_loader, model, optimizer, criterion):
    #test loop
    model.eval()
    with torch.no_grad():
        test_losses = []
        y_true, y_pred, pred_prob = [], [], []
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
            pred_prob.append(output.cpu().numpy())
            y_true.append(y.cpu().tolist())
            y_pred.append(predicted.cpu().tolist())
        test_loss = float(np.mean(test_losses))
        test_acc = 100 * correct / len(test_loader.dataset)
        y_true = [element for sublist in y_true for element in sublist]
        y_pred = [element for sublist in y_pred for element in sublist]
        # Visualisations
        cats = ["Pneumothorax",
               "Nodule",
               "Infiltration",
               "Effusion",
               "Atelectasis",
                "No decease",]
        vis_cm(y_true, y_pred,cats)
        pred_prob = np.concatenate(pred_prob)
        vis_roc(np.asarray(y_true),pred_prob, cats)
        return test_acc, (y_true, y_pred)


def train_loop(epochs, train_loader, device, optimizer, model, criterion, test_loader, early_stop_thresh, timeout, fast=False):
    """train loop if fast=True doesnt run validation every epoch
    """
    #TODO implement fast training, dont log nothing just as fast as possible
    #TODO add timeout after 1hour 45mins
    start = datetime.now()
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
        if timeout:
            now = datetime.now()
            if (now-start) > timedelta(hours=1, minutes=50):
                checkpoint(model, epochs, 'timeout')
                return best
    checkpoint(model, epochs, 'final')
    return best


def demo2(device, test_loader, model):
    threshold = .4
    cats = ["Pneumothorax", "Nodule", "Infiltration", "Effusion", "Atelectasis", "No decease"]
    model = model.to(device)
    data_iter = iter(test_loader)
    random_batch = next(data_iter)
    inputs, _ = random_batch
    inputs = inputs.to(device)
    with torch.no_grad():  # Disable gradient calculations for memory efficiency
        softmax = torch.nn.Softmax(dim=1)
        probabilities = softmax(model(inputs)).cpu().numpy()
    # Visualize the first 10 images and their predicted class probabilities
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        img = (inputs[i].cpu().numpy().transpose((1, 2, 0)) * 0.5) + 0.5
        ax.imshow(img, cmap='gray' if img.shape[2] == 1 else None)
        title_text = "\n".join([f"{cats[j]}: {probabilities[i][j]:.2f}"
                            for j in range(len(probabilities[i]))])
        flag = any(prob > threshold for prob in probabilities[i][:-1])
        flag_color = 'red' if flag else 'black'
        title_text += f"\nFurther screening: {flag}"
        ax.set_title(title_text, fontsize=10)
        ax.text(0, -30, title_text, color=flag_color, fontsize=10, transform=ax.transAxes)
        ax.axis("off")
    plt.show()


def demo(device, test_loader, model):
    threshold = .4
    cats = ["Pneumothorax", "Nodule", "Infiltration", "Effusion", "Atelectasis", "No decease"]
    model = model.to(device)
    data_iter = iter(test_loader)
    random_batch = next(data_iter)
    inputs, _ = random_batch
    inputs = inputs.to(device)
    with torch.no_grad():  # Disable gradient calculations for memory efficiency
        softmax = torch.nn.Softmax(dim=1)
        probabilities = softmax(model(inputs)).cpu().numpy()
    # Visualize the first 10 images and their predicted class probabilities
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        img = (inputs[i].cpu().numpy().transpose((1, 2, 0)) * 0.5) + 0.5
        ax.imshow(img, cmap='gray' if img.shape[2] == 1 else None)
        title_text = "\n".join([f"{cats[j]}: {probabilities[i][j]:.2f}"
                            for j in range(len(probabilities[i]))])
        ax.set_title(title_text, fontsize=10, pad=10)

        flag = any(prob > threshold for prob in probabilities[i][:-1])
        flag_color = 'red' if flag else 'black'
        flag_text = f"Further screening: {flag}"
        ax.text(0, -0.1, flag_text, color=flag_color, fontsize=10, transform=ax.transAxes)

        ax.axis("off")
    plt.show()



class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """
    TODO: use this with adam
    """
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
            return lr_factor

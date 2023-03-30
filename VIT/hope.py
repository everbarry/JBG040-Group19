import numpy as np
from tqdm import tqdm
import timm
import wandb
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchsummary import summary
from pathlib import Path
from image_dataset import AugImageDataset
from sklearn.metrics import confusion_matrix, cohen_kappa_score


#TODO: clean up code, document, split in modules, add f1 score & plot, add runtime inference, add parser, stop training before 2 hours.

# seed everything
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(13)
torch.manual_seed(13)


#hyperparams
BATCH_SIZE = 256
LEARNING_RATE = 3e-5
N_EPOCHS = 50
early_stop_thresh = 4
DROPOUT = 0.4
ATTN_DROPOUT = 0.1
DEPTH=10


model = timm.create_model(
    'vit_tiny_patch16_224',  # model architecture
    pretrained=False,        # weights randomly initialized
    num_classes=6,           # number of classes
    img_size=128,            # img size
    in_chans=1,              # number of channels of image
    drop_rate=DROPOUT,       # dropout rate
    depth=DEPTH
)

#model.load_state_dict(torch.load('models/ViT-final-e70-03:29:05:08.pth'))
model.to(device)
#summary(model, tuple([1,1,128,128]), device=str(device))
# list models
#print(timm.list_models('*vit_tiny*'))



#wandb stuff
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="ViT",
    # track hyperparameters and run metadata
    config={
    "learning_rate": LEARNING_RATE,
    "architecture": "vit_tiny_patch16_224",
    "dataset": "XRay Dataset",
    "epochs": N_EPOCHS,
    "dropout": DROPOUT,
        "depth": DEPTH}
)


#data
train_dataset = AugImageDataset(Path('../data/X_train.npy'), Path('../data/Y_train.npy'))
test_dataset = AugImageDataset(Path('../data/X_test.npy'), Path('../data/Y_test.npy'))
train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset)//BATCH_SIZE, shuffle=True, num_workers=8)
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset)//BATCH_SIZE, shuffle=True, num_workers=8)


#model stuff
criterion = CrossEntropyLoss()
#criterion = CrossEntropyLoss(torch.Tensor([0.1496942,  0.13764028, 0.17599905, 0.36238941, 0.09696574, 0.07731132]).to(device))
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)


#training stuff
def checkpoint(model, epoch, name):
    #save weights
    timestamp = datetime.now().strftime('%m:%d:%H:%M')
    r = 'ViT-{}-e{}-{}.pth'.format(name, epoch, timestamp)
    torch.save(model.state_dict(), f"models/{r}")
    print(f'checkpointed model: {r}')


def test_loop(device, test_loader, model, optimizer, criterion):
    #test loop
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
        #print(f'Test set: Average loss: {test_acc:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_acc:.2f})')
        return test_acc, (y_true, y_pred)


#train loop
for epoch in range(N_EPOCHS):
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
    elif epoch - dct['best_epoch'] == early_stop_thresh:
        print('Terminating training, early stopping model')
        break
    print(f'Epoch: {epoch}, Average loss: {train_loss:.4f}, Train Accuracy: {dct["correct"]}/{dct["total"]} ({train_acc:.2f})%, Test Acc: ({dct["best_acc"]:.2f})%')
checkpoint(model, N_EPOCHS, 'final')


# plot results
y_true = [element for sublist in y_true for element in sublist]
y_pred = [element for sublist in y_pred for element in sublist]
print(f'\nCohens Kappa coeff: {cohen_kappa_score(y_true, y_pred)}')
print(f'\nConfusion Matrix:\n{confusion_matrix(y_true, y_pred)}\n')

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18,10))
# create loss & accuracy plot
axs[0].plot(range(len(dct['accuracies'])), dct['accuracies'], label='Test Acc')
axs[1].plot(range(len(dct['train_losses'])), dct['train_losses'], label='Train Loss')


# Set the titles of each subplot
axs[0].set_title('Test Accuracy during training')
axs[1].set_title('Training Loss')

# Set the axis labels of each subplot
axs[0].set_xlabel('epoch*n_batches')
axs[0].set_ylabel('Test set accuracy')
axs[1].set_xlabel('epoch*n_batches')
axs[1].set_ylabel('Training loss')
plt.tight_layout()
plt.show()
plt.savefig('results.svg', format='svg')
wandb.finish()

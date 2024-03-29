import numpy as np
from torch.nn import BCEWithLogitsLoss
import wandb
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
import timm
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score
import os
import requests
import io
import argparse
# import modules
from image_dataset import AugImageDataset
from new_train_test import train_loop, test_loop
from model import CosineWarmupScheduler

def initEnv(random_seed=19):
    """init device, sets random seeed for reproducibilit.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# if cuda gpu available use that
    # support Apple Silicon
    device = torch.device('mps' if torch.backends.mps.is_available() else device)
    # set fixed random seed for repeatability
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    print(f'initializing model using {device}')
    return device


def checkData():
    """check if .npy data is present on machine, if not download it.
    """
    file_list = ["X_train.npy", "X_test.npy", "Y_train.npy", "Y_test.npy"]
    if any(not os.path.isfile(os.path.join('../data/', file)) for file in file_list):
        urls = ["https://surfdrive.surf.nl/files/index.php/s/4rwSf9SYO1ydGtK/download",
                "https://surfdrive.surf.nl/files/index.php/s/dvY2LpvFo6dHef0/download",
                "https://surfdrive.surf.nl/files/index.php/s/i6MvQ8nqoiQ9Tci/download",
                "https://surfdrive.surf.nl/files/index.php/s/wLXiOjVAW4AWlXY/download"]
        for url, file in zip(urls, file_list):

            np.save(os.path.join('../data/', file), load_numpy_arr_from_url(url))


def load_numpy_arr_from_url(url: str) -> np.ndarray:
    """Loads a numpy array from surfdrive url.
    """
    response = requests.get(url)
    response.raise_for_status()
    return np.load(io.BytesIO(response.content))


def parser():
    """function that parses command line arguments and returns them as dict"""
    # create parser object
    parser = argparse.ArgumentParser(prog = 'ViT Vision transformer',
                    description = 'alternative model trained on the same Xray dataset',
                    epilog = 'group 19 DBL')
    # add parser arguments
    parser.add_argument('-t', '--train', help='if specified model is trained with given hyperparams', action='store_true')
    parser.add_argument('-m', '--model', help='specify model to train, to specify only if training the model', choices=['CNN', 'ViT'], default='ViT')
    parser.add_argument('-e', '--epochs', help='specify number of epochs', default=20, type=int)
    parser.add_argument('-l', '--learning-rate', help='specify learning rate', default=3e-5, type=float)
    parser.add_argument('-p', '--patch-size', help='specify size of patch to section image,\n e.g: 8 = divide 128x128 image in 8x8 patches', default=8, type=int)
    parser.add_argument('-d', '--depth', help='specify number of layers in the hidden dimension of the model', default=10, type=int)
    parser.add_argument('-o', '--optimizer', help='speecify optimizer to use with the model', choices=['SGD','Adam', 'RAdam'], default='Adam')
    parser.add_argument('-w', '--n-workers', help='specify number of CPU threads to load data', default=8, type=int)
    parser.add_argument('-st', '--early-stop-thresh', help='threshold of number of epochs to early stop training if test acc doesnt improve', default=5, type=int)
    parser.add_argument('-b', '--n-batches', help='specify number of batches to split the dataset into', default=256, type=int)
    parser.add_argument('-s', '--scheduler', help='specify the learning rate scheduler to use', choices=['CosineWarmup', 'StepLR'], default='StepLR')
    parser.add_argument('-dr', '--drop-rate', help='specify drop rate from model (.0-.99)', default=0.35, type=float)
    parser.add_argument('-c', '--criterion', help='specify which loss function to use', choices=['CrossEntropy', 'BCELoss'], default='CrossEntropy')
    return parser.parse_args()


def getData(batch_size, weights):
    """creates torch Dataframe, Weighted Samples the data to balance batches and return DataLoader ready for training.
    """
    #TODO hardcoded paths
    # Create DataSet objs
    train_dataset = AugImageDataset(Path('../data/X_train.npy'), Path('../data/Y_train.npy'))
    test_dataset = AugImageDataset(Path('../data/X_test.npy'), Path('../data/Y_test.npy'))
    # Create the WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    # Create DataLoader objs
    train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset)//batch_size, num_workers=8, shuffle=True)#sampler=sampler)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset)//batch_size, shuffle=True, num_workers=8)
    return train_loader, test_loader


def getClassWeights():
    """loads numpy array of labels and return class weights
    """
    #TODO hardcoded paths
    arr = np.load('../data/Y_train.npy')
    _, counts = np.unique(arr, return_counts=True)   # Calculate unique class labels and their counts
    weights = arr.size / counts                      # Compute class weights
    return weights


def runResults(model_name, dropout, depth, device, test_loader, criterion, weights, checkpoint):
    """runs validation and metrics on given model settings
    """
    # create model
    if model_name == 'ViT':
        #TODO integrate timm in project
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=6, img_size=128,
                                in_chans=1, drop_rate=dropout)
        model.load_state_dict(torch.load(f'models/{checkpoint}'))
        model.to(device)

    elif model_name == 'CNN':
        #TODO integrate CNN
        raise NotImplementedError
    # create optimizer, will not be used anyways as we are not computing gradients
    optimizer = Adam(model.parameters(), lr=0.001)
    # create loss function
    match criterion:
        case 'CrossEntropy':
            criterion = CrossEntropyLoss(torch.Tensor(weights).to(device))
        case 'BCELoss':
            criterion = BCEWithLogitsLoss(pos_weight=torch.Tensor(weights).to(device))
    acc, (y_true, y_pred) = test_loop(device, test_loader, model, optimizer, criterion)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    f1, f1_balanced, f1_per_class = f1_score(y_true, y_pred, average='macro'), f1_score(y_true, y_pred, average='weighted'), f1_score(y_true, y_pred, average=None)
    print(f'\n {"-"*20}\nTest set accuracy: {acc}')
    print(f'unbalanced F1 score: {f1}')
    print(f'balanced F1 score: {f1_balanced}')
    print(f'per class F1 score: {f1_per_class}')
    print(f'\nCohens Kappa coeff: {kappa}')
    print(f'\nConfusion Matrix:\n{cm}\n')



def trainModel(model_name, device, dropout, optimizer, lr, scheduler, criterion, weights, epochs, train_loader, test_loader, early_stop_thresh, checkpoint=None):
    """trains model given settings.
    """
    # create model
    if model_name == 'ViT':
        #TODO integrate timm in project
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=6, img_size=128,
                                in_chans=1, drop_rate=dropout)
        if checkpoint is not None:
            model.load_state_dict(torch.load(f'models/{checkpoint}'))
        model.to(device)
    elif model_name == 'CNN':
        #TODO integrate CNN
        raise NotImplementedError
    # create optimizer
    match optimizer:
        case 'Adam':
            optimizer = Adam(model.parameters(), lr=lr)
        case 'SGD':
            optimizer = SGD(model.parameters(), lr=lr, momentum=.1)
        case 'RAdam':
            raise NotImplementedError
    # create learning rate scheduler
    match scheduler:
        case 'CosineWarmup':
            scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=10, max_iters=200)
        case 'StepLR':
            scheduler =  StepLR(optimizer, step_size=1, gamma=.7)
    # create loss function
    match criterion:
        case 'CrossEntropy':
            criterion = CrossEntropyLoss(torch.Tensor(weights).to(device))
        case 'BCELoss':
            criterion = BCEWithLogitsLoss(pos_weight=torch.Tensor(weights).to(device))
    # train model
    best_model = train_loop(epochs, train_loader, device, optimizer, model, criterion, test_loader, early_stop_thresh, fast=False)
    return best_model


def main():
    """main that runs the hole project
    """
    args = parser()
    device = initEnv()

    checkData()
    weights = getClassWeights()
    train_loader, test_loader = getData(args.n_batches, weights)

    if args.train is False:
        #TODO hardcoded weights loading
        runResults('ViT', args.drop_rate, args.depth, device, test_loader, args.criterion, weights, 'ViT-final-e40-03:28:21:50.pth')
        runResults('CNN', args.drop_rate, args.depth, device, test_loader, args.criterion, weights, '')
        return
    else:
        # set the wandb project where this run will be logged
        wandb.init(project="dc1", config={"learning_rate": args.learning_rate, "architecture": args.model,
                                          "dataset": "XRay Dataset", "epochs": args.epochs,
                                          "dropout":args.drop_rate, "depth": args.depth})
        best_model = trainModel(args.model, device, args.drop_rate, args.optimizer, args.learning_rate, args.scheduler, args.criterion, weights, args.epochs, train_loader, test_loader, args.early_stop_thresh)
        runResults('ViT', args.drop_rate, args.depth, device, test_loader, args.criterion, weights, best_model)
        wandb.finish()
        return

if __name__ == '__main__' :
    main()

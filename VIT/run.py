import argparse
import io
import requests
import os
import numpy as np
import torch
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss
from torchsummary import summary
from torch.utils.data import DataLoader
# import modules
from dataset import XrayDataset
from model import VisionTransformer, CosineWarmupScheduler
from train_test import trainLoop, testLoop, visAttention


def initEnv(random_seed=19):
    """ function that initializes and return device, sets random seeed for repeatable results. """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# if cuda gpu available use that
    # support Apple Silicon
    # if torch.backends.mps.is_available():
    #     device = 'mps'
    # set fixed random seed for repeatability
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    print(f'initializing model using {device}')
    # check data
    checkData()
    return device


def checkData():
    """ check if .npy data is presento on machine, if not  download it."""
    flag = False
    file_list = ["X_train.npy", "X_test.npy", "Y_train.npy", "Y_test.npy"]
    for file in file_list:
        file_path = os.path.join('../data/', file)
        if not os.path.isfile(file_path):
            flag = True
    if flag:
        ### Load labels
        train_y = load_numpy_arr_from_url(url="https://surfdrive.surf.nl/files/index.php/s/i6MvQ8nqoiQ9Tci/download")
        np.save("../data/Y_train.npy", train_y)
        test_y = load_numpy_arr_from_url(url="https://surfdrive.surf.nl/files/index.php/s/wLXiOjVAW4AWlXY/download")
        np.save("../data/Y_test.npy", test_y)
        ### Load data
        train_x = load_numpy_arr_from_url(url="https://surfdrive.surf.nl/files/index.php/s/4rwSf9SYO1ydGtK/download")
        np.save("../data/X_train.npy", train_x)
        test_x = load_numpy_arr_from_url(url="https://surfdrive.surf.nl/files/index.php/s/dvY2LpvFo6dHef0/download")
        np.save("../data/X_test.npy", test_x)


def load_numpy_arr_from_url(url: str) -> np.ndarray:
    """
    Loads a numpy array from surfdrive.
    Input:
    url: Download link of dataset
    Outputs:
    dataset: numpy array with input features or labels
    """
    response = requests.get(url)
    response.raise_for_status()
    return np.load(io.BytesIO(response.content))



def parser():
    """function that parses command line arguments and returns them"""
    # create parser object
    parser = argparse.ArgumentParser(prog = 'ViT Vision transformer',
                    description = 'alternative model trained on the same Xray dataset',
                    epilog = 'group 19 DBL')
    # add parser arguments
    parser.add_argument('-e', '--epochs', help='specify number of epochs', default=2, type=int)
    parser.add_argument('-l', '--learning-rate', help='specify learning rate', default=0.001, type=float)
    parser.add_argument('-p', '--patch-size', help='specify size of patch to section image,\n e.g: 8 = divide 128x128 image in 8x8 patches', default=8, type=int)
    parser.add_argument('-d', '--hidden-dim', help='specify number of layers in the hidden dimension of every transformer head', default=15, type=int)
    parser.add_argument('-o', '--optimizer', help='speecify optimizer to use with the model', choices=['SGD','Adam', 'RAdam'], default='Adam')
    parser.add_argument('-w', '--n-workers', help='specify number of CPU threads to load data', default=6, type=int)
    parser.add_argument('-b', '--n-batches', help='specify number of batches to split the dataset into', default=64, type=int)
    parser.add_argument('-s', '--model-summary', help='if specified model summary will be displayed', action='store_true')
    parser.add_argument('-v', '--val', help='if specified load best weights, validate model and display visualizations', action='store_true')
    parser.add_argument('-ws', '--warmup-scheduler', help='specify the number of batches of warmup for the learning rate scheduler', default=30, type=int)
    # return arguments
    return parser.parse_args()


def main():
    """ void main function that creates, trains and evaluates the model with user selected args. """
    # hyperparams
    num_heads=3
    output_dim=6
    input_size=[1,128,128]
    args = parser()
    device = initEnv()

    # get data
    train_dataset = XrayDataset('../data/X_train.npy', '../data/Y_train.npy')
    test_dataset = XrayDataset('../data/X_test.npy', '../data/Y_test.npy')
    train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset)//args.n_batches+1, shuffle=True, num_workers=args.n_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset)//args.n_batches+1, shuffle=True, num_workers=args.n_workers)

    # init model
    model = VisionTransformer(input_size, [args.patch_size,args.patch_size], args.hidden_dim, num_heads, output_dim)
    model = model.to(device)
    # criterion = CrossEntropyLoss()
    pos_weight = torch.tensor([0.1494, 0.1400, 0.1782, 0.3573, 0.0966, 0.0785]).to(device)  # All weights are equal to 1
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(),lr=args.learning_rate)
    elif args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.1)
    elif args.optimizer == 'RAdam':
        raise NotImplementedError
        # TODO implement adam warmup for better training
        # p = torch.nn.Parameter(torch.empty(4,4))
        # optimizer = Adam([p], lr=1e-3)
        # lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=10, max_iters=200)
    else:
        print('invalid optimizer')
        quit()
    scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=args.warmup_scheduler, max_iters=2000)
    if args.model_summary:
        summary(model, tuple(input_size), device=str(device))

    if args.val:
        weights = torch.load('models/test.pt', map_location=device)
        # Load the weights into the model
        model.load_state_dict(weights)
        print('Loaded pretrained weights...')
        #visAttention(device, optimizer, model)
        results = testLoop(device, model, criterion, optimizer, test_loader)
        return
        # train & test model
    train_accuracy = trainLoop(device, optimizer, scheduler, criterion, train_loader, model, args.epochs)
    results = testLoop(device, model, criterion, optimizer, test_loader)

if __name__ == '__main__':
    main()

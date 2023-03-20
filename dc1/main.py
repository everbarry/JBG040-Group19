# Custom imports
from batch_sampler import BatchSampler
from image_dataset import ImageDataset, AugImageDataset
from net import Net
from train_test import train_model, test_model

# Visualizations
from vis.modelvis import modelvis
from vis.augvis import visualize_augmentation

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary  # type: ignore

# Other imports
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import figure
import os
import argparse
import plotext  # type: ignore
from datetime import datetime
from pathlib import Path
from typing import List


def checkpoint(model: Net) -> str:
    """
    Checkpoint the model and return the filename of the checkpoints.
    """
    now = datetime.now()
    if not Path("model_weights/").exists():
        os.mkdir(Path("model_weights/"))
    filename = f"model_weights/model_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.txt"
    torch.save(model.state_dict(), filename)
    return filename


def resume(model: Net, filename: str) -> None:
    model.load_state_dict(torch.load(filename))


def main(args: argparse.Namespace, activeloop: bool = True) -> None:

    # Load the train and test data set
    if args.augiter > 1:
        train_dataset = AugImageDataset(
            Path("../data/X_train.npy"), Path("../data/Y_train.npy"),
            augmentation_iter=args.augiter)
    else:
        train_dataset = ImageDataset(
            Path("../data/X_train.npy"), Path("../data/Y_train.npy"))
            
    test_dataset = ImageDataset(
        Path("../data/X_test.npy"), Path("../data/Y_test.npy"))

    # Visualize dataset
    if args.augvis:
        visualize_augmentation(train_dataset)

    # Load the Neural Net. NOTE: set number of distinct labels here
    model = Net(n_classes=6)

    # Initialize optimizer(s) and loss function(s)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)
    loss_function = nn.CrossEntropyLoss()

    # fetch epoch and batch count from arguments
    n_epochs = args.nb_epochs
    batch_size = args.batch_size

    # IMPORTANT! Set this to True to see actual errors regarding
    # the structure of your model (GPU acceleration hides them)!
    # Also make sure you set this to False again for actual model training
    # as training your model with GPU-acceleration (CUDA/MPS) is much faster.
    DEBUG = False

    # Moving our model to the right device (CUDA will speed training up significantly!)
    if torch.cuda.is_available() and not DEBUG:
        print("@@@ CUDA device found, enabling CUDA training...")
        device = "cuda"
        model.to(device)
        # Creating a summary of our model and its layers:
        summary(model, (1, 128, 128), device=device)
    elif (
        torch.backends.mps.is_available() and not DEBUG
    ):  # PyTorch supports Apple Silicon GPU's from version 1.12
        print("@@@ Apple silicon device enabled, training with Metal backend...")
        device = "mps"
        model.to(device)
    else:
        print("@@@ No GPU boosting device found, training on CPU...")
        device = "cpu"
        # Creating a summary of our model and its layers:
        summary(model, (1, 128, 128), device=device)

    # Lets now train and test our model for multiple epochs:
    train_sampler = BatchSampler(
        batch_size=batch_size, dataset=train_dataset, balanced=args.balanced_batches
    )
    test_sampler = BatchSampler(
        batch_size=100, dataset=test_dataset, balanced=args.balanced_batches
    )

    mean_losses_train: List[torch.Tensor] = []
    mean_losses_test: List[torch.Tensor] = []
    correct_tr = 0
    total_tr = 0
    correct_test = 0
    total_test = 0
    early_stop_thresh = args.early_stop_thresh
    best_checkpoint = ""
    best_accuracy = -1
    best_epoch = -1
    total_tp_tr = 0
    total_fp_tr = 0
    total_fn_tr = 0
    total_tp_test = 0
    total_fp_test = 0
    total_fn_test = 0
    for e in range(n_epochs):
        if activeloop:

            # Training:
            a_0, a_1, a_2, a_3, a_4, a_5 = train_model(model, train_sampler, optimizer, loss_function, device)
            # Calculating and printing statistics:
            losses = a_0
            mean_loss = sum(losses) / len(losses)
            mean_losses_train.append(mean_loss)
            print(f"\nEpoch {e + 1} training done, loss on train set: {mean_loss}\n")
            correct_tr += a_1
            total_tr += a_2
            total_tp_tr += a_3
            total_fp_tr += a_4
            total_fn_tr += a_5
            # Testing:
            b_0, b_1, b_2, b_3, b_4, b_5 = test_model(model, test_sampler, loss_function, device)
            losses = b_0
            # # Calculating and printing statistics:
            mean_loss = sum(losses) / len(losses)
            mean_losses_test.append(mean_loss)
            print(f"\nEpoch {e + 1} testing done, loss on test set: {mean_loss}\n")
            correct_test += b_1
            total_test += b_2
            acc = b_1/ b_2
            total_tp_test += b_3
            total_fp_test += b_4
            total_fn_test += b_5
            if acc > best_accuracy:
                best_accuracy = acc
                best_epoch = e
                best_checkpoint = checkpoint(model)
            elif e - best_epoch > early_stop_thresh:
                print("Early stopped training at epoch %d" % e)
                n_epochs = e + 1
                break  # terminate the training loop

            ### Plotting during training
            plotext.clf()
            plotext.scatter(mean_losses_train, label="train")
            plotext.scatter(mean_losses_test, label="test")
            plotext.title("Train and test loss")

            plotext.xticks([i for i in range(len(mean_losses_train) + 1)])

            plotext.show()

    resume(model, best_checkpoint)
    print(f'correct: {correct_tr}/{total_tr}\nacc: {correct_tr/total_tr:.2f}')
    print(f'correct: {correct_test}/{total_test}\nacc: {correct_test / total_test:.2f}')
    # retrieve current time to label artifacts
    now = datetime.now()
    #Precision = TruePositives / (TruePositives + FalsePositives)
    precision_tr = total_tp_tr / (total_fp_tr + total_tp_tr)
    print(f'precision training: {precision_tr:.2f}')

    precision_test = total_tp_test / (total_fp_test + total_tp_test)
    print(f'precision training: {precision_test:.2f}')

    recall_tr = total_tp_tr / (total_tp_tr + total_fn_tr)
    recall_test = total_tp_test / (total_fp_test + total_fn_test)
    #F1 = 2 x [(Precision x Recall) / (Precision + Recall)]
    f1_test = 2 * [(precision_tr * recall_tr) / (precision_tr + recall_tr)]
    f1_train = 2 * [(precision_test * recall_test) / (precision_test + recall_test)]
    print(f'f1 score training: {f1_train:.2f}')
    print(f'f1 score testing: {f1_test:.2f}')

    # Create plot of losses
    figure(figsize=(9, 10), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    
    ax1.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_train], label="Train", color="blue")
    ax2.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_test], label="Test", color="red")
    fig.legend()
    
    # Check if /artifacts/ subdir exists
    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))

    # save plot of losses
    fig.savefig(Path("artifacts") / f"session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nb_epochs", help="number of training iterations", default=10, type=int
    )
    parser.add_argument("--batch_size", help="batch_size", default=25, type=int)
    parser.add_argument(
        "--balanced_batches",
        help="whether to balance batches for class labels",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "-p", "--plot",
        help="Plot during training",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-v", "--modelvis",
        help="Enable model vis. Open a model visualization in the browser (default: False)",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-A", "--showattrib",
        help="Show attributes during model vis (default: False)",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-c", "--showsaved",
        help="Show saved tensors during model vis (default: True)",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--augvis",
        help="Augmentation visualization",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-a", "--augiter",
        help="Augmentation itterations. Expands Training data set x fold.",
        default=5,
        type=int
    )
    parser.add_argument(
        "-s", "--early_stop_thresh",
        help="Epoch threshhold for early stopping",
        default=5,
        type=int
    )


    args = parser.parse_args()

    main(args)

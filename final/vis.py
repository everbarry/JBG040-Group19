import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

import os
from datetime import datetime
from pathlib import Path

try:
    import wandb
except Exception as e:
    print("Warning: Invironment is not configured with weights and biases")

def vis_roc(true_labels, pred_prob, categories, show=False):
    """
    Visualization of the ROC curves.
    """
    one_hot_encode = lambda y, n_classes: np.eye(n_classes)[y]
    fpr = dict()
    tpr = dict()
    # pred_prob = one_hot_encode(pred_prob,6)

    plt.figure()
    for i,cat in enumerate(categories):
        # if wandb.run is not None:
        #     wandb.log({"roc": wandb.plot.roc_curve(np.asarray(true_labels == i,dtype="int"), pred_prob[:, i])})
        # Plot the ROC curves
        fpr, tpr, _ = roc_curve(true_labels == i, pred_prob[:, i])
        plt.step(fpr, tpr, label=cat)
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Multiclass ROC curve')
    plt.legend()


    # Check if /artifacts/ subdir exists
    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))
    #save the figure at the end of the  
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"artifacts/roc-{timestamp}.png")

    if show:
        plt.show()


def vis_cm(ground_truth, predictions, class_names, show=False):
    """
    Visualization of the Convolution Matrix.
    """
    if wandb.run is not None:
        cm = wandb.plot.confusion_matrix(
            y_true=ground_truth,
            preds=predictions,
            class_names=class_names)
        
        wandb.log({"conf_mat": cm})
 

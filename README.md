# Data Challenge 1
This repository contains the template code for the TU/e course JBG040 Data Challenge 1.
Please read this document carefully as it has been filled out with important information on how to run the code.

## Get Started .
to get started working on this open a terminal, type in `cd <theDirectory/youWant/toWorkin>`, then only the first time you download the repo do:
```
git clone https://github.com/everbarry/JBG040-Group19.git
cd JBG040-Group19/
conda create --name dbl python=3.10 
conda activate dbl 
conda install --file requirements.txt
```
if the last command fails, try executing: `pip install -r requirements.txt`
this downloads the repository, creates a new environment for this project and then installs all the dependencies.
Now anytime you want to run anything related to this project before make sure you are in the correct environment by typing in the terminal `conda activate dbl`, once activated navigate to the `JBG040-Group19` folder with `cd /path/to/JBG040-Group19`.
Now to download the data in that terminal run:
```
cd final
python3 main.py
```
this runs both models (CNN, ViT) on the test set with our best pretrained weights and downloads the data to the `data` folder 
for more informaton about the ViT see the README in its folder

## Repository structure
The repo is split in the `dc1` folder, used for the CNN developement, the `VIT` folder used for ViT developement and the `final` folder which contains the final deliverable.
#### DC1
- `batch_sampler.py`: contains batch sampler to samples uniformely distributed sample from data
- `cross_val_test.py`: used for testing cross validation
- `image_dataset.py`: contains DataSet class that reads data from `.npy` files in the `../data` folder
- `main.py`: used to run the training and testing loop of the CNN
- `net.py`: contains Class where CNN is defined 
- `train_test.py`: contains train, test functions

#### VIT 
- `image_dataset.py`: dataset class, similair to the one in DC1
- `model.py`: contains version of ViT Class
- `hope.py`: new, faster training pipeline, barebones training script used to train the final version of VIT
-  `train_test.py`: contains train, test functions
- 
The template code is structured into multiple files, based on their functionality. 
There are five `.py` files in total, each containing a different part of the code. 
Feel free to create new files to explore the data or experiment with other ideas.

### final
- `models/` folder: contains models weights
- `Cohen KAPPA.ipynb`: includes analysis of cohens kappa coefficient
- `cnnnet.py`: contains class for the CNN
- `image_dataset`: dataset class, with data augmentation
- `train_test.py`: contains train, test functions
- `main.py`: main script to train, eval and compare models, all with cli flags

## how to run
open the command prompt and change to the `final` directory containing the main.py file.
For example, if you're main file is in C:\Data-Challenge-1-template-main\final\, 
type `cd C:\Data-Challenge-1-template-main\final\` into the command prompt and press enter.

Then, main.py can be run by, for example, typing `python main.py --epochs 10 --balanced_batches`.
This would run the script with 10 epochs, a batch size of 25, and balanced batches, which is also the current default.
If you would want to run the script with 20 epochs, a batch size of 5, and batches that are not balanced, 
you would type `main.py --nb_epochs 20 --batch_size 5 --no-balanced_batches`.
Here are all the command line arguments:
**Usage**: `ViT Vision transformer [-h] [-t] [-m {CNN,ViT}] [-e EPOCHS] [-l LEARNING_RATE] [-p PATCH_SIZE] [-d DEPTH] [-o {SGD,Adam,RAdam}] [-w N_WORKERS] [-st EARLY_STOP_THRESH] [-b N_BATCHES] [-s {CosineWarmup,StepLR}] [-dr DROP_RATE] [-c {CrossEntropy,BCELoss}] [-cw] [-bb] [-to]`

**Description**: alternative model trained on the same Xray dataset

**Options**:
- `-h, --help`: show this help message and exit
- `-t, --train`: if specified model is trained with given hyperparams
- `-m {CNN,ViT}, --model {CNN,ViT}`: specify model to train, to specify only if training the model
- `-e EPOCHS, --epochs EPOCHS`: specify number of epochs
- `-l LEARNING_RATE, --learning-rate LEARNING_RATE`: specify learning rate
- `-p PATCH_SIZE, --patch-size PATCH_SIZE`: specify size of patch to section image, e.g: 8 = divide 128x128 image in 8x8 patches
- `-d DEPTH, --depth DEPTH`: specify number of layers in the hidden dimension of the model
- `-o {SGD,Adam,RAdam}, --optimizer {SGD,Adam,RAdam}`: specify optimizer to use with the model
- `-w N_WORKERS, --n-workers N_WORKERS`: specify number of CPU threads to load data
- `-st EARLY_STOP_THRESH, --early-stop-thresh EARLY_STOP_THRESH`: threshold of number of epochs to early stop training if test acc doesnt improve
- `-b N_BATCHES, --n-batches N_BATCHES`: specify number of batches to split the dataset into
- `-s {CosineWarmup,StepLR}, --scheduler {CosineWarmup,StepLR}`: specify the learning rate scheduler to use
- `-dr DROP_RATE, --drop-rate DROP_RATE`: specify drop rate from model (.0-.99)
- `-c {CrossEntropy,BCELoss}, --criterion {CrossEntropy,BCELoss}`: specify which loss function to use
- `-cw, --class-weights`: if specified data class weights are used to balance the loss function
- `-bb, --balance-batches`: if specified WeightedRandomSampler is used to sample from data to uniformize class distribution.
- `-to, --timeout`: if specified training is stopped after 1 hour 50 mins

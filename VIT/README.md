# ViT - Transformer for CV classification
this is an alternative model we are testing for comparison to the given model.

## Code structure
this template code is structured into multiple files, based on their functionality.
- dataset: pytorch dataset class used to load data from ```data``` folder
- eda: some snippets of exploratory data analysis
- main: old way to run model
- model: pytorch model class
- run: new and correct way to run model
- train_test: functions to run train and test loops

**Note**: before running the model be sure you have already downloaded the data. to do that refer to the main documentation

- To run the whole training/evaluation pipeline: run ```run.py```. This script does the following:
  - Load your train and test data
  - Initializes the neural network as defined in the ```model.py``` file.
  - Initialize loss functions and optimizers. If you want to change the loss function/optimizer, use the run command line flags or change them in the ```run.py``` file.
  - Define number of training epochs and batch size with the command line flags
  - Check and enable GPU acceleration for training (if you have CUDA enabled device)
  - Train the neural network and perform evaluation on test set at the end of training.
  - Provide results about the training losses both during training in the command line.
  - Finally, save your trained model's weights in the /models/ subdirectory so that you can reload them later.

## Argparse
Argparse functionality is included in the run.py file. This means the file can be run from the command line while passing arguments to the main function.
To make use of this functionality, first open the command prompt and change to the directory containing the run.py file.
For example, if you're main file is in ```C:\Data-Challenge-1-template-main\VIT```,
type ```cd C:\Data-Challenge-1-template-main\VIT\``` into the command prompt and press enter.
Then, run.py can be run by, for example, typing ```python3 run.py --n-epochs 3 --batch-size 28.```
This would run the script with 3 epochs, a batch size of 28, which is also the current default.
If you would want to run the script with 20 epochs, a batch size of 5, you would type ```python3 run.py --n_epochs 20 --batch_size 5```

these are all command line argument options:

- -h, --help                                           show this help message and exit
- -e EPOCHS, --epochs EPOCHS                           specify number of epochs

- -l LEARNING_RATE, --learning-rate LEARNING_RATE      specify learning rate
- -p PATCH_SIZE, --patch-size PATCH_SIZE             specify size of patch to section image, e.g: 8 = divide 128x128 image in 8x8 patches
- -d HIDDEN_DIM, --hidden-dim HIDDEN_DIM             specify number of layers in the hidden dimension of every transformer head
- -o {SGD,Adam}, --optimizer {SGD,Adam}              speecify optimizer to use with the model
- -w N_WORKERS, --n-workers N_WORKERS                specify number of CPU threads to load data
- -b N_BATCHES, --n-batches N_BATCHES                specify number of batches to split the dataset into
- -v, --model-summary                                if specified model summary will be displayed

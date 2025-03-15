## Overview

This repository contains a PyTorch implementation of a ResNet model for CIFAR-10 classification. The project is structured to facilitate training, evaluation, and inference on the CIFAR-10 dataset, with a focus on ease of use and customization.

"""
ResNet Implementation in PyTorch

This implementation is based on the paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

## Model Details

The following table describes the best performing model configuration and results:

| Category | Parameter | Value |
|----------|-----------|-------|
| **Model Architecture** | Blocks per layer | [3, 3, 4] |
|  | Channels per layer | [60, 122, 244] |
|  | Kernel size | 3 |
|  | Skip kernel size | 1 |
|  | Pool size | 8 |
|  | Number of parameters | 4,996,838 |
| **Training Setup** | Epochs | 150 |
|  | Training batch size | 128 |
|  | Testing batch size | 128 |
|  | Data augmentation | RandomCrop, RandomHorizontalFlip, RandomErasing |
|  | Number of workers | 4 |
|  | CutMix/MixUp | Disabled |
| **Optimization** | Loss function | CrossEntropyLoss with label smoothing (0.05) |
|  | Optimizer | SGD with momentum (0.9) and weight decay (5e-4) |
|  | Scheduler | CosineAnnealingLR |
| **Results** | Final train loss | 0.309 |
|  | Final train accuracy | 99.11% |
|  | Final test loss | 0.382 |
|  | Final test accuracy | 96.69% |

## Directory Structure

### `report`
This directory contains a detailed PDF report documenting the methodology used in this project, including model architecture decisions, hyperparameter selection process, training strategies, and comprehensive experimental results with performance analysis and comparisons.

### `logs`
This directory contains csv files, graphs and explainations of the comparisons of hyperparameter configurations that led to the paramters used for this model.

### `results`
This directory contains the model that achieved the best training accuracy, as well as the logs of it's training and it's performance on the hidden dataset for the kaggle competition.

### `src/model.py`
This file contains the implementation of the ResNet model and its building blocks. It defines the `BasicBlock` and `ResNet` classes, which are used to create the model architecture. The `create_model` function allows for easy instantiation of the model with customizable parameters:
    -Number of blocks per layer
    -Number of channels per layer
    -Kernel sizes
    -Pooling size

### `src/dataset.py`
This file contains functions for creating data loaders for the CIFAR-10 dataset. It supports customizable data augmentations and normalization, making it easy to experiment with different preprocessing techniques.

### `src/train.py`
This file handles the training process. It includes functions for training the model, testing its performance, saving checkpoints, and managing training metrics. The `main` function orchestrates the entire training process, including data loading, model initialization, and training loop execution.

### `src/utils.py`
This file provides utility functions used throughout the project. It includes functions for checking the environment (e.g., if running on Kaggle), managing file paths, and finding the best model checkpoints based on accuracy.

### `src/evaluate.py`
This file is responsible for evaluating the trained model on the hidden test dataset. It includes functions for loading the test data, loading the model from a checkpoint, running inference, and saving the predictions.


## Using the Kaggle Notebook

The `cifar-10-classifier-kaggle.ipynb` is the window to use the code in this repository to train and evaluate the ResNet model on the CIFAR-10 dataset on Kaggle. It is optimized to make selecting hyperparamters, saving data and submitting data as easy as possible.  

The notebook includes the following steps:

1. **Environment Setup**: Clone the repository and set up file paths.
    ```markdown
    !git clone https://github.com/jessenb16/CIFAR_10_Classification.git
    %cd CIFAR_10_Classification
    ```
2. **Model Creation**: Define the model architecture and create an instance of the model.

    ```markdown
    # Model Architecture
    BLOCKS_PER_LAYER = [3,3,4]           # Structure of the network
    CHANNELS_PER_LAYER = [60,120,240]    # Channels in each layer
    KERNEL_SIZE = 3                      # Size of convolutional kernels
    SKIP_KERNEL_SIZE = 1                 # Size of skip connection kernels
    POOL_SIZE = 8                        # Global pooling size

    # Training Parameters
    EPOCHS = 50
    TRAIN_BATCH_SIZE = 128
    TEST_BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    SMOOTHING = 0.05                     # Label smoothing factor
    RESUME = True                        # Resume from checkpoint

    AUGMENTATIONS = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5),    
    ]
    ```
3. **Training**: Train the model using the specified parameters, optimizer, and scheduler.
    ```markdown
    # Define optimizer
    OPTIMIZER = optim.SGD(model.parameters(), lr=LEARNING_RATE, 
                          momentum=0.9, weight_decay=5e-4, nesterov=True)

    # Define scheduler
    SCHEDULER = optim.lr_scheduler.CosineAnnealingLR(OPTIMIZER, T_max=EPOCHS)

    # Train the model
    train(model, EPOCHS, train_batch_size=TRAIN_BATCH_SIZE, 
          test_batch_size=TEST_BATCH_SIZE, augmentations=AUGMENTATIONS,
          optimizer=OPTIMIZER, scheduler=SCHEDULER, learning_rate=LEARNING_RATE, 
          smoothing=SMOOTHING, resume=RESUME, cutmix_mixup=True)
    ```
4. **Evaluation**: Evaluate the trained model on the hidden test dataset and save the predictions.

    ```markdown
    evaluate()  # Runs on hidden test dataset
    ```

    ## Advanced Features

    ### Resume Training
    Set `RESUME=True` to automatically find and load the best checkpoint:

    - Searches in `/kaggle/working/saved_models` and `/kaggle/input/cifar10-model-checkpoints`
    - Loads model weights, optimizer state, and scheduler state
    - Continues training from the last epoch

    ### CutMix and MixUp Augmentation
    Enable advanced augmentations with `cutmix_mixup=True`:

    - Randomly applies either CutMix or MixUp to training batches
    - Automatically handles one-hot encoded labels
    - Improves model generalization and accuracy

    ### Checkpoint Management
    The system automatically:

    - Keeps only the top 5 performing models
    - Deletes lower-performing checkpoints
    - Uses informative naming with accuracy in the filename

## Conclusion

This repository provides a comprehensive framework for training and evaluating a ResNet model on the CIFAR-10 dataset. The modular structure allows for easy customization and experimentation with different model architectures, training strategies, and data augmentations.


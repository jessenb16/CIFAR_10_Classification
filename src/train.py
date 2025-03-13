import torch
import torch.optim as optim
import torch.nn as nn
import os
import torch
import pandas as pd
import csv
from src.dataset import create_cifar10_dataloaders
from src.model import *
from src.utils import get_paths, find_best_checkpoint, is_kaggle
import glob
import heapq
import torchvision.transforms.v2 as v2

_, SAVED_MODELS_PATH, _, SAVED_DATA_PATH = get_paths()

# Global list to track the best checkpoints (min heap of tuples: (accuracy, filename))
best_checkpoints = []

def manage_checkpoints(new_checkpoint, max_to_keep=5):
    """
    Manages the best checkpoints, keeping only the top 'max_to_keep'.
    
    Args:
        new_checkpoint: Tuple of (accuracy, filename)
        max_to_keep: Maximum number of checkpoints to keep
    """
    global best_checkpoints
    
    # Add new checkpoint to our tracking list
    heapq.heappush(best_checkpoints, new_checkpoint)
    
    # If we have more than max_to_keep, remove the worst one(s)
    while len(best_checkpoints) > max_to_keep:
        accuracy, filename = heapq.heappop(best_checkpoints)  # Remove worst checkpoint
        filepath = os.path.join(SAVED_MODELS_PATH, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Removed checkpoint {filename} with accuracy {accuracy:.2f}%")

def save_performance(epoch, train_accuracy, test_accuracy, train_loss, test_loss, lr, model_name):
    """Save training performance metrics to CSV file"""
    os.makedirs(SAVED_DATA_PATH, exist_ok=True)
    csv_file = os.path.join(SAVED_DATA_PATH, f'{model_name}_training_history.csv')
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Write header only if file did not exist
        if not file_exists:
            writer.writerow(['epoch', 'train_accuracy', 'test_accuracy', 'train_loss', 'test_loss', 'learning_rate'])
        writer.writerow([epoch, train_accuracy, test_accuracy, train_loss, test_loss, lr])

def save_model(model, epoch, accuracy, optimizer=None, scheduler=None, max_to_keep=5):
    """Save model with informative filename and manage checkpoint retention"""
    os.makedirs(SAVED_MODELS_PATH, exist_ok=True)
    
    # Get model name or use default
    model_name = getattr(model, 'name', type(model).__name__)
    accuracy_str = f"{accuracy:.2f}".replace('.', '_')
    filename = f'{model_name}_epoch{epoch}_acc{accuracy_str}.pth'
    filepath = os.path.join(SAVED_MODELS_PATH, filename)
    
    # Save both complete model and state dict
    checkpoint = {
        'model': model,
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'accuracy': accuracy
    }
    
    # Add optimizer and scheduler if provided
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
    torch.save(checkpoint, filepath)
    print(f'Model checkpoint saved as {filename}')
    
    # Manage checkpoints - keep only the best max_to_keep
    manage_checkpoints((accuracy, filename), max_to_keep)
    
    return filename


def train(model, trainloader, loss_func, optimizer, device, cutmix_mixup=False):
    model.train()  # Set model to training mode
    train_loss, correct, total = 0, 0, 0

    # Setup CutMix/MixUp transforms if enabled
    cutmix_or_mixup = None
    if cutmix_mixup:
        cutmix = v2.CutMix(num_classes=10)
        mixup = v2.MixUp(num_classes=10)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)

        # Apply CutMix/MixUp if enabled
        if cutmix_mixup:
            images, labels = cutmix_or_mixup(images, labels)

        optimizer.zero_grad()   # Reset gradients

        outputs = model(images)             # Forward pass
        loss = loss_func(outputs, labels)   # Compute loss

        loss.backward()     # Backward pass
        optimizer.step()    # Update weights

        train_loss += loss.item() * images.size(0)  # Accumulate total loss (scaled by batch size)

        # Calculate accuracy - handle one-hot labels from CutMix/MixUp
        _, predicted = outputs.max(1)
        correct += (predicted == labels.argmax(dim=1)).sum().item() if cutmix_mixup else (predicted == labels).sum().item()
        total += labels.size(0)

    # Compute true average loss and accuracy for the epoch
    train_loss /= total  # Divide by total number of samples
    accuracy = 100 * correct / total

    # Print final loss and accuracy for the epoch
    print(f'Train Loss: {train_loss:.3f} | Train Acc: {accuracy:.2f}% ({correct}/{total})')

    return train_loss, accuracy

def test(model, testloader, loss_func, device):
    model.eval()  # Set model to evaluation mode
    test_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_func(outputs, labels)

            test_loss += loss.item() * images.size(0)  # Scale loss by batch size

            _, predicted = outputs.max(1)  # Get predicted class
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    # Compute average loss per sample
    test_loss /= total 
    accuracy = 100 * correct / total  # Compute accuracy

    print(f'Test Loss: {test_loss:.4f} | Accuracy: {accuracy:.2f}% ({correct}/{total})')

    return test_loss, accuracy  


def main(model, epochs, train_batch_size=128, test_batch_size=128, augmentations=None,
         optimizer=None, scheduler=None, smoothing=0.0, learning_rate=0.1, num_workers=2, 
         resume=False, cutmix_mixup=False):
    """
    Main function to train and test the model.
    Args:
        model: The model to train.
        epochs (int): Number of epochs to train.
        train_batch_size (int): Batch size for training data.
        test_batch_size (int): Batch size for testing data.
        augmentations (list): List of torchvision transforms for training.
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler.
        smoothing (float): Label smoothing factor.
        learning_rate (float): Learning rate for the optimizer.
        num_workers (int): Number of workers for data loading.
        resume (bool): Whether to resume from a checkpoint.
        cutmix_mixup (bool): Whether to use CutMix/MixUp augmentation.
    """
    
    # Count model parameters
    total_params = count_parameters(model)
    print(f'Total model parameters: {total_params}')
    if total_params > 5_000_000:
        raise ValueError('Model cannot have more than 5 million parameters')

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    trainloader, testloader = create_cifar10_dataloaders(
        train_batch_size=train_batch_size, 
        test_batch_size=test_batch_size,
        augmentations=augmentations,
        num_workers=num_workers)

    csv_file = os.path.join(SAVED_DATA_PATH, 'training_log.csv')
    checkpoint_file = os.path.join(SAVED_MODELS_PATH, 'checkpoint.pth')

    start_epoch = 0
    best_accuracy = 0.0

    # Resume training if specified
    if resume:
        # For Kaggle, also check the dataset inputs
        kaggle_input_path = '/kaggle/input/cifar10-model-checkpoints' if is_kaggle() else None
        best_checkpoint_path, is_kaggle_checkpoint = find_best_checkpoint(SAVED_MODELS_PATH, kaggle_input_path)
        
        if best_checkpoint_path:
            print(f"Loading checkpoint: {os.path.basename(best_checkpoint_path)}")
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            
            # Load model state
            if 'model' in checkpoint:
                # Full model saved
                model = checkpoint['model'].to(device)
            elif 'model_state_dict' in checkpoint:
                # Only state dict saved
                model.load_state_dict(checkpoint['model_state_dict'])
            
            # Get training info
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_accuracy = checkpoint.get('accuracy', 0.0)
            
            # Check if optimizer needs to be loaded
            if optimizer is None:
                # Create default optimizer if None provided
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
                # Try to load optimizer state if available
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        print("Loaded optimizer state from checkpoint")
                    except:
                        print("Warning: Could not load optimizer state - using fresh optimizer")
            else:
                # Custom optimizer provided - don't load state
                print("Using new optimizer - ignoring saved optimizer state")
                
            # Check if scheduler needs to be loaded
            if scheduler is None:
                # Create default scheduler if None provided
                if 'scheduler_state_dict' in checkpoint:
                    # Try to recreate scheduler type from checkpoint
                    try:
                        scheduler_type = checkpoint.get('scheduler_type', 'StepLR')
                        if scheduler_type == 'StepLR':
                            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
                        # Add other scheduler types as needed
                        
                        # Load scheduler state
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        print("Loaded scheduler state from checkpoint")
                    except:
                        print("Warning: Could not recreate scheduler - using default")
            else:
                # Custom scheduler provided - don't load state
                print("Using new scheduler - ignoring saved scheduler state")
                
            # Initialize best_checkpoints tracking list with this checkpoint
            if not is_kaggle_checkpoint:  # Only track local checkpoints
                global best_checkpoints
                filename = os.path.basename(best_checkpoint_path)
                best_checkpoints = [(best_accuracy, filename)]
                
            print(f"Resumed from epoch {start_epoch-1} with accuracy {best_accuracy:.2f}% with learning rate {optimizer.param_groups[0]['lr']:.6f}")
        else:
            print("No checkpoint found - starting from scratch")
    
    # Initialize model
    print('Initializing model...')
    model = model.to(device)

    # Define loss function
    if smoothing > 0:
        loss_func = nn.CrossEntropyLoss(label_smoothing=smoothing)
    else:
        loss_func = nn.CrossEntropyLoss()
    
    # Ensure optimizer is valid
    if optimizer and not isinstance(optimizer, optim.Optimizer):
        raise TypeError('Optimizer must be an instance of torch.optim.Optimizer')

    # Define optimizer
    optimizer = optimizer if optimizer else optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    print(optimizer)

    # Ensure scheduler is valid
    if scheduler and not isinstance(scheduler, optim.lr_scheduler.LRScheduler):
        raise TypeError('Scheduler must be an instance of torch.optim.lr_scheduler.LRScheduler')
    print(scheduler)
    

    # Train model for multiple epochs
    print('Training model...')
    if not resume:
        best_accuracy, best_epoch = 0.0, 0
    else:
        best_epoch = start_epoch - 1  # Use the epoch from loaded checkpoint

    for epoch in range(start_epoch, start_epoch + epochs + 1):
        print(f'Epoch: {epoch}')
        train_loss, train_accuracy = train(model, trainloader, loss_func, optimizer, device, cutmix_mixup)
        test_loss, test_accuracy = test(model, testloader, loss_func, device)

        # Save the model if test accuracy improves
        if test_accuracy > best_accuracy:
            print("Saving checkpoint...")
            # Use the new save_model function
            best_model_file = save_model(model, epoch, test_accuracy, optimizer, scheduler)
            best_accuracy = test_accuracy
            best_epoch = epoch

        # Save performance metrics
        model_name = getattr(model, 'name', type(model).__name__)
        save_performance(
            epoch, 
            train_accuracy, 
            test_accuracy, 
            train_loss, 
            test_loss,
            optimizer.param_groups[0]['lr'],
            model_name
        )

        # Apply scheduler if provided
        scheduler.step() if scheduler else None
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print()

    print('Training complete')
    print(f'Best test accuracy: {best_accuracy:.2f}% at epoch {best_epoch}')
    if best_model_file:
        print(f'Best model saved as: {best_model_file}')

if __name__ == "__main__":
    model = create_model(
        blocks_per_layer=[2, 2, 2, 2],       # ResNet-18 style
        channels_per_layer=[64, 128, 256, 512],  # Standard ResNet filters
        kernel_size=3,                       # Standard 3x3 convs
        skip_kernel_size=1,                  # 1x1 convs for skip connections
        pool_size=1,                         # Global average pooling
    )

    main(model, epochs=5, train_batch_size=128, test_batch_size=128, augmentations=None, optimizer=None, scheduler=None, smoothing=0.0, learning_rate=0.01, num_workers=2, resume=False)



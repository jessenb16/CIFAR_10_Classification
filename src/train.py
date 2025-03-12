import torch
import torch.optim as optim
import torch.nn as nn
import os
import torch
import pandas as pd
from src.dataset import create_cifar10_dataloaders
from src.model import *
from src.utils import get_paths

_, SAVED_MODELS_PATH, _, SAVED_DATA_PATH = get_paths()


def train(model, trainloader, loss_func, optimizer, device):
    model.train()  # Set model to training mode
    train_loss, correct, total = 0, 0, 0

    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()   # Reset gradients

        outputs = model(images)             # Forward pass
        loss = loss_func(outputs, labels)   # Compute loss

        loss.backward()     # Backward pass
        optimizer.step()    # Update weights

        train_loss += loss.item() * images.size(0)  # Accumulate total loss (scaled by batch size)

        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
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

    print(f'TEST: Loss: {test_loss:.4f} | Accuracy: {accuracy:.2f}% ({correct}/{total})')

    return test_loss, accuracy  


def main(model, epochs, train_batch_size=128, test_batch_size=128, augmentations=None,
         optimizer=None, scheduler=None, smoothing=0.0, learning_rate=0.01, num_workers=2, resume=False):
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
    """
    
    # Count model parameters
    total_params = count_parameters(model)
    print(f'Total model parameters: {total_params}')
    if total_params > 5_000_000:
        raise ValueError('Model cannot have more than 5 million parameters')


    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    trainloader, testloader = create_cifar10_dataloaders(train_batch_size, test_batch_size,
                                                          augmentations=augmentations, num_workers=num_workers)

    csv_file = os.path.join(SAVED_DATA_PATH, 'training_log.csv')
    checkpoint_file = os.path.join(SAVED_MODELS_PATH, 'checkpoint.pth')

    start_epoch = 1
    # Resume training if specified
    if resume:
        if os.path.isfile(checkpoint_file):
            print(f"Loading checkpoint '{checkpoint_file}'")
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_accuracy = checkpoint['test_accuracy']
        else:
            print(f"No checkpoint found at '{checkpoint_file}'")
            start_epoch = 1
    
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

    # Ensure scheduler is valid
    if scheduler and not isinstance(scheduler, optim.lr_scheduler.LRScheduler):
        raise TypeError('Scheduler must be an instance of torch.optim.lr_scheduler.LRScheduler')
    

    # Train model for multiple epochs
    print('Training model...')
    best_accuracy, best_epoch = 0.0, 0
    for epoch in range(start_epoch, epochs + 1):
        print(f'Epoch: {epoch}')
        train_loss, train_accuracy = train(model, trainloader, loss_func, optimizer, device)
        test_loss, test_accuracy = test(model, testloader, loss_func, device)

        # Save the model if test accuracy improves
        if test_accuracy > best_accuracy:
            print("Saving checkpoint...")
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_accuracy': test_accuracy,
            }
            torch.save(state, checkpoint_file)
            best_accuracy = test_accuracy
            best_epoch = epoch

        # Save training log
        df_data = pd.DataFrame([{
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'learning_rate': optimizer.param_groups[0]['lr'],
        }])

        if os.path.isfile(csv_file):
            df_data.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df_data.to_csv(csv_file, mode='w', header=True, index=False)

        scheduler.step() if scheduler else None
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')

    print('Training complete')
    print(f'Best test accuracy: {best_accuracy:.2f}% at epoch {best_epoch}')

if __name__ == "__main__":
    model = create_model(
        blocks_per_layer=[2, 2, 2, 2],       # ResNet-18 style
        channels_per_layer=[64, 128, 256, 512],  # Standard ResNet filters
        kernel_size=3,                       # Standard 3x3 convs
        skip_kernel_size=1,                  # 1x1 convs for skip connections
        pool_size=1,                         # Global average pooling
    )

    main(model, epochs=5, train_batch_size=128, test_batch_size=128, augmentations=None, optimizer=None, scheduler=None, smoothing=0.0, learning_rate=0.01, num_workers=2, resume=False)




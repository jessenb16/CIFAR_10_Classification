import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader




# Function to create CIFAR-10 DataLoader with Custom Augmentations
def create_cifar10_dataloaders(
    data_path="./data",
    train_batch_size=128,
    test_batch_size=256,
    num_workers=2,
    augmentations=None,
    normalize=True
):
    """
    Creates DataLoaders for CIFAR-10 with customizable augmentations.

    Args:
        data_path (str): Path to store the dataset.
        train_batch_size (int): Batch size for training data.
        test_batch_size (int): Batch size for testing data.
        num_workers (int): Number of workers for data loading.
        augmentations (list): List of torchvision transforms for training.
        normalize (bool): Whether to normalize data.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and test DataLoaders.
    """
    
    # CIFAR-10 normalization statistics
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    if not normalize:
        mean, std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)  # No normalization
    
    # Default augmentations if none are provided
    if augmentations is None:
        augmentations = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor()
        ]
    
    # Define the transformations
    train_transform = transforms.Compose(augmentations + [transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    # Load CIFAR-10 Dataset
    train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, transform=train_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, transform=test_transform, download=True)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

if __name__ == "__main__":
    # Example Usage
    train_loader, test_loader = create_cifar10_dataloaders(
        train_batch_size=128,
        test_batch_size=256,
        num_workers=4,
        augmentations=[
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor()
        ]
    )
    # Check DataLoader output
    print("Train set shape", train_loader.dataset.data.shape)
    print("Test set shape", test_loader.dataset.data.shape)
    print("Augmentations applied to training set:", train_loader.dataset.transform)


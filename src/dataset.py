import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def get_cifar10_loaders(batch_size=128, data_path="/content/data/cifar-10-python"):
    """
    Loads the CIFAR-10 dataset from the Kaggle competition dataset in Colab.

    Args:
        batch_size (int): Batch size for training/testing DataLoaders.
        data_path (str): Path to the dataset.

    Returns:
        trainloader (DataLoader): PyTorch DataLoader for training.
        testloader (DataLoader): PyTorch DataLoader for testing.
    """

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load dataset from the correct directory
    trainset = datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root=data_path, train=False, download=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

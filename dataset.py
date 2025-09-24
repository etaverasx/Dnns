import torch
import torchvision
import torchvision.transforms as transforms

def get_dataloader(dataset_name="MNIST", batch_size=64, train=True):
    """
    Returns a DataLoader for MNIST or CIFAR-10 datasets.
    """
    if dataset_name.upper() == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = torchvision.datasets.MNIST(
            root="./data", train=train, download=True, transform=transform
        )

    elif dataset_name.upper() == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = torchvision.datasets.CIFAR10(
            root="./data", train=train, download=True, transform=transform
        )

    else:
        raise ValueError("Dataset not supported. Choose MNIST or CIFAR10.")

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)

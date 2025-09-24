import torch
import torchvision
import torchvision.transforms as transforms

def get_dataloader(dataset_name="MNIST", batch_size=64, train=True, img_size=32):
    """
    Returns a DataLoader for MNIST or CIFAR-10 datasets,
    resizing inputs to the requested img_size.
    - Default: 32x32 (for LeNet, ResNet18)
    - Use 224x224 when training ViT
    """
    if dataset_name.upper() == "MNIST":
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = torchvision.datasets.MNIST(
            root="./data", train=train, download=True, transform=transform
        )

    elif dataset_name.upper() == "CIFAR10":
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = torchvision.datasets.CIFAR10(
            root="./data", train=train, download=True, transform=transform
        )

    else:
        raise ValueError("Dataset not supported. Choose MNIST or CIFAR10.")

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)

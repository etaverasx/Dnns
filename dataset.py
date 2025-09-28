# dataset.py
# This file provides separate DataLoader functions for Task 1 and Task 2.
# Task 1 uses a simple pipeline (resize + normalize).
# Task 2 adds optional augmentations (rotation, horizontal flip).

import torch
import torchvision
import torchvision.transforms as transforms

# ======================
# Task 1 DataLoader
# ======================
def get_dataloader_task1(dataset_name="MNIST", batch_size=64, train=True, img_size=32):
    """
    Task 1:
    Returns a DataLoader for MNIST or CIFAR-10 datasets.
    Images are resized and normalized, no augmentations are applied.
    - Default image size: 32x32 (works with LeNet and ResNet18).
    - Use 224x224 for Vision Transformer.
    """

    # If dataset is MNIST (grayscale, 1 channel)
    if dataset_name.upper() == "MNIST":
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),    # resize to match model input size
            transforms.ToTensor(),                     # convert image to tensor
            transforms.Normalize((0.5,), (0.5,))       # normalize to [-1, 1]
        ])
        dataset = torchvision.datasets.MNIST(
            root="./data", train=train, download=True, transform=transform
        )

    # If dataset is CIFAR10 (color, 3 channels)
    elif dataset_name.upper() == "CIFAR10":
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),               # resize image
            transforms.ToTensor(),                                # convert to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize RGB channels
        ])
        dataset = torchvision.datasets.CIFAR10(
            root="./data", train=train, download=True, transform=transform
        )

    # Unsupported dataset
    else:
        raise ValueError("Dataset not supported. Choose MNIST or CIFAR10.")

    # Return DataLoader object (batches data for training/testing)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)


# ======================
# Task 2 DataLoader
# ======================
def get_dataloader_task2(dataset_name="MNIST", batch_size=64, train=True, img_size=32, augment="none"):
    """
    Task 2:
    Returns a DataLoader for MNIST or CIFAR-10 datasets.
    Adds augmentation options when training:
      - "rotation": applies random rotation
      - "flip": applies random horizontal flip
      - "none": no augmentation
    """

    # Decide normalization values depending on dataset
    if dataset_name.upper() == "MNIST":
        mean, std = (0.5,), (0.5,)  # grayscale normalization
    elif dataset_name.upper() == "CIFAR10":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)  # RGB normalization
    else:
        raise ValueError("Dataset not supported. Choose MNIST or CIFAR10.")

    # Start building transformation list
    transform_list = [transforms.Resize((img_size, img_size))]

    # Apply augmentations only when training
    if train:
        if augment == "rotation":
            transform_list.append(transforms.RandomRotation(15))  # rotate up to 15 degrees
        elif augment == "flip":
            transform_list.append(transforms.RandomHorizontalFlip())  # flip horizontally

    # Add tensor conversion and normalization
    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    # Combine into a single transform pipeline
    transform = transforms.Compose(transform_list)

    # Load dataset with transforms
    if dataset_name.upper() == "MNIST":
        dataset = torchvision.datasets.MNIST(
            root="./data", train=train, download=True, transform=transform
        )
    elif dataset_name.upper() == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10(
            root="./data", train=train, download=True, transform=transform
        )

    # Return DataLoader object
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)

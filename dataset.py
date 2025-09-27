import torch
import torchvision
import torchvision.transforms as transforms

def get_dataloader(dataset_name="MNIST", batch_size=64, train=True, img_size=32):
    """
    Returns a DataLoader for MNIST or CIFAR-10 datasets.
    Images are resized to the requested img_size and normalized.
    Default size: 32x32 (for LeNet and ResNet18)
    Use 224x224 when training ViT.
    """
    
    # MNIST: grayscale images (1 channel)
    if dataset_name.upper() == "MNIST":
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),   # resize to chosen input size
            transforms.ToTensor(),                    # convert to tensor [0,1]
            transforms.Normalize((0.5,), (0.5,))      # normalize to [-1,1]
        ])
        dataset = torchvision.datasets.MNIST(
            root="./data", train=train, download=True, transform=transform
        )

    # CIFAR-10: RGB images (3 channels)
    elif dataset_name.upper() == "CIFAR10":
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),             # resize to chosen input size
            transforms.ToTensor(),                              # convert to tensor [0,1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize each channel
        ])
        dataset = torchvision.datasets.CIFAR10(
            root="./data", train=train, download=True, transform=transform
        )

    # Invalid dataset case
    else:
        raise ValueError("Dataset not supported. Choose MNIST or CIFAR10.")

    # Return DataLoader (shuffles only for training)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)





"""
TASK 2 - Dataset Loader with Augmentation
extend the dataset loader 
Rotation augmentation (train with vs. without rotation)
Horizontal flip augmentation (train with vs. without flip)
Optimizer choice (SGD vs. Adam)
Batch size variations (small vs. large)
Learning rate variations (0.001 vs. 0.01)
"""

# import torch
# import torchvision
# import torchvision.transforms as transforms
#
# def get_dataloader(dataset_name="MNIST", batch_size=64, train=True, img_size=32, augment="none"):
#     """
#     Returns a DataLoader for MNIST or CIFAR-10 datasets.
#     Images are resized to the requested img_size and normalized.
#
#     Augmentation options (applied only during training):
#       - "rotation": random rotation up to ±15 degrees
#       - "flip": random horizontal flip
#       - "none": no augmentation
#     """
#     
#     # Mean and std values depend on dataset
#     if dataset_name.upper() == "MNIST":
#         mean, std = (0.5,), (0.5,)  # grayscale normalization
#     elif dataset_name.upper() == "CIFAR10":
#         mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)  # RGB normalization
#     else:
#         raise ValueError("Dataset not supported. Choose MNIST or CIFAR10.")
#
#     # Start with resize transform
#     transform_list = [transforms.Resize((img_size, img_size))]
#
#     # Add augmentations only for training
#     if train:
#         if augment == "rotation":
#             transform_list.append(transforms.RandomRotation(15))   # rotate up to ±15 degrees
#         elif augment == "flip":
#             transform_list.append(transforms.RandomHorizontalFlip())  # flip with 50% chance
#
#     # Add conversion and normalization
#     transform_list += [
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)
#     ]
#     transform = transforms.Compose(transform_list)
#
#     # Load dataset with transforms
#     if dataset_name.upper() == "MNIST":
#         dataset = torchvision.datasets.MNIST(
#             root="./data", train=train, download=True, transform=transform
#         )
#     elif dataset_name.upper() == "CIFAR10":
#         dataset = torchvision.datasets.CIFAR10(
#             root="./data", train=train, download=True, transform=transform
#         )
#
#     # Return DataLoader (shuffle only when training)
#     return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)

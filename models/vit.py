import torch
import torch.nn as nn
import torchvision.models as models


def get_vit(num_classes=10, in_channels=1):
    """
    Returns a Vision Transformer (ViT-B/16) from torchvision,
    modified for MNIST/CIFAR10 classification.
    """
    # Load a ViT without pretrained weights (fair comparison)
    model = models.vit_b_16(weights=None)

    # Adjust input channels (ViT expects 3-channel images)
    if in_channels == 1:
        # Expand grayscale MNIST images to 3 channels
        model.conv_proj = nn.Conv2d(
            1,
            model.conv_proj.out_channels,
            kernel_size=model.conv_proj.kernel_size,
            stride=model.conv_proj.stride,
            padding=model.conv_proj.padding,
        )

    # Replace classification head with Sequential (so keys match checkpoints)
    model.heads = nn.Sequential(
        nn.Linear(model.heads.head.in_features, num_classes)
    )

    return model

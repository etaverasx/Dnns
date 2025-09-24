import torch.nn as nn
import torchvision.models as models

def get_vit(num_classes=10, in_channels=1, img_size=224):
    """
    Returns a Vision Transformer (ViT) model for MNIST or CIFAR10.
    - Default img_size = 224 to match ViT training setup.
    - If you want to use 32x32, you can override with img_size=32 (not recommended).
    """
    # Base ViT (patch size 16, expects 224x224 inputs by default)
    model = models.vit_b_16(weights=None)

    # Adjust patch embedding for grayscale (MNIST)
    if in_channels == 1:
        model.conv_proj = nn.Conv2d(
            1, model.conv_proj.out_channels,
            kernel_size=model.conv_proj.kernel_size,
            stride=model.conv_proj.stride,
            padding=model.conv_proj.padding,
            bias=False
        )

    # Replace classifier head safely
    in_features = model.heads.head.in_features
    model.heads = nn.Sequential(nn.Linear(in_features, num_classes))

    return model

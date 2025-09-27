import torch
import torch.nn as nn
import torchvision.models as models

def get_vit(num_classes=10, in_channels=1):
    # load a Vision Transformer (ViT-B/16) without pretrained weights
    model = models.vit_b_16(weights=None)

    # if dataset is grayscale (MNIST), change first conv to accept 1 channel
    if in_channels == 1:
        model.conv_proj = nn.Conv2d(
            1,  # input channels
            model.conv_proj.out_channels,  # keep original output channels
            kernel_size=model.conv_proj.kernel_size,  # same kernel size
            stride=model.conv_proj.stride,  # same stride
            padding=model.conv_proj.padding  # same padding
        )

    # replace the classification head with a new linear layer for num_classes
    model.heads = nn.Sequential(
        nn.Linear(model.heads.head.in_features, num_classes)
    )

    # return the modified model
    return model

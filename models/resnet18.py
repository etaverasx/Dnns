import torch.nn as nn
import torchvision.models as models

def get_resnet18(num_classes=10, in_channels=1):
    """
    Returns a ResNet18 model adapted for MNIST (1 channel) or CIFAR10 (3 channels).
    """
    model = models.resnet18(weights=None)  # no pretrained weights

    # Adapt first conv layer if grayscale (MNIST)
    if in_channels == 1:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Adapt final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

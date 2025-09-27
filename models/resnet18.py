import torch.nn as nn
import torchvision.models as models

def get_resnet18(num_classes=10, in_channels=1):
    # load a ResNet18 model without pretrained weights
    model = models.resnet18(weights=None)

    # if input is grayscale (MNIST), replace first conv to accept 1 channel instead of 3
    if in_channels == 1:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # replace the final fully connected layer to match the number of classes (10 for MNIST/CIFAR10)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # return the modified model
    return model

import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super(LeNet, self).__init__()
        # first conv layer: adjust channels for MNIST (1) or CIFAR10 (3), output 6 feature maps
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        # average pooling layer to reduce spatial size
        self.pool = nn.AvgPool2d(2, stride=2)
        # second conv layer: take 6 channels from conv1, output 16 feature maps
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # first fully connected layer: 16*5*5 = 400 inputs → 120 outputs
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # second fully connected layer: 120 → 84
        self.fc2 = nn.Linear(120, 84)
        # output layer: 84 → number of classes (10 for MNIST/CIFAR10)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # conv1 + relu + pooling
        x = self.pool(F.relu(self.conv1(x)))
        # conv2 + relu + pooling
        x = self.pool(F.relu(self.conv2(x)))
        # flatten to (batch_size, features)
        x = x.view(x.size(0), -1)
        # fc1 + relu
        x = F.relu(self.fc1(x))
        # fc2 + relu
        x = F.relu(self.fc2(x))
        # final output (logits)
        x = self.fc3(x)
        return x

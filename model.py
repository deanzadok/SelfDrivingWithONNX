from __future__ import print_function
import torch
from torch.nn import Module, Conv2d, Linear
from torch.nn.functional import relu

class ILModel(Module):
    def __init__(self, num_actions=5):
        super(ILModel, self).__init__()
        self.conv1 = Conv2d(1, 16, 8, stride=4)
        self.conv2 = Conv2d(16, 32, 4, stride=2)
        self.conv3 = Conv2d(32, 32, 3, stride=1)
        self.fc1 = Linear(7*7*32, 256)
        self.fc2 = Linear(256, num_actions)

    def forward(self, x):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = relu(self.conv3(x))
        x = x.view(-1, 7*7*32)
        x = relu(self.fc1(x))
        x = self.fc2(x)
        return x
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_size, 128, 1, 2)
        self.pool1 = nn.MaxPool2d(1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 64, 1, 2)
        self.pool2 = nn.MaxPool2d(1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 32, 1, 2)
        self.pool3 = nn.MaxPool2d(1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 8, 1, 4)
        self.pool4 = nn.MaxPool2d(1)
        self.relu4 = nn.ReLU()
        self.flat = nn.Flatten()
        self.linear = nn.Linear(512, output_size)
        self.softmax = nn.Softmax(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.relu4(x)
        x = self.flat(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x

import torch
import torch.nn as nn
from timm.models.resnet import resnet18

class ResNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        self.model = resnet18(pretrained=True, num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        return self.model(x)

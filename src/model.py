import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, input_size: int, output_size: int, activation, subsample: bool = False):
        super().__init__()
        if not subsample:
            output_size = input_size

        self.net = nn.Sequential(
            nn.Conv2d(
                input_size, 
                output_size, 
                kernel_size=3, 
                padding=1, 
                stride=1 if not subsample else 2,
                bias=False,
            ),
            nn.BatchNorm2d(output_size),
            activation(),
            nn.Conv2d(output_size, output_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_size),
        )

        self.downsample = nn.Conv2d(input_size, output_size, kernel_size=1, stride=2) if subsample else None
        self.act = activation()

    def forward(self, x: torch.Tensor):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        return self.act(out)

class ResNet(nn.Module):
    def __init__(
        self, 
        n_classes: int, 
        n_blocks: list[int]=[3,3,3], 
        hidden_size: list[int]=[16,32,64],
        activation=nn.ReLU,
        
    ):
        super().__init__()
        assert len(n_blocks) == len(hidden_size)
        self.act_name = activation().__str__()[:-2].lower()
        self.input_net = nn.Sequential(
            nn.Conv2d(3, hidden_size[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size[0]),
            activation(),
        )

        blocks = []
        for idx, block_count in enumerate(n_blocks):
            for bc in range(block_count):
                subsample = (bc == 0 and idx > 0)
                blocks.append(
                    ResNetBlock(
                        input_size=hidden_size[idx if not subsample else (idx - 1)],
                        activation=activation,
                        subsample=subsample,
                        output_size=hidden_size[idx]
                    )
                )

        self.blocks = nn.Sequential(*blocks)

        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(hidden_size[-1], n_classes)
        )

        self._init_params()

    def forward(self, x: torch.Tensor):
        x = self.input_net(x)
        x = self.blocks(x)
        return self.output_net(x)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=self.act_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

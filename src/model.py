import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, input_size: int, bottleneck_size: int, growth_rate, activation):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(input_size),
            activation(),
            nn.Conv2d(input_size, bottleneck_size * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_size * growth_rate),
            activation(),
            nn.Conv2d(bottleneck_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        out = self.net(x)
        return torch.cat([out, x], dim=1)

class DenseBlock(nn.Module):
    def __init__(self, input_size, num_layers, bottleneck_size, growth_rate, activation):
        super().__init__()
        layers = []
        for idx in range(num_layers):
            layers.append(
                DenseLayer(
                    input_size=input_size + idx * growth_rate,
                    bottleneck_size=bottleneck_size,
                    growth_rate=growth_rate,
                    activation=activation
                )
            )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class TransitionLayer(nn.Module):
    def __init__(self, input_size, output_size, activation):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(input_size),
            activation(),
            nn.Conv2d(input_size, output_size, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)

class DenseNet(nn.Module):
    def __init__(
            self, 
            n_classes: int, 
            input_size: int, 
            n_layers: list[int] = [6, 6, 6, 6], 
            bottleneck_size: int = 2, 
            growth_rate: int = 16, 
            activation = nn.ReLU,
    ):
        super().__init__()

        hidden_size = growth_rate * bottleneck_size
        self.input_net = nn.Conv2d(input_size, hidden_size, kernel_size=3, padding=1)

        blocks = []
        for block_idx, layers in enumerate(n_layers):
            blocks.append(
                DenseBlock(
                    hidden_size,
                    layers,
                    bottleneck_size,
                    growth_rate,
                    activation,
                )
            )
            hidden_size += layers + growth_rate
            if block_idx < len(n_layers) - 1:
                blocks.append(
                    TransitionLayer(
                        hidden_size,
                        hidden_size // 2,
                        activation,
                    )
                )

                hidden_size //= 2
        self.blocks = nn.Sequential(*blocks)

        self.output = nn.Sequential(
            nn.BatchNorm2d(hidden_size),
            activation(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_size, n_classes),
        )

        self.act_name = activation().__str__()[:-2].lower()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity=self.act_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        return self.output(x)

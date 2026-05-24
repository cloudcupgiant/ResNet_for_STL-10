"""Small ResNet implementation used as the project baseline model."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def make_activation(name: str) -> nn.Module:
    """Return an activation module used by SmallResNet blocks."""
    key = name.lower()
    if key == "relu":
        return nn.ReLU(inplace=True)
    if key == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    if key == "tanh":
        return nn.Tanh()
    if key == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation: {name}")


class BasicBlock(nn.Module):
    """Two 3x3 convolutions plus a residual shortcut."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = make_activation(activation)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = make_activation(activation)

        # Match shape changes caused by downsampling or channel expansion before addition.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        # Residual branch: conv-bn-act followed by conv-bn, then add the shortcut.
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        return self.act2(out)


class SmallResNet(nn.Module):
    """A ResNet-18 style network adapted for 96x96 STL-10 images."""

    def __init__(
        self,
        block: type[BasicBlock],
        layers: Iterable[int],
        num_classes: int = 10,
        dropout: float = 0.2,
        activation: str = "relu",
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.in_channels = base_channels
        self.activation_name = activation

        # STL-10 images are already small enough that the stem avoids a 7x7 stride-2 conv.
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            make_activation(activation),
        )
        self.layer1 = self._make_layer(block, base_channels, layers[0], stride=1, activation=activation)
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2, activation=activation)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2, activation=activation)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=2, activation=activation)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(base_channels * 8 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(
        self,
        block: type[BasicBlock],
        out_channels: int,
        num_blocks: int,
        stride: int,
        activation: str,
    ) -> nn.Sequential:
        """Create one ResNet stage; only the first block may change resolution."""
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for block_stride in strides:
            blocks.append(block(self.in_channels, out_channels, block_stride, activation=activation))
            # Track output channels so the next block receives the correct input shape.
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        """Initialize layers with common ResNet defaults."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape flow: NCHW image -> convolutional stages -> pooled vector -> logits.
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


def small_resnet18(
    num_classes: int = 10,
    dropout: float = 0.2,
    activation: str = "relu",
    base_channels: int = 64,
) -> SmallResNet:
    """Build a Small ResNet-18 from scratch for STL-10."""
    return SmallResNet(
        BasicBlock,
        layers=(2, 2, 2, 2),
        num_classes=num_classes,
        dropout=dropout,
        activation=activation,
        base_channels=base_channels,
    )

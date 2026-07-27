from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y or diff_x:
            x = nn.functional.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class EfficientUNet(nn.Module):
    """EfficientNet encoder + U-Net decoder for canopy height regression."""

    def __init__(self, in_channels: int = 4, out_channels: int = 1, pretrained: bool = False):
        super().__init__()
        backbone = efficientnet_b0(weights=None if not pretrained else "DEFAULT")

        if in_channels != 3:
            first = backbone.features[0][0]
            backbone.features[0][0] = nn.Conv2d(
                in_channels,
                first.out_channels,
                kernel_size=first.kernel_size,
                stride=first.stride,
                padding=first.padding,
                bias=False,
            )

        self.encoder = backbone.features

        self.dec4 = UpBlock(1280, 112, 256)
        self.dec3 = UpBlock(256, 40, 128)
        self.dec2 = UpBlock(128, 24, 64)
        self.dec1 = UpBlock(64, 16, 32)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            ConvBlock(16, 16),
            nn.Conv2d(16, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.encoder[0](x)
        x1 = self.encoder[1](x0)
        x2 = self.encoder[2](x1)
        x3 = self.encoder[3](x2)
        x4 = self.encoder[4](x3)
        x5 = self.encoder[5](x4)
        x6 = self.encoder[6](x5)
        x7 = self.encoder[7](x6)
        x8 = self.encoder[8](x7)

        x = self.dec4(x8, x5)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)
        x = self.final_up(x)
        return self.head(x)


if __name__ == "__main__":
    model = EfficientUNet(in_channels=4, out_channels=1)
    x = torch.randn(1, 4, 128, 128)
    y = model(x)
    print(y.shape)

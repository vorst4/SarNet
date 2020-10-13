import torch.nn as nn
from abc import ABC
from .blocks import Reshape


class ConvAE2(nn.Module, ABC):
    """
    Convolutional auto-encoder 2
        bottleneck: 1 dense connected layer, batch norm and relu
        decoder: 5 times (upsample, conv2d, batch norm, relu)
    """

    lr_ideal = 1e-7

    def __init__(self):
        super().__init__()

        channels = 512

        # layers
        self.encoder = SimplifiedEncoder()
        self.bottleneck = nn.Sequential(
            nn.Linear(2, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU())
        self.decoder = nn.Sequential(
            Reshape((-1, channels, 1, 1)),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(channels, channels, 3, padding=1),  # 512x2x2 -> 512x2x2
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(channels, channels//2, 3, padding=1),  # -> 256x4x4
            nn.BatchNorm2d(channels//2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(channels//2, channels//4, 3, padding=1),  # -> 128x8x8
            nn.BatchNorm2d(channels//4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(channels//4, channels//8, 3, padding=1),  # -> 64x16x16
            nn.BatchNorm2d(channels//8),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(channels//8, 1, 3, padding=1)  # -> 1x32x32
        )

    def forward(self, input_img, input_meta):
        x = self.encoder(input_img, input_meta)
        x = self.bottleneck(x)
        return self.decoder(x)

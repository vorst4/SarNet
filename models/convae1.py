from abc import ABC
import torch.nn as nn
from .blocks import Reshape
from .blocks import SimplifiedEncoder


class ConvAE1(nn.Module, ABC):
    """
    Convolutional Auto-encoder 1
        bottleneck: 1 dense connected layer, batch norm and relu
        decoder: 5 times (transpose conv2d, batch norm, relu)
    """

    lr_ideal = 1e-5

    def __init__(self):
        super().__init__()

        channels = 1024

        # layers
        self.encoder = nn.Sequential(
            # 3 x 64 x 64 ->
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(2, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU())
        self.decoder = nn.Sequential(

            # -> 1024x1x1
            Reshape((-1, channels, 1, 1)),

            # 1024x1x1 -> 512x2x2
            nn.ConvTranspose2d(channels, channels // 2, 2, stride=2),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),

            # -> 256x4x4
            nn.ConvTranspose2d(channels // 2, channels // 4, 2, stride=2),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(),

            # -> 128x8x8
            nn.ConvTranspose2d(channels // 4, channels // 8, 2, stride=2),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU(),

            # -> 64x16x16
            nn.ConvTranspose2d(channels // 8, channels // 16, 2, stride=2),
            nn.BatchNorm2d(channels // 16),
            nn.ReLU(),

            # -> 1x32x32
            nn.ConvTranspose2d(channels // 16, 1, 2, stride=2)
        )

    def forward(self, input_img, input_meta):
        x = self.encoder(input_img, input_meta)
        x = self.bottleneck(x)
        return self.decoder(x)

from abc import ABC
import torch.nn as nn
from .blocks import Reshape
from .blocks import ConvBnHs, Combine, LinBnHs, IResBlock


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
            # Use just a convolutional layer for the first layer
            IResBlock(ci=3, co=32, ri=64, k=3, expand=False, squeeze=False),
            IResBlock(ci=32, co=64, ri=64, k=3, downsample=True),
            IResBlock(ci=64, co=96, ri=32, k=5, downsample=True),
            IResBlock(ci=96, co=128, ri=16, k=3, downsample=True),
            IResBlock(ci=128, co=256, ri=8, k=5, downsample=True),
            IResBlock(ci=256, co=512, ri=4, k=3, downsample=True),
            IResBlock(ci=512, co=1024, ri=2, k=3, downsample=True),
        )

        # combination of encoder and meta-data input
        self.combine = Combine()

        # bottleneck
        ch = 1024
        self.bottleneck = nn.Sequential(
            LinBnHs(ci=1024 * 1 + 24, co=ch),
            LinBnHs(co=ch),
            LinBnHs(co=ch * 2)
        )

        self.decoder = nn.Sequential(
            Reshape(channels=512, resolution=2),
            ConvBnHs(ci=512, co=256, k=3, s=2, t=True),  # 2 -> 4
            ConvBnHs(co=256, k=3, s=2, t=True),  # -> 8
            ConvBnHs(co=128, k=5, s=2, t=True),  # -> 16
            ConvBnHs(co=64, k=5, s=2, t=True),  # -> 32
            ConvBnHs(co=32, k=3, s=2, t=True),  # -> 64
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, input_img, input_meta):
        x = self.encoder(input_img)
        x = self.combine(x, input_meta)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

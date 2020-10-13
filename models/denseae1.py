from abc import ABC
import torch.nn as nn
from .blocks import Reshape, IResBlock, Combine, LinBnHs


class DenseAE1(nn.Module, ABC):
    """
    Auto-encoder consisting out of 5 densely connected layers, with batch
    normalisation and relu.
    """

    lr_ideal = 1e-7

    def __init__(self):
        super().__init__()

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

        # dense version does not have a bottleneck

        # decoder
        ch = 1024
        self.decoder = nn.Sequential(
            LinBnHs(ci=ch * 1 + 24, co=ch),
            LinBnHs(co=ch),
            LinBnHs(co=ch * 2),
            LinBnHs(co=64*64),
            Reshape(channels=1, resolution=64)
        )

    def forward(self, input_img, input_meta):
        x = self.encoder(input_img)
        x = self.combine(x, input_meta)
        # <no bottleneck>
        x = self.decoder(x)
        return x

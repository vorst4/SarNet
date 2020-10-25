from abc import ABC

import torch.nn as nn

from .blocks import ConvBnHs, Combine, LinBnHs, IResBlock, Reshape, \
    InverseResidualEncoder
from util.timer import Timer
from util.log import Log


class ConvAE2(nn.Module, ABC):
    """
    Convolutional Auto-encoder 1
        bottleneck: 1 dense connected layer, batch norm and relu
        decoder: 5 times (transpose conv2d, batch norm, relu)
    """

    lr_ideal = 1e-4

    def __init__(self):
        super().__init__()

        channels = 1024

        # layers
        self.encoder = nn.Sequential(
            # block 1 (64x64) -> (32x32)
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.Hardswish(),
            # block 2 (32x32) -> (16x16)
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.Hardswish(),
            # block 3 (16x16) -> (8x8)
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=7,
                      stride=2,
                      padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.Hardswish(),
            # block 4 (8x8) -> (4x4)
            nn.Conv2d(in_channels=128,
                      out_channels=512,
                      kernel_size=7,
                      stride=2,
                      padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.Hardswish(),
        )

        # combination of encoder and meta-data input
        self.combine = Combine()

        # bottleneck
        ch = 1024
        self.bottleneck = nn.Sequential(
            LinBnHs(ci=512 * 4 * 4 + 24, co=512 * 4 * 4),
        )

        self.decoder = nn.Sequential(
            Reshape(co=228, ro=1),
            # block 1
            nn.ConvTranspose2d(in_channels=228,
                               out_channels=448,
                               kernel_size=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(448),
            nn.ReLU(),
            # block 2
            nn.ConvTranspose2d(in_channels=448,
                               out_channels=256,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # block 3
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.ReLU(),
            # block 4
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.ReLU(),
            # block 5
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=1,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.ReLU(),
            # upsample the 28x28px image to 32x32
            nn.Upsample(size=(32, 32), mode='bicubic', align_corners=False)
        )

    def forward(self, input_img, input_meta):
        timer = Timer(Log(None)).start()
        x = self.encoder(input_img)
        timer.stop('\t\tforwarded through encoder')
        timer.start()
        x = self.combine(x, input_meta)
        timer.stop('\t\tforwarded through combine')
        timer.start()
        x = self.bottleneck(x)
        timer.stop('\t\tforwarded through bottleneck')
        timer.start()
        x = self.decoder(x)
        timer.stop('\t\tforwarded through decoder')
        return x

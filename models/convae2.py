from abc import ABC

import torch.nn as nn

from .blocks import ConvBnHs, Combine, LinBnHs, InvResBlock, Reshape, \
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
        self.encoder = []
        self.decoder = []

        # layers

        # block 1 (32x32) -> (16x16)
        self.encoder.append(nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.Hardswish()
        ))
        # block 2 (16x16) -> (8x8)
        self.encoder.append(nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.Hardswish()
        ))
        # block 3 (8x8) -> (4x4)
        self.encoder.append(nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=256,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.BatchNorm2d(num_features=256),
            nn.Hardswish(),
        ))
        # block 3 (8x8) -> (4x4)
        self.encoder.append(nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=4,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(num_features=256),
            nn.Hardswish(),
        ))

        # combination of encoder and meta-data input
        self.combine = Combine()

        # bottleneck
        self.bottleneck = nn.Sequential(
            LinBnHs(ci=256 + 24, co=228),
        )

        # decoder
        self.decoder.append(
            Reshape(co=228, ro=1)
        )
        self.decoder.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels=228,
                               out_channels=448,
                               kernel_size=2,
                               stride=1,
                               bias=False),
            nn.BatchNorm2d(448),
            nn.ReLU()
        ))
        self.decoder.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels=448,
                               out_channels=256,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ))
        self.decoder.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.ReLU()
        ))
        self.decoder.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.ReLU()
        ))
        self.decoder.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=1,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.ReLU()
        ))
        self.decoder.append(
            # upsample the 28x28px image to 32x32
            nn.Upsample(size=(32, 32), mode='bicubic', align_corners=False)
        )

    def forward(self, input_img, input_meta):
        timer = Timer(Log(None)).start()
        x = input_img
        for block in self.encoder:
            x = block(x)
        timer.stop('\t\tforwarded through encoder')

        timer.start()
        x = self.combine(x, input_meta)
        timer.stop('\t\tforwarded through combine')

        timer.start()
        x = self.bottleneck(x)
        timer.stop('\t\tforwarded through bottleneck')

        timer.start()
        for block in self.decoder:
            x = block(x)
        timer.stop('\t\tforwarded through decoder')
        return x

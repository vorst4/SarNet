from abc import ABC

import torch.nn as nn

import models.blocks as blk
from util.timer import Timer
from util.log import Log
import settings
import torch


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
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.Hardswish()
        )
        # block 2 (16x16) -> (8x8)
        self.enc2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.Hardswish()
        )
        # block 3 (8x8) -> (4x4)
        self.enc3 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=256,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.BatchNorm2d(num_features=256),
            nn.Hardswish(),
        )
        # block 3 (8x8) -> (4x4)
        self.enc4 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=4,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(num_features=256),
            nn.Hardswish(),
        )

        # combination of encoder and meta-data input
        self.flatten = nn.Flatten()
        self.combine = blk.Combine()
        # self.unflatten = nn.Unflatten()

        # bottleneck
        self.bottleneck = nn.Sequential(
            blk.LinBnHs(ci=256 + 24, co=228),
        )

        # decoder
        self.dec1 = blk.Reshape(co=228, ro=1)

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=228,
                               out_channels=448,
                               kernel_size=2,
                               stride=1,
                               bias=False),
            nn.BatchNorm2d(448),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=448,
                               out_channels=256,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.ReLU()
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.ReLU()
        )
        self.dec6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=1,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.ReLU()
        )
        # upsample the 28x28px image to 32x32
        self.dec7 = nn.Upsample(size=(32, 32),
                                mode='bicubic',
                                align_corners=False)

    def forward(self, input_img, input_meta):
        timer = Timer(Log(None)).start()
        x = input_img
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        print(x.shape)

        # x = self.combine(x, input_meta)
        n = x.shape[0]
        print(input_meta.reshape(n, -1, 1, 1).shape)
        x = torch.cat([x, input_meta.reshape(n, -1, 1, 1)], dim=1)
        print(x.shape)

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = self.dec6(x)
        x = self.dec7(x)

        return x

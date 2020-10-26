from abc import ABC

import torch
import torch.nn as nn

from util.timer import Timer
from util.log import Log
import models.blocks as blk
import settings


class ConvBlock(nn.Module, ABC):
    def __init__(self, ci, co, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=ci,
                              out_channels=co,
                              kernel_size=k,
                              stride=s,
                              padding=p)
        self.bn = nn.BatchNorm2d(num_features=co)
        self.hs = nn.Hardswish()

    def forward(self, x):
        return self.hs(self.bn(self.conv(x)))


class InvConvBlock(nn.Module, ABC):
    def __init__(self, ci, co, k=2, s=2, p=0, po=0):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=ci,
                                       out_channels=co,
                                       kernel_size=k,
                                       stride=s,
                                       padding=p,
                                       output_padding=po)
        self.bn = nn.BatchNorm2d(num_features=co)
        self.hs = nn.Hardswish()

    def forward(self, x):
        return self.hs(self.bn(self.conv(x)))


class Upsample(nn.Module, ABC):
    def __init__(self, co, res):
        super().__init__()
        self.upsample = nn.Upsample(size=res,
                                    mode='bicubic',
                                    align_corners=False)
        self.bn = nn.BatchNorm2d(num_features=co)
        self.hs = nn.Hardswish()

    def forward(self, x):
        return self.hs(self.bn(self.upsample(x)))


class Stage(nn.Module, ABC):
    def __init__(self, ci, co, dr):
        super().__init__()
        self.down = ConvBlock(ci=ci, co=ci, k=2, s=2, p=0)
        self.conv1 = ConvBlock(ci=ci, co=co, k=3, s=1, p=1)
        self.conv2 = ConvBlock(ci=co, co=co, k=3, s=1, p=1)
        self.drp = nn.Dropout2d(p=dr)

    def forward(self, x):
        return self.drp(self.conv2(self.conv1(self.down(x))))


class InvStage(nn.Module, ABC):
    def __init__(self, ci, co, mode, dr):
        super().__init__()
        self.conv1 = ConvBlock(ci=2 * ci, co=ci, k=3, s=1, p=1)
        self.conv2 = ConvBlock(ci=ci, co=ci, k=3, s=1, p=1)
        if mode == 'transpose':
            self.up = InvConvBlock(ci=ci, co=co, k=2, s=2, p=0)
        elif mode == 'bicubic':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bicubic',
                            align_corners=False),
                ConvBlock(ci=ci, co=co, k=1, s=1, p=0))
        else:
            raise ValueError('mode should either be transpose or bicubic')
        self.drp = nn.Dropout2d(p=dr)

    def forward(self, x1, x2):
        return self.drp(self.up(self.conv2(self.conv1(
            torch.cat((x1, x2), dim=1)
        ))))


class ConvUNet(nn.Module, ABC):
    """
    Convolutional Auto-encoder 1
        bottleneck: 1 dense connected layer, batch norm and relu
        decoder: 5 times (transpose conv2d, batch norm, relu)
    """

    lr_ideal = 1e-4

    def __init__(self):
        super().__init__()

        channels = 1024
        dr = settings.dropout_rate

        self.enc1 = nn.Sequential(
            ConvBlock(ci=3, co=64, k=3, s=1, p=1),
            ConvBlock(ci=64, co=64, k=3, s=1, p=1),
            nn.Dropout2d(p=dr)
        )
        self.enc2 = Stage(64, 128, dr=dr)  # 32 -> 16
        self.enc3 = Stage(128, 256, dr=dr)  # 16 -> 8
        self.enc4 = Stage(256, 512, dr=dr // 2)  # 8 -> 4
        self.enc5 = ConvBlock(ci=512, co=4096 - 24, k=4, s=1, p=0)  # 4 -> 1

        self.bn1 = blk.Combine()
        self.bn2 = blk.Reshape(co=4096, ro=1)
        self.bn3 = InvConvBlock(ci=4096, co=512, k=4, s=1, p=0)  # 1 -> 4

        self.dec1 = InvStage(ci=512, co=256, mode='transpose', dr=dr//2)
        self.dec2 = InvStage(ci=256, co=128, mode='transpose', dr=dr)
        self.dec3 = InvStage(ci=128, co=64, mode='transpose', dr=dr)
        self.dec4 = nn.Sequential(
            ConvBlock(ci=128, co=64, k=3, s=1, p=1),
            ConvBlock(ci=64, co=64, k=3, s=1, p=1),
            ConvBlock(ci=64, co=1, k=1, s=1, p=0),
        )

    def forward(self, input_img, input_meta):
        x1 = self.enc1(input_img)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x = self.enc5(x4)

        x = self.bn1(x, input_meta)
        x = self.bn2(x)
        x = self.bn3(x)

        x = self.dec1(x, x4)
        x = self.dec2(x, x3)
        x = self.dec3(x, x2)
        x = self.dec4(torch.cat((x, x1), dim=1))

        return x

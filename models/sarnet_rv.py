import torch.nn as nn
from abc import ABC
import models.blocks as blk
import settings
import torch
from typing import Optional, List
from util.timer import Timer


class SarNetRV(nn.Module, ABC):
    lr_ideal = 1e-8

    def __init__(self):
        super().__init__()

        c = int(1024)  # channels
        r = settings.IMG_RESOLUTION  # resolution input
        n_meta = 24  # number of meta-data input variables
        d = settings.dropout_rate
        # d = 0
        n = 5

        self.encoder = nn.Sequential(
            # start:  32
            ConvHsDr(ci=3, co=c // 32, k=3, p=1, d=0),
            nn.Hardswish(),
            ResBlocks(c=c // 32, n=n, d=0),
            # stage 1: 32 -> 16
            Down(ci=c // 32, co=c // 16),
            ResBlocks(c=c // 16, n=n, d=0),
            # stage 2: 16 -> 8
            Down(ci=c // 16, co=c // 8),
            ResBlocks(c=c // 8, n=n, d=0),
            # stage 3: 8 -> 4
            Down(ci=c // 8, co=c // 4),
            ResBlocks(c=c // 4, n=n, d=0),
            # stage 4: 4 -> 1
            ConvHsDr(ci=c // 4, co=c - 2 * n_meta, k=4, p=0, d=0),
            nn.Hardswish(),
        )

        self.decoder = nn.Sequential(
            # inverted stage 4: 1 -> 4
            ConvTHsDr(ci=c // 2, co=c // 4, k=4, p=0, po=0, d=d),
            nn.Hardswish(),
            # inverted stage 3: 4 -> 8
            ResBlocks(c=c // 4, n=n, d=d),
            Up(ci=c // 4, co=c // 8),
            # inverted stage 2: 8 -> 16
            ResBlocks(c=c // 8, n=n, d=d),
            Up(ci=c // 8, co=c // 16),
            # inverted stage 2: 16 -> 32
            ResBlocks(c=c // 16, n=n, d=d),
            Up(ci=c // 16, co=c // 32),
            # end
            ResBlocks(c=c // 32, n=n, d=d),
            nn.Conv2d(in_channels=c//32, out_channels=1, kernel_size=1),
        )

    def forward(self, input_img, input_meta):
        # encoder
        x = self.encoder(input_img)

        # bottleneck
        n_batch, n_par = x.shape[0], x.shape[1] // 2
        mu, var = x[:, :n_par, :], x[:, n_par:, :]
        std = torch.exp(0.5 * var)
        if self.training:
            x = mu + std * torch.randn_like(std)
        else:
            x = mu + std ** 2
        # concatenate with applicator settings
        x = torch.cat([x, input_meta.view(x.shape[0], -1, 1, 1)], dim=1)

        # decoder
        x = self.decoder(x)

        return {'y': x, 'mu': mu, 'var': var}


class Down(nn.Module, ABC):
    def __init__(self, ci, co):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=ci, out_channels=co, kernel_size=2,
                              stride=2, padding=0)
        self.hs = nn.Hardswish()

    def forward(self, x):
        return self.hs(self.conv(x))


class Up(nn.Module, ABC):
    def __init__(self, ci, co):
        super().__init__()
        self.convt = nn.ConvTranspose2d(in_channels=ci, out_channels=co,
                                        kernel_size=2, stride=2, padding=0)
        self.hs = nn.Hardswish()

    def forward(self, x):
        return self.hs(self.convt(x))


class ResBlock(nn.Module, ABC):
    def __init__(self, c, d):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3,
                               padding=1)
        self.activation1 = nn.Hardswish()
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3,
                               padding=1)
        self.activation2 = nn.Hardswish()
        if d is not 0:
            self.dropout = nn.Dropout2d(p=d)
        self.d = d

    def forward(self, x):
        i = x
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        if self.d is not 0:
            x = self.dropout(x)
        return x + i


class ResBlocks(nn.Module, ABC):
    def __init__(self, c, d, n):
        super().__init__()
        blocks = []
        for idx in range(n):
            blocks.append(ResBlock(c, d))
        self.resblocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.resblocks(x)


class ConvHsDr(nn.Module, ABC):
    def __init__(self, ci, co, k, p, d, s=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=ci,
                              out_channels=co,
                              kernel_size=k,
                              padding=p,
                              stride=s)
        self.hardswish = nn.Hardswish()
        if d is not 0:
            self.dropout = nn.Dropout2d(p=d)
        self.d = d

    def forward(self, x):
        x = self.conv(x)
        x = self.hardswish(x)
        if self.d is not 0:
            x = self.dropout(x)
        return x


class ConvTHsDr(nn.Module, ABC):
    def __init__(self, ci, co, k, p, po, d, s=1):
        super().__init__()
        self.convt = nn.ConvTranspose2d(in_channels=ci,
                                        out_channels=co,
                                        kernel_size=k,
                                        padding=p,
                                        output_padding=po,
                                        stride=s)
        self.hardswish = nn.Hardswish()
        if d is not 0:
            self.dropout = nn.Dropout2d(p=d)
        self.d = d

    def forward(self, x):
        x = self.convt(x)
        x = self.hardswish(x)
        if self.d is not 0:
            x = self.dropout(x)
        return x

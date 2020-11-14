import torch.nn as nn
from abc import ABC
import models.blocks as blk
import settings
import torch
from typing import Optional, List
from util.timer import Timer
from math import gcd
from models.sarnet_rv import ConvHs, ConvHsDr, TConvHsDr, Down, Conv


class SarNetRV4(nn.Module, ABC):
    """
    mobile inverse blocks variant of SarNetRV
    """
    lr_ideal = 1e-3

    def __init__(
            self,
            d: float = 0.2,
            n: int = 2,
            train_enc: bool = None
    ):
        super().__init__()

        c = int(1024)  # channels
        r = settings.IMG_RESOLUTION  # resolution input
        n_meta = 24  # number of meta-data input variables
        self.train_enc = train_enc
        self.encoder = Encoder(c=c, n=n, n_meta=n_meta,
                               train_enc=settings.use_ae_dataset)
        self.decoder = Decoder(c=c, n=n, d=d,
                               train_encoder=settings.use_ae_dataset)

    def forward(self, xp, xs):

        xp = self.encoder(xp)

        # bottleneck
        n_batch, n_par = xp.shape[0], xp.shape[1] // 2
        mu, var = xp[:, :n_par, :], xp[:, n_par:, :]
        std = torch.exp(0.5 * var)
        if self.training:
            xp = mu + std * torch.randn_like(std)
        else:
            xp = mu + std ** 2
        # concatenate with applicator settings
        xp = torch.cat([xp, xs.view(xp.shape[0], -1, 1, 1)], dim=1)

        # decoder
        xp = self.decoder(xp)
        return {'y': xp, 'var': var, 'mu': mu}


class Encoder(nn.Module, ABC):
    def __init__(self, c, n, n_meta, train_enc: bool):
        super().__init__()
        self.train_enc = train_enc
        self.encoder = nn.Sequential(
            # start:  32
            ConvHs(ci=3, co=c // 32, k=3, p=1),
            ResBlocks(c=c // 32, n=n),
            # stage 1: 32 -> 16
            Down(ci=c // 32, co=c // 16),
            ResBlocks(c=c // 16, n=n),
            # stage 2: 16 -> 8
            Down(ci=c // 16, co=c // 8),
            ResBlocks(c=c // 8, n=n),
            # stage 3: 8 -> 4
            Down(ci=c // 8, co=c // 4),
            ResBlocks(c=c // 4, n=n),
            # stage 4: 4 -> 1
            ConvHs(ci=c // 4, co=c - 2 * n_meta, k=4, p=0),
        )

    def update_grad(self):

        if settings.use_ae_dataset is False:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module, ABC):
    def __init__(self, c, n, d, train_encoder: bool):
        super().__init__()
        co = 3 if train_encoder else 1
        self.decoder = nn.Sequential(
            # inverted stage 4: 1 -> 4
            TConvHsDr(ci=c // 2, co=c // 4, k=4, p=0, po=0, d=d),
            # inverted stage 3: 4 -> 8
            ResBnHsDrBlocks(c=c // 4, n=n, d=d),
            Up(ci=c // 4, co=c // 8),
            # inverted stage 2: 8 -> 16
            ResBnHsDrBlocks(c=c // 8, n=n, d=d),
            Up(ci=c // 8, co=c // 16),
            # inverted stage 2: 16 -> 32
            ResBnHsDrBlocks(c=c // 16, n=n, d=d),
            Up(ci=c // 16, co=c // 32),
            # end
            ResBnHsDrBlocks(c=c // 32, n=n, d=d),
            Conv(ci=c // 32, co=co, k=1, p=0),
        )

    def forward(self, x):
        return self.decoder(x)


class ResBlock(nn.Module, ABC):
    def __init__(self, c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3,
                               groups=c, padding=1)
        self.activation1 = nn.Hardswish()
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1,
                               padding=0)
        self.activation2 = nn.Hardswish()

    def forward(self, x):
        i = x
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        return x + i


class Up(nn.Module, ABC):
    def __init__(self, ci, co):
        super().__init__()
        self.convt = nn.ConvTranspose2d(in_channels=ci, out_channels=co,
                                        kernel_size=2, stride=2, padding=0,
                                        groups=gcd(ci, co))
        self.hs = nn.Hardswish()

    def forward(self, x):
        return self.hs(self.convt(x))


class ResBnHsDrBlock(nn.Module, ABC):
    """
    Residual + Dropout block with hardswish activation
    """

    def __init__(self, c, d):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3,
                               groups=c, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(c, **settings.batch_norm)
        self.activation1 = nn.Hardswish()

        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1,
                               padding=0, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(c, **settings.batch_norm)
        self.activation2 = nn.Hardswish()

        self.dropout = nn.Dropout2d(p=d)

    def forward(self, x):
        i = x
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.activation1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation2(x)

        x = self.dropout(x)
        return x + i


class ResBnHsDrBlocks(nn.Module, ABC):
    def __init__(self, c, d, n):
        super().__init__()
        blocks = []
        for idx in range(n):
            blocks.append(ResBnHsDrBlock(c, d))
        self.res_dr_blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.res_dr_blocks(x)


class ResBlocks(nn.Module, ABC):
    def __init__(self, c, n):
        super().__init__()
        blocks = []
        for idx in range(n):
            blocks.append(ResBlock(c))
        self.resblocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.resblocks(x)

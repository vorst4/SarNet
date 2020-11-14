import torch.nn as nn
from abc import ABC
import models.blocks as blk
import settings
import torch
from typing import Optional, List
from util.timer import Timer
from math import gcd
from models.sarnet_rv import ConvHs, ConvHsDr, TConvHsDr, Down, Up, Conv


class SarNetRV4(nn.Module, ABC):
    """
    mobile inverse blocks variant of SarNetRV
    """
    lr_ideal = 1e-8

    def __init__(
            self,
            d: float = 0.2,
            n: int = 2
    ):
        super().__init__()

        c = int(1024)  # channels
        r = settings.IMG_RESOLUTION  # resolution input
        n_meta = 24  # number of meta-data input variables

        self.encoder = nn.Sequential(
            # start:  32
            ConvHs(ci=3, co=c // 32, k=3, p=1),
            ResBlocks(c=c // 32, n=n, res=32),
            # stage 1: 32 -> 16
            Down(ci=c // 32, co=c // 16),
            ResBlocks(c=c // 16, n=n, res=16),
            # stage 2: 16 -> 8
            Down(ci=c // 16, co=c // 8),
            ResBlocks(c=c // 8, n=n, res=8),
            # stage 3: 8 -> 4
            Down(ci=c // 8, co=c // 4),
            ResBlocks(c=c // 4, n=n, res=4),
            # stage 4: 4 -> 1
            ConvHsDr(ci=c // 4, co=c - 2 * n_meta, k=4, p=0, d=d),
        )

        self.decoder = nn.Sequential(
            # inverted stage 4: 1 -> 4
            TConvHsDr(ci=c // 2, co=c // 4, k=4, p=0, po=0, d=d),
            # inverted stage 3: 4 -> 8
            ResBlocks(c=c // 4, n=n, d=d, res=4),
            Up(ci=c // 4, co=c // 8),
            # inverted stage 2: 8 -> 16
            ResBlocks(c=c // 8, n=n, d=d, res=8),
            Up(ci=c // 8, co=c // 16),
            # inverted stage 2: 16 -> 32
            ResBlocks(c=c // 16, n=n, d=d, res=16),
            Up(ci=c // 16, co=c // 32),
            # end
            ResBlocks(c=c // 32, n=n, d=d, res=32),
            Conv(ci=c // 32, co=1, k=1, p=0),
        )


class ResBlocks(nn.Module, ABC):
    def __init__(self, c, n, res, d=0.):
        super().__init__()
        blocks = []
        for idx in range(n):
            blocks.append(blk.InvResBlock(
                ci=c,
                co=c,
                k=3,
                p=1,
                dropout=d,
                ri=res,
                squeeze=False,
                expand=False
            ))
        self.res_blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.res_blocks(x)


class SarNetRV5(nn.Module, ABC):
    """
    SarNetRV4 without dropout
    """

    def __init__(self):
        super().__init__(d=0)

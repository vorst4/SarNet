import torch.nn as nn
from abc import ABC
import models.blocks as blk
import settings
import torch
from util.timer import Timer


def rep_res(ci, co, k, d, r, n, p=1, sq=False, ex=False):
    blocks = []
    for idx in range(n):
        blocks.append(blk.InvResBlock(
            ci=ci if idx is 0 else co,
            co=co,
            k=k,
            p=p,
            dropout=d,
            ri=r,
            squeeze=sq,
            expand=ex,
        ))
    return nn.Sequential(*blocks)


class _SarNetR(nn.Module, ABC):

    def __init__(self, skip: bool, sq: bool, ex: bool):
        super().__init__()

        a = 2 if skip else 1
        c = int(1024 * 1.5)  # channels
        r = settings.IMG_RESOLUTION  # resolution input
        n_meta = 24  # number of meta-data input variables
        d = settings.dropout_rate
        n = 2

        self.encoder = nn.Sequential(
            # start
            rep_res(ci=3, co=c // 32, k=3, d=d, r=r, n=n),
            # 32 --> 16
            _Down(ci=c // 32, co=c // 32, ri=r, d=d, sq=sq, ex=ex, n=n),
            # 16 --> 8
            _Down(ci=c // 32, co=c // 8, ri=r // 2, d=d, sq=sq, ex=ex, n=n),
            # 8 --> 4
            _Down(ci=c // 8, co=c // 4, ri=r // 4, d=d, sq=sq, ex=ex, n=n),
            # 4 --> 1
            blk.ConvBnHsDr(ci=c // 4, co=c - n_meta, k=4, p=0, po=0, s=1,
                           d=d),
        )

        self.decoder = nn.Sequential(
            # 1 -> 4
            blk.ConvBnHsDr(ci=c, co=c // 4, k=4, s=1, p=0, po=0, t=True, d=d),
            # 4 -> 8
            _Up(ci=c // 4 * a, co=c // 8, ri=r // 8, d=d, sq=sq, ex=ex, n=n),
            # 8 -> 16
            _Up(ci=c // 8 * a, co=c // 32, ri=r // 4, d=d, sq=sq,
                ex=ex, n=n),
            # 16 -> 32
            _Up(ci=c // 32 * a, co=c // 32, ri=r // 2, d=d,
                sq=sq, ex=ex, mode='bicubic', n=n),
            # end
            nn.Sequential(
                rep_res(ci=c // 32 * a, co=c // 32, k=3, d=d, r=r, n=n),
                nn.Conv2d(in_channels=c // 32, out_channels=1, kernel_size=1),
            ),
        )


class _Down(nn.Module, ABC):

    def __init__(self, ci, co, ri, d, ex, sq, n):
        super().__init__()
        self.down = blk.ConvBnHs(ci=ci, co=ci, k=2, g=ci, p=0, s=2)
        self.res = rep_res(ci=ci, co=co, k=3, p=1, d=d, r=ri // 2, ex=ex,
                           sq=sq, n=n)

    def forward(self, x):
        return self.res(self.down(x))


class _Up(nn.Module, ABC):

    def __init__(self, ci, co, ri, d, ex, sq, n, mode='transpose'):
        super().__init__()
        self.res = rep_res(ci=ci, co=co, k=3, p=1, d=d, r=ri, ex=ex, sq=sq,
                           n=n)
        self.up = self._get_up(co, ri, mode)

    @staticmethod
    def _get_up(co, ri, mode):
        if mode is 'transpose':
            return blk.ConvBnHs(ci=co, co=co, k=2, g=co, p=0, po=0, s=2,
                                b=False, t=True)
        elif mode is 'bicubic':
            return nn.Upsample(size=(ri * 2, ri * 2),
                               mode='bicubic',
                               align_corners=False)
        else:
            raise ValueError('invalid mode, can only be tranpose or bicubic')

    def forward(self, x):
        return self.up(self.res(x))


class SarNetRS(_SarNetR, ABC):
    lr_ideal = 1e-5

    def __init__(self, sq=False, ex=False):
        super().__init__(skip=True, sq=sq, ex=ex)

    def forward(self, input_img, input_meta):
        # encoder
        x1 = self.encoder[0](input_img)
        x2 = self.encoder[1](x1)
        x3 = self.encoder[2](x2)
        x4 = self.encoder[3](x3)
        x = self.encoder[4](x4)

        # bottleneck
        x = torch.cat([x, input_meta.reshape(x.shape[0], -1, 1, 1)], dim=1)

        # decoder
        x = self.decoder[0](x)
        x = self.decoder[1](torch.cat([x, x4], dim=1))
        x = self.decoder[2](torch.cat([x, x3], dim=1))
        x = self.decoder[3](torch.cat([x, x2], dim=1))
        x = self.decoder[4](torch.cat([x, x1], dim=1))

        return x


class SarNetRN(_SarNetR, ABC):
    lr_ideal = 1e-5

    def __init__(self, sq=False, ex=False):
        super().__init__(skip=False, sq=sq, ex=ex)

    def forward(self, input_img, input_meta):
        # encoder
        x = self.encoder(input_img)

        # bottleneck
        x = torch.cat([x, input_meta.reshape(x.shape[0], -1, 1, 1)], dim=1)

        # decoder
        x = self.decoder(x)

        return x

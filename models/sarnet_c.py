from abc import ABC

import torch
import torch.nn as nn

from util.timer import Timer
from util.log import Log
import models.blocks as blk
import settings


def rep_conv(ci, co, k, s, p, d, n):
    blocks = []
    for idx in range(n):
        ci = ci if idx is 0 else co
        blocks.append(blk.ConvBnHsDr(ci=ci, co=co, k=k, s=s, p=p, d=d))
    return nn.Sequential(*blocks)


class _SarNetC(nn.Module, ABC):

    def __init__(self, skip: bool):
        super().__init__()

        a = 2 if skip else 1
        c = int(1024 * 1.5)
        r = settings.IMG_RESOLUTION  # resolution input
        n_meta = 24  # number of meta-data input variables
        d = settings.dropout_rate
        n = 2

        self.encoder = nn.Sequential(
            # start
            rep_conv(ci=3, co=c // 32, k=3, s=1, p=1, d=d, n=2),
            # 32 -> 16
            _Down(c // 32, c // 16, d=d, n=n),
            # 16 -> 8
            _Down(c // 16, c // 8, d=d, n=n),
            # 8 -> 4
            _Down(c // 8, c // 4, d=d, n=n),
            # 4 -> 1
            blk.ConvBnHsDr(ci=c // 4, co=c - n_meta, k=4, s=1, p=0, d=d),
        )

        self.decoder = nn.Sequential(
            # 1 -> 4
            blk.ConvBnHsDr(ci=c, co=c // 4, k=4, s=1, p=0, po=0, t=True, d=d),
            # 4 -> 8
            _Up(ci=c // 4 * a, co=c // 8, ri=r // 8, m='transpose', d=d, n=n),
            # 8 -> 16
            _Up(ci=c // 8 * a, co=c // 16, ri=r // 4, m='transpose', d=d, n=n),
            # 16 -> 32
            _Up(ci=c // 16 * a, co=c // 32, ri=r // 2, m='transpose', d=d,
                n=n),
            # end
            nn.Sequential(
                rep_conv(ci=c // 32 * a, co=c // 32, k=3, s=1, p=1, d=d, n=n),
                nn.Conv2d(in_channels=c // 32, out_channels=1, kernel_size=1),
            ),
        )


class _Down(nn.Module, ABC):
    def __init__(self, ci, co, d, n):
        super().__init__()
        self.down = blk.ConvBnHs(ci=ci, co=ci, k=2, g=ci, p=0, s=2)
        self.conv = rep_conv(ci=ci, co=co, k=3, s=1, p=1, d=d, n=n)

    def forward(self, x):
        return self.conv(self.down(x))


class _Up(nn.Module, ABC):
    def __init__(self, ci, co, ri, m, d, n):
        super().__init__()
        self.conv = rep_conv(ci=ci, co=co, k=3, s=1, p=1, d=d, n=n)
        self.up = self._get_up(co, ri, m)

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
        return self.up(self.conv(x))


# class SarNetCS(_SarNetC, ABC):
#     lr_ideal = 1e-6
#
#     def __init__(self):
#         super().__init__(skip=True)
#
#     def forward(self, input_img, input_meta):
#         # encoder
#         x1 = self.encoder[0](input_img)
#         x2 = self.encoder[1](x1)
#         x3 = self.encoder[2](x2)
#         x4 = self.encoder[3](x3)
#         x = self.encoder[4](x4)
#
#         # bottleneck
#         x = torch.cat([x, input_meta.reshape(x.shape[0], -1, 1, 1)], dim=1)
#
#         # decoder
#         x = self.decoder[0](x)
#         x = self.decoder[1](torch.cat([x, x4], dim=1))
#         x = self.decoder[2](torch.cat([x, x3], dim=1))
#         x = self.decoder[3](torch.cat([x, x2], dim=1))
#         x = self.decoder[4](torch.cat([x, x1], dim=1))
#
#         return x


class SarNetC(_SarNetC, ABC):
    lr_ideal = 1e-7

    def __init__(self):
        super().__init__(skip=False)

    def forward(self, input_img, input_meta):
        # encoder
        x = self.encoder(input_img)

        # bottleneck
        x = torch.cat([x, input_meta.reshape(x.shape[0], -1, 1, 1)], dim=1)

        # decoder
        x = self.decoder(x)

        return x

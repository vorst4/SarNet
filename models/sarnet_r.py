import torch.nn as nn
from abc import ABC
import models.blocks as blk
import settings
import torch
from util.timer import Timer


class _Down(nn.Module, ABC):

    def __init__(self, ci, co, ri, dr, ex, sq, n):
        super().__init__()
        self.down = nn.Conv2d(
            in_channels=ci,
            out_channels=ci,
            kernel_size=2,
            groups=ci,
            stride=2,
            bias=False
        )
        blocks = []
        for idx in range(n):
            blocks.append(blk.InvResBlock(
                ci=ci if idx is 0 else co,
                co=co,
                k=3,
                dropout=dr,
                ri=ri // 2,
                expand=ex,
                squeeze=sq
            ))
        self.res = nn.Sequential(*blocks)

    def forward(self, x):
        return self.res(self.down(x))


class _Up(nn.Module, ABC):

    def __init__(self, ci, co, ri, dr, ex, sq, n, mode='transpose'):
        super().__init__()
        blocks = []
        for idx in range(n):
            blocks.append(blk.InvResBlock(
                ci=ci if idx is 0 else co,
                co=co,
                k=3,
                dropout=dr,
                ri=ri,
                expand=ex,
                squeeze=sq
            ))
        self.res = nn.Sequential(*blocks)
        self.up = self._get_up(co, ri, mode)

    @staticmethod
    def _get_up(co, ri, mode):
        if mode is 'transpose':
            return nn.ConvTranspose2d(in_channels=co,
                                      out_channels=co,
                                      kernel_size=2,
                                      groups=co,
                                      stride=2,
                                      bias=False)
        elif mode is 'bicubic':
            return nn.Upsample(size=(ri * 2, ri * 2),
                               mode='bicubic',
                               align_corners=False)
        else:
            raise ValueError('invalid mode, can only be tranpose or bicubic')

    def forward(self, x):
        return self.up(self.res(x))


class _SarNetR(nn.Module, ABC):

    def __init__(self, skip: bool, sq: bool, ex: bool):
        super().__init__()

        a = 2 if skip else 1
        c = 1024  # channels
        r = settings.IMG_RESOLUTION  # resolution input
        ni_meta = 24  # number of meta-data input variables
        d = settings.dropout_rate

        # encoder
        self.encoder = nn.Sequential(
            # start
            blk.InvResBlock(
                ci=3,
                co=c // 32,
                k=3,
                dropout=d,
                ri=r,
                expand=False,
                squeeze=False
            ),
            # 32 --> 16
            _Down(ci=c // 32, co=c // 32, ri=r, dr=d, sq=sq, ex=ex, n=3),
            # 16 --> 8
            _Down(ci=c // 32, co=c // 8, ri=r // 2, dr=d, sq=sq, ex=ex, n=3),
            # 8 --> 4
            _Down(ci=c // 8, co=c // 4, ri=r // 4, dr=d, sq=sq, ex=ex, n=3),
            # 4 --> 1
            blk.ConvBnHsDr(ci=c // 4, co=c - ni_meta, k=4, s=1, dr=d),
        )

        self.decoder = nn.Sequential(
            # 1 --> 4
            blk.ConvBnHsDr(ci=c, co=c // 4, k=4, s=1, t=True, dr=d),
            # 4 --> 8
            _Up(ci=c // 4 * a, co=c // 8, ri=r // 8, dr=d, sq=sq, ex=ex, n=3),
            # 8 --> 16
            _Up(ci=c // 8 * a, co=c // 32, ri=r // 4, dr=d, sq=sq,
                ex=ex, n=3),
            # 16 --> 32
            _Up(ci=c // 32 * a, co=c // 32, ri=r // 2, dr=d,
                sq=sq, ex=ex, mode='bicubic', n=3),
            # end
            blk.InvResBlock(
                ci=c // 32 * a,
                co=c // 32,
                k=3,
                dropout=d,
                squeeze=False,
                expand=False
            ),
            nn.Conv2d(
                in_channels=c // 32,
                out_channels=1,
                kernel_size=1,
                padding=0,
                bias=True
            ),
        )


class SarNetRS(_SarNetR, ABC):
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
        x = torch.cat([x, input_meta.view(x.shape[0], -1, 1, 1)], dim=1)

        # decoder
        x = self.decoder[0](x)
        x = self.decoder[1](torch.cat([x, x4], dim=1))
        x = self.decoder[2](torch.cat([x, x3], dim=1))
        x = self.decoder[3](torch.cat([x, x2], dim=1))
        x = self.decoder[4](torch.cat([x, x1], dim=1))
        x = self.decoder[5](x)

        return x


class SarNetRN(_SarNetR, ABC):
    def __init__(self, sq=False, ex=False):
        super().__init__(skip=False, sq=sq, ex=ex)

    def forward(self, input_img, input_meta):
        # encoder
        x = self.encoder(input_img)

        # bottleneck
        x = torch.cat([x, input_meta.view(x.shape[0], -1, 1, 1)], dim=1)

        # decoder
        x = self.decoder(x)

        return x


class SarNetRSSE(SarNetRS, ABC):
    def __init__(self):
        super().__init__(sq=True, ex=True)


class SarNetRNSE(SarNetRN, ABC):
    def __init__(self):
        super().__init__(sq=True, ex=True)

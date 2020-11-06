from abc import ABC

import torch
import torch.nn as nn

from util.timer import Timer
from util.log import Log
import models.blocks as blk
import settings


class _SarNetC(nn.Module, ABC):
    """
    Convolutional Auto-encoder 1
        bottleneck: 1 dense connected layer, batch norm and relu
        decoder: 5 times (transpose conv2d, batch norm, relu)
    """

    def __init__(self, skip: bool):
        super().__init__()

        a = 2 if skip else 1
        c = int(1024 * 1.5)
        r = settings.IMG_RESOLUTION  # resolution input
        n_meta = 24  # number of meta-data input variables
        dr = settings.dropout_rate

        self.encoder = nn.Sequential(
            # start
            nn.Sequential(
                blk.ConvBnHsDr(ci=3, co=c // 32, k=3, s=1, p=1, dr=dr),
                blk.ConvBnHsDr(ci=c // 32, co=c // 32, k=3, s=1, p=1, dr=dr),
            ),
            # 32 -> 16
            _Down(c // 32, c // 16, dr=dr),
            # 16 -> 8
            _Down(c // 16, c // 8, dr=dr),
            # 8 -> 4
            _Down(c // 8, c // 4, dr=dr),
            # 4 -> 1
            blk.ConvBnHsDr(ci=c // 4, co=c - n_meta, k=4, s=1, p=0, dr=dr),
        )

        self.decoder = nn.Sequential(
            # 1 -> 4
            blk.ConvBnHsDr(ci=c, co=c//4, k=4, s=1, p=0, po=0, t=True, dr=dr),
            # 4 -> 8
            _Up(ci=c // 4 * a, co=c // 8, ri=r // 8, m='transpose', dr=dr),
            # 8 -> 16
            _Up(ci=c // 8 * a, co=c // 16, ri=r // 4, m='transpose', dr=dr),
            # 16 -> 32
            _Up(ci=c // 16 * a, co=c // 32, ri=r // 2, m='transpose', dr=dr),
            # end
            nn.Sequential(
                blk.ConvBnHsDr(ci=c // 32 * a, co=c // 32, k=3, s=1, p=1,
                               dr=dr),
                blk.ConvBnHsDr(ci=c // 32, co=c // 32, k=3, s=1, p=1, dr=dr),
                nn.Conv2d(in_channels=c // 32, out_channels=1, kernel_size=1),
            ),
        )


class _Down(nn.Module, ABC):
    def __init__(self, ci, co, dr):
        super().__init__()
        self.down = blk.ConvBnHs(ci=ci, co=ci, k=2, g=ci, p=0, s=2)
        self.conv1 = blk.ConvBnHsDr(ci=ci, co=co, k=3, s=1, p=1, dr=dr)
        self.conv2 = blk.ConvBnHsDr(ci=co, co=co, k=3, s=1, p=1, dr=dr)

    def forward(self, x):
        return self.conv2(self.conv1(self.down(x)))


class _Up(nn.Module, ABC):
    def __init__(self, ci, co, ri, m, dr):
        super().__init__()
        self.conv1 = blk.ConvBnHsDr(ci=ci, co=co, k=3, s=1, p=1, dr=dr)
        self.conv2 = blk.ConvBnHsDr(ci=co, co=co, k=3, s=1, p=1, dr=dr)
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
        return self.up(self.conv2(self.conv1(x)))


class SarNetCS(_SarNetC, ABC):
    lr_ideal = 1e-6  # 1e-5 is also stable

    def __init__(self):
        super().__init__(skip=True)

    def forward(self, input_img, input_meta):
        x = input_img

        # encoder
        x1 = self.encoder[0](x)
        x2 = self.encoder[1](x1)
        x3 = self.encoder[2](x2)
        x4 = self.encoder[3](x3)
        x = self.encoder[4](x4)

        # bottleneck
        n = x.shape[0]
        x = torch.cat([x, input_meta.reshape(n, -1, 1, 1)], dim=1)

        # decoder
        x = self.decoder[0](x)
        x = self.decoder[1](torch.cat([x, x4], dim=1))
        x = self.decoder[2](torch.cat([x, x3], dim=1))
        x = self.decoder[3](torch.cat([x, x2], dim=1))
        x = self.decoder[4](torch.cat([x, x1], dim=1))

        return x


class SarNetCN(_SarNetC, ABC):
    lr_ideal = 1e-6  # 1e-5 is also stable

    def __init__(self):
        super().__init__(skip=False)

    def forward(self, input_img, input_meta):
        x = self.encoder(input_img)

        # bottleneck
        x = torch.cat([x, input_meta.reshape(x.shape[0], -1, 1, 1)], dim=1)

        # decoder
        x = self.decoder(x)

        return x

import torch.nn as nn
from abc import ABC
import models.blocks as blk
import settings
import torch
from util.timer import Timer


class ConvBnHs(nn.Module, ABC):
    """
    (Transpose) Convolutional filter + batch normalisation + hard swish
    """

    def __init__(
            self,
            ci: int,  # input channels
            co: int,  # output channels
            k: int = 2,  # kernel size
            s: int = 2,  # stride
            p: int = 0,  # padding, if None -> padding is self determined
            po: int = 0,  # output_padding (only applies for transpose conv)
            t: bool = False  # transpose convolution or not
    ):
        super().__init__()

        # use either normal or transpose conv filter
        if not t:
            self.conv = nn.Conv2d(in_channels=ci,
                                  out_channels=co,
                                  kernel_size=k,
                                  stride=s,
                                  padding=p,
                                  bias=False)
        else:
            self.conv = nn.ConvTranspose2d(in_channels=ci,
                                           out_channels=co,
                                           kernel_size=k,
                                           stride=s,
                                           padding=p,
                                           output_padding=po,
                                           bias=False)
        self.bn = nn.BatchNorm2d(num_features=co)
        self.hs = nn.Hardswish()

    def forward(self, x):
        # return self.hs(self.bn(self.conv(x)))
        return self.conv(x)


class Down(nn.Module, ABC):

    def __init__(self, ci, co, ri, sq=False, ex=False):
        super().__init__()
        d = settings.dropout_rate
        self.down = ConvBnHs(ci=ci, co=ci)
        self.res = blk.InvResBlock(ci=ci, co=co, k=3, dropout=d,
                                   ri=ri // 2,
                                   expand=ex,
                                   squeeze=sq)

    def forward(self, x):
        return self.res(self.down(x))


class Up(nn.Module, ABC):

    def __init__(self, ci, co, ri, mode='transpose', sq=False, ex=False):
        super().__init__()
        d = settings.dropout_rate
        self.res = blk.InvResBlock(ci=ci, co=co, k=3, dropout=d,
                                   ri=ri,
                                   expand=ex,
                                   squeeze=sq)
        if mode is 'transpose':
            self.up = ConvBnHs(ci=co, co=co, t=True)
        elif mode is 'bicubic':
            self.up = nn.Upsample(size=(ri * 2, ri * 2),
                                  mode='bicubic',
                                  align_corners=False)
        else:
            raise ValueError('invalid mode, can only be tranpose or bicubic')

    def forward(self, x):
        return self.up(self.res(x))


class SarNetRSc(nn.Module, ABC):
    lr_ideal = 1e-5

    def __init__(self):
        super().__init__()

        c = 1024  # channels
        r = settings.IMG_RESOLUTION  # resolution input
        ni_meta = 24  # number of meta-data input variables
        d = settings.dropout_rate

        # encoder
        self.enc1 = blk.InvResBlock(ci=3, co=c // 32, k=3, dropout=d,
                                    ri=r,
                                    expand=False,
                                    squeeze=False)
        # 32 --> 16
        self.enc2 = Down(ci=c // 32, co=c // 32, sq=False, ex=False, ri=r)
        # 16 --> 8
        self.enc3 = Down(ci=c // 32, co=c // 8, ri=r // 2)
        # 8 --> 4
        self.enc4 = Down(ci=c // 8, co=c // 4, ri=r // 4)
        # 4 --> 1
        self.enc5 = ConvBnHs(ci=c // 4, co=c - ni_meta, k=4, s=1)

        # decoder
        # 1 --> 4
        self.dec1 = ConvBnHs(ci=c, co=c // 4, k=4, s=1, t=True)
        # 4 --> 8
        self.dec2 = Up(ci=c // 4 * 2, co=c // 8, ri=r // 8)
        # 8 --> 16
        self.dec3 = Up(ci=c // 8 * 2, co=c // 32, ri=r // 4)
        # 16 --> 32
        self.dec4 = Up(ci=c // 32 * 2, co=c // 32, ri=r // 2, mode='bicubic')
        self.dec5 = blk.InvResBlock(ci=c // 32 * 2,
                                    co=c // 32, k=3,
                                    dropout=d,
                                    squeeze=False, expand=False)
        self.dec6 = blk.Conv2d(co=1, k=3, p=1, bias=True)

    def forward(self, input_img, input_meta):
        # encoder
        x = input_img
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x = self.enc5(x4)

        # bottleneck
        n = x.shape[0]
        x = torch.cat([x, input_meta.reshape(n, -1, 1, 1)], dim=1)

        # decoder
        x = self.dec1(x)
        x = self.dec2(torch.cat([x, x4], dim=1))
        x = self.dec3(torch.cat([x, x3], dim=1))
        x = self.dec4(torch.cat([x, x2], dim=1))
        x = self.dec5(torch.cat([x, x1], dim=1))
        x = self.dec6(x)

        return x

import torch.nn as nn
from abc import ABC
import models.blocks as blk
import settings
import torch


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
        return self.hs(self.bn(self.conv(x)))


class ResNetAE3(nn.Module, ABC):
    lr_ideal = 1e-5

    def __init__(self):
        super().__init__()

        c = 1024  # channels
        ri = settings.IMG_RESOLUTION  # resolution input
        ni_meta = 24  # number of meta-data input variables
        d = settings.dropout_rate

        self.enc1 = blk.InvResBlock(co=c // 32, k=3,
                                    dropout=d,
                                    expand=False,
                                    squeeze=False)
        self.enc2 = ConvBnHs(ci=c // 32, co=c // 32)  # 32 --> 16
        self.enc3 = blk.InvResBlock(co=c // 16, k=3,
                                    dropout=d,
                                    expand=False,
                                    squeeze=False)
        self.enc4 = ConvBnHs(ci=c // 16, co=c // 16)  # 16 --> 8
        self.enc5 = blk.InvResBlock(co=c // 8, k=3,
                                    dropout=d,
                                    expand=False,
                                    squeeze=False)
        self.enc6 = ConvBnHs(ci=c // 8, co=c // 8)  # 8 --> 4
        self.enc7 = blk.InvResBlock(co=c // 4, k=3,
                                    dropout=d,
                                    expand=False,
                                    squeeze=False)
        self.enc8 = ConvBnHs(ci=c // 4, co=c - ni_meta, k=4, s=1)  # 4 --> 1

        # decoder
        self.dec1 = ConvBnHs(ci=c, co=c // 4, k=4, s=1, t=True)  # 1 --> 4
        self.dec2 = blk.InvResBlock(ci=c // 4, co=c // 8, k=3,
                                    dropout=d,
                                    expand=False,
                                    squeeze=False)
        self.dec3 = ConvBnHs(ci=c // 8, co=c // 8, t=True)  # 4 --> 8
        self.dec4 = blk.InvResBlock(co=c // 16, k=3,
                                    dropout=d,
                                    expand=False,
                                    squeeze=False)
        self.dec5 = ConvBnHs(ci=c // 16, co=c // 16, t=True)  # 8 --> 16
        self.dec6 = blk.InvResBlock(co=c // 32, k=3,
                                    dropout=d,
                                    expand=False,
                                    squeeze=False)
        self.dec7 = nn.Upsample(size=(ri, ri), mode='bicubic')
        self.dec7 = ConvBnHs(ci=c // 32, co=c // 32, t=True)  # 16 --> 32
        self.dec8 = blk.InvResBlock(co=c // 32, k=3,
                                    dropout=d,
                                    expand=False,
                                    squeeze=False)
        self.dec9 = blk.Conv2d(co=1, k=3, p=1, bias=True)

    def forward(self, input_img, input_meta):
        # encoder
        x = input_img
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.enc6(x)
        x = self.enc7(x)
        x = self.enc8(x)

        # bottleneck
        n = x.shape[0]
        x = torch.cat([x, input_meta.reshape(n, -1, 1, 1)], dim=1)

        # decoder
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = self.dec6(x)
        x = self.dec7(x)
        x = self.dec8(x)
        x = self.dec9(x)

        return x

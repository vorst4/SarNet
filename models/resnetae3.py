import torch.nn as nn
from abc import ABC
import models.blocks as blk
import settings


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

        c = 512  # channels
        ri = settings.IMG_RESOLUTION  # resolution input
        ni_meta = 24  # number of meta-data input variables
        d = settings.dropout_rate

        self.enc1 = blk.InvResBlock(co=c // 32, k=3,
                                    downsample=False,
                                    dropout=d * (1 - 0.2 * 0),
                                    expand=False,
                                    squeeze=False)
        self.enc2 = nn.Conv2d(in_channels=c // 32,
                              out_channels=c // 32,
                              kernel_size=2,
                              stride=2,
                              bias=False)
        # self.enc3
        self.enc2 = blk.InvResBlock(co=c // 16, k=3,
                                    downsample=True,
                                    dropout=d * (1 - 0.2 * 1),
                                    expand=False,
                                    squeeze=False)
        self.enc3 = blk.InvResBlock(co=c // 8, k=3,
                                    downsample=True,
                                    dropout=d * (1 - 0.2 * 2),
                                    expand=False,
                                    squeeze=False)
        self.enc2 = blk.InvResBlock(co=c // 4, k=3,
                                    downsample=True,
                                    dropout=d * (1 - 0.2 * 3),
                                    expand=False,
                                    squeeze=False)
        self.enc2 = blk.InvResBlock(co=c // 2, k=3,
                                    downsample=True,
                                    dropout=d * (1 - 0.2 * 4),
                                    expand=False,
                                    squeeze=False)
        self.enc2 = blk.InvResBlock(co=c, k=3,
                                    downsample=True,
                                    dropout=d * (1 - 0.2 * 5),
                                    expand=False,
                                    squeeze=False)

        self.decoder = nn.Sequential(
            blk.Reshape(co=c, ro=1),
            blk.InvResBlock(co=c, k=2,
                            upsample=True,
                            dropout=d * (1 - 0.2 * 4),
                            expand=False,
                            squeeze=False),
            blk.InvResBlock(co=c // 2, k=2,
                            upsample=True,
                            dropout=d * (1 - 0.2 * 3),
                            expand=False,
                            squeeze=False),
            blk.InvResBlock(co=c // 4, k=2,
                            upsample=True,
                            dropout=d * (1 - 0.2 * 2),
                            expand=False,
                            squeeze=False),
            blk.InvResBlock(co=c // 8, k=3,
                            upsample=True,
                            upsample_mode='bicubic',
                            dropout=d * (1 - 0.2 * 1),
                            expand=False,
                            squeeze=False),
            blk.InvResBlock(co=c // 16, k=3,
                            upsample=True,
                            upsample_mode='bicubic',
                            dropout=d * (1 - 0.2 * 0),
                            expand=False,
                            squeeze=False),
            blk.Conv2d(co=1, k=3, p=1, bias=True),

        )

    def forward(self, input_img, input_meta):
        x = self.encoder(input_img)
        x = self.combine(x, input_meta)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

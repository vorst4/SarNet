import torch.nn as nn
from abc import ABC
import models.blocks as blk
import settings


class ResNetAE2(nn.Module, ABC):
    lr_ideal = 1e-5

    def __init__(self):
        super().__init__()

        c = 512  # channels
        ri = settings.IMG_RESOLUTION  # resolution input
        ni_meta = 24  # number of meta-data input variables
        d = settings.dropout_rate

        self.encoder = nn.Sequential(
            blk.IResBlock(co=c // 32, k=3,
                          downsample=True,
                          dropout=d * (1 - 0.2 * 0),
                          expand=False,
                          squeeze=False),
            blk.IResBlock(co=c // 16, k=3,
                          downsample=True,
                          dropout=d * (1 - 0.2 * 1),
                          expand=False,
                          squeeze=False),
            blk.IResBlock(co=c // 8, k=3,
                          downsample=True,
                          dropout=d * (1 - 0.2 * 2),
                          expand=False,
                          squeeze=False),
            blk.IResBlock(co=c // 4, k=3,
                          downsample=True,
                          dropout=d * (1 - 0.2 * 3),
                          expand=False,
                          squeeze=False),
            blk.IResBlock(co=c // 2, k=3,
                          downsample=True,
                          dropout=d * (1 - 0.2 * 4),
                          expand=False,
                          squeeze=False),
            blk.IResBlock(co=c, k=3,
                          downsample=True,
                          dropout=d * (1 - 0.2 * 5),
                          expand=False,
                          squeeze=False),
        )

        self.combine = blk.Combine()

        self.bottleneck = blk.LinBnHs(ci=c + ni_meta, co=c)

        self.decoder = nn.Sequential(
            blk.Reshape(co=c, ro=1),
            blk.IResBlock(co=c, k=2,
                          upsample=True,
                          dropout=d * (1 - 0.2 * 4),
                          expand=False,
                          squeeze=False),
            blk.IResBlock(co=c // 2, k=2,
                          upsample=True,
                          dropout=d * (1 - 0.2 * 3),
                          expand=False,
                          squeeze=False),
            blk.IResBlock(co=c // 4, k=2,
                          upsample=True,
                          dropout=d * (1 - 0.2 * 2),
                          expand=False,
                          squeeze=False),
            blk.IResBlock(co=c // 8, k=3,
                          upsample=True,
                          upsample_mode='bicubic',
                          dropout=d * (1 - 0.2 * 1),
                          expand=False,
                          squeeze=False),
            blk.IResBlock(co=c // 16, k=3,
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

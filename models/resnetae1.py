from abc import ABC

import torch.nn as nn

from .blocks import Reshape, IResBlock, Combine, LinBnHs


class ResNetAE1(nn.Module, ABC):
    lr_ideal = 1e-5

    def __init__(self):
        super().__init__()

        # encoder
        # self.encoder = nn.Sequential(
        #     # Use just a convolutional layer for the first layer
        #     IResBlock(ci=3, co=32, ri=64, k=3, expand=False, squeeze=False),
        #     IResBlock(ci=32, co=24, ri=64, k=3, downsample=True),
        #     IResBlock(ci=24, co=24, ri=32, k=3),
        #     IResBlock(ci=24, co=40, ri=32, k=5, downsample=True),
        #     IResBlock(ci=40, co=40, ri=16, k=5),
        #     IResBlock(ci=40, co=80, ri=16, k=3, downsample=True),
        #     IResBlock(ci=80, co=80, ri=8, k=3),
        #     IResBlock(ci=80, co=112, ri=8, k=5, downsample=True),
        #     IResBlock(ci=112, co=128, ri=4, k=5),
        # )
        self.encoder = nn.Sequential(
            # Use just a convolutional layer for the first layer
            IResBlock(ci=3, co=32, ri=64, k=3, expand=False, squeeze=False),
            IResBlock(ci=32, co=64, ri=64, k=3, downsample=True),
            IResBlock(ci=64, co=96, ri=32, k=5, downsample=True),
            IResBlock(ci=96, co=128, ri=16, k=3, downsample=True),
            IResBlock(ci=128, co=256, ri=8, k=5, downsample=True),
            IResBlock(ci=256, co=512, ri=4, k=3, downsample=True),
            IResBlock(ci=512, co=1024, ri=2, k=3, downsample=True),
        )

        # combination of encoder and meta-data input
        self.combine = Combine()

        # bottleneck
        ch = 1024
        self.bottleneck = nn.Sequential(
            LinBnHs(ci=1024 * 1 + 24, co=ch),
            LinBnHs(co=ch),
            LinBnHs(co=ch * 2)
        )

        # self.decoder = nn.Sequential(
        #     Reshape(channels=128, resolution=4),
        #     IResBlock(ci=128, co=112, ri=4, k=5, upsample=True),
        #     IResBlock(ci=112, co=80, ri=8, k=5),
        #     IResBlock(ci=80, co=80, ri=8, k=3, upsample=True),
        #     IResBlock(ci=80, co=40, ri=16, k=3),
        #     IResBlock(ci=40, co=40, ri=16, k=3, upsample=True),
        #     IResBlock(ci=40, co=24, ri=32, k=5),
        #     IResBlock(ci=24, co=32, ri=32, k=3, upsample=True),
        #     IResBlock(ci=32, co=1, ri=64, k=3)
        # )

        self.decoder = nn.Sequential(
            Reshape(channels=512, resolution=2),
            IResBlock(ci=512, co=256, ri=2, k=3, upsample=True),
            IResBlock(ci=256, co=128, ri=4, k=3, upsample=True),
            IResBlock(ci=128, co=64, ri=8, k=5, upsample=True),
            IResBlock(ci=64, co=32, ri=16, k=3, upsample=True),
            IResBlock(ci=32, co=16, ri=32, k=3, upsample=True),
            nn.Conv2d(16, 1, 3, padding=1)
        )

    def forward(self, input_img, input_meta):
        x = self.encoder(input_img)

        x = self.combine(x, input_meta)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x
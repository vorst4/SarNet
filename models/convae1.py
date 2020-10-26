from abc import ABC

import torch.nn as nn

from .blocks import ConvBnHs, Combine, LinBnHs, InvResBlock, Reshape, \
    InverseResidualEncoder
from util.timer import Timer
from util.log import Log


class ConvAE1(nn.Module, ABC):
    """
    Convolutional Auto-encoder 1
        bottleneck: 1 dense connected layer, batch norm and relu
        decoder: 5 times (transpose conv2d, batch norm, relu)
    """

    lr_ideal = 1e-3

    def __init__(self):
        super().__init__()

        channels = 1024

        # layers
        self.encoder = InverseResidualEncoder()

        # combination of encoder and meta-data input
        self.combine = Combine()

        # bottleneck
        ch = 1024
        self.bottleneck = nn.Sequential(
            LinBnHs(ci=1024 * 1 + 24, co=ch * 2),
        )

        self.decoder = nn.Sequential(
            Reshape(c=512, ro=2),
            ConvBnHs(ci=512, co=256, k=2, s=2, po=0, t=True),  # 2 -> 4
            ConvBnHs(co=256, k=2, s=2, po=0, t=True),  # -> 8
            ConvBnHs(co=128, k=4, s=2, po=0, t=True),  # -> 16
            ConvBnHs(co=64, k=4, s=2, po=0, t=True),  # -> 32
            ConvBnHs(co=32, k=2, s=2, po=0, t=True),  # -> 64
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, input_img, input_meta):
        timer = Timer(Log(None)).start()
        x = self.encoder(input_img)
        timer.stop('\t\tforwarded through encoder')
        timer.start()
        x = self.combine(x, input_meta)
        timer.stop('\t\tforwarded through combine')
        timer.start()
        x = self.bottleneck(x)
        timer.stop('\t\tforwarded through bottleneck')
        timer.start()
        x = self.decoder(x)
        timer.stop('\t\tforwarded through decoder')
        return x

import settings
import models.blocks as blk
import torch.nn as nn
import torch
from abc import ABC


class _SarNetL(nn.Module, ABC):

    def __init__(self, skip: bool):
        super().__init__()

        a = 2 if skip else 1
        c = 1024 * 2
        r = settings.IMG_RESOLUTION  # resolution input
        n_meta = 24  # number of meta-data input variables
        d = settings.dropout_rate

        # encoder
        self.enc1 = nn.Sequential(
            nn.Flatten(),
            blk.LinBnHsDo(ci=3 * r ** 2, co=c, dr=d)
        )
        self.enc2 = blk.LinBnHsDo(ci=c, co=c // 2, dr=d)
        self.enc3 = blk.LinBnHsDo(ci=c // 2, co=c // 4, dr=d)
        self.enc4 = blk.LinBnHsDo(ci=c // 4, co=c // 8, dr=d)
        self.enc5 = blk.LinBnHsDo(ci=c // 8, co=c // 16 - n_meta, dr=d)

        # decoder
        self.dec1 = blk.LinBnHsDo(ci=c // 16, co=c // 8, dr=d)
        self.dec2 = blk.LinBnHsDo(ci=c // 8 * a, co=c // 4, dr=d)
        self.dec3 = blk.LinBnHsDo(ci=c // 4 * a, co=c // 2, dr=d)
        self.dec4 = blk.LinBnHsDo(ci=c // 2 * a, co=c, dr=d)
        self.dec5 = nn.Sequential(
            blk.LinBnHs(ci=c * a, co=r ** 2),
            blk.Reshape(co=1, ro=r)
        )


class SarNetL(_SarNetL, ABC):
    lr_ideal = 1e-6

    def __init__(self):
        super().__init__(False)

    def forward(self, input_img, input_meta):
        # encoder
        x = self.enc1(input_img)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)

        # decoder
        x = self.dec1(torch.cat([x, input_meta], dim=1))
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)

        return x


class SarNetLS(_SarNetL, ABC):
    lr_ideal = 1e-5

    def __init__(self):
        super().__init__(True)

    def forward(self, input_img, input_meta):
        # encoder
        x1 = self.enc1(input_img)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x = self.enc5(x4)

        # decoder
        x = self.dec1(torch.cat([x, input_meta], dim=1))
        x = self.dec2(torch.cat([x, x4], dim=1))
        x = self.dec3(torch.cat([x, x3], dim=1))
        x = self.dec4(torch.cat([x, x2], dim=1))
        x = self.dec5(torch.cat([x, x1], dim=1))

        return x

import torch.nn as nn
from abc import ABC
from .blocks import Reshape


class InfoAE(nn.Module, ABC):
    """
    autoencoder where the decoder is based on the generator of StyleGAN (
    celebA model). However, the last activation  function (tanh) is replaced
    by sigmoid
    """

    lr_ideal = 1e-6

    def __init__(self):
        super().__init__()

        self.encoder = SimplifiedEncoder(

        )
        self.bottleneck = nn.Sequential(
            nn.Linear(2, 228),
            nn.BatchNorm1d(228),
            nn.ReLU())
        self.decoder = nn.Sequential(
            Reshape((-1, 228, 1, 1)),
            nn.ConvTranspose2d(228, 448, 2, 1, bias=False),
            nn.BatchNorm2d(448),
            nn.ReLU(),
            nn.ConvTranspose2d(448, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, padding=1, bias=False),
            nn.ReLU(),
            # upsample the 28x28px image to 32x32
            nn.Upsample(size=(32, 32), mode='bicubic', align_corners=False)
        )

    def forward(self, input_img, input_meta):
        x = self.encoder(input_img, input_meta)
        x = self.bottleneck(x)
        return self.decoder(x)

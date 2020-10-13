import torch.nn as nn
from abc import ABC
from .blocks import Reshape


class DenseAE2(nn.Module, ABC):
    """
    Auto-encoder consisting out of 5 densely connected layers, with batch normalisation and relu, and sigmoid as last
    layer
    """

    lr_ideal = 1e-6

    def __init__(self):
        super().__init__()

        features = 4096
        features_out = 1024

        # layers
        self.encoder = SimplifiedEncoder()
        self.decoder = nn.Sequential(
            nn.Linear(2, features//16),
            nn.BatchNorm1d(features//16),
            nn.ReLU(),
            nn.Linear(features//16, features//8),
            nn.BatchNorm1d(features//8),
            nn.ReLU(),
            nn.Linear(features//8, features//4),
            nn.BatchNorm1d(features//4),
            nn.ReLU(),
            nn.Linear(features//4, features//2),
            nn.BatchNorm1d(features//2),
            nn.ReLU(),
            nn.Linear(features//2, features_out),
            Reshape((-1, 1, 32, 32)),
            nn.Sigmoid())

    def forward(self, input_img, input_meta):
        x = self.encoder(input_img, input_meta)
        # <no bottleneck>
        return self.decoder(x)

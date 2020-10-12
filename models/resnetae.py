import torch.nn as nn
from abc import ABC
from .blocks import SimplifiedEncoder
from .blocks import TransposeResNetBlock
from .blocks import Reshape
from .blocks import IResBlock, Flatten, Combine


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
            IResBlock(ci=128, co=160, ri=8, k=5, downsample=True),
            IResBlock(ci=160, co=256, ri=4, k=5, downsample=True),
        )

        # bottleneck
        ch = 1024
        self.combine = Combine()
        self.bottleneck = nn.Sequential(
            nn.Linear(256 * 4 + 24, ch),
            nn.BatchNorm1d(ch),
            nn.Hardswish(),
            nn.Linear(ch, ch),
            nn.BatchNorm1d(ch),
            nn.Hardswish(),
            nn.Linear(ch, 256*4),
            nn.BatchNorm1d(256*4),
            nn.Hardswish(),
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
            Reshape(channels=256, resolution=2),
            IResBlock(ci=256, co=160, ri=2, k=5, upsample=True),
            IResBlock(ci=160, co=96, ri=4, k=3, upsample=True),
            IResBlock(ci=96, co=64, ri=8, k=5, upsample=True),
            IResBlock(ci=64, co=32, ri=16, k=3, upsample=True),
            IResBlock(ci=32, co=1, ri=32, k=3, upsample=True),
        )

    def forward(self, input_img, input_meta):
        x = self.encoder(input_img)

        x = self.combine(x, input_meta)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

#
#
#
#
# class _ResNetBlock(nn.Module):
#
#     def __init__(self, filter_in, filter_out, downsample=False, upsample=False):
#         """
#         create ResNet block, the up/down sampling is always a factor 2 and is done using strides in first convolutional
#         layer.
#
#         :param filter_in: number of input filters
#         :param filter_out: number of output filters
#         :param downsample: boolean, whether to upsample by a factor 2 or not
#         :param upsample: boolean, whether to downsample by a factor 2 or not
#         """
#         super().__init__()
#
#         # check input
#         if downsample and upsample:
#             raise Exception("downsample and upsample can't be true at the same time")
#
#         # define layers
#         if downsample:
#             self.conv1 = nn.Conv2d(filter_in, filter_out, kernel_size=3, stride=2, padding=1, bias=False)
#             self.residual = nn.Conv2d(filter_in, filter_out, kernel_size=1, stride=2, padding=0, bias=False)
#         elif upsample:
#             self.conv1 = nn.ConvTranspose2d(filter_in, filter_out, kernel_size=2, stride=2, padding=0, bias=False)
#             self.residual = nn.ConvTranspose2d(filter_in, filter_out, kernel_size=2, stride=2, padding=0, bias=False)
#         else:
#             self.conv1 = nn.Conv2d(filter_in, filter_out, kernel_size=3, stride=1, padding=1, bias=False)
#             self.residual = nn.Conv2d(filter_in, filter_out, kernel_size=1, stride=1, padding=0, bias=False)
#
#         self.bn = nn.BatchNorm2d(filter_out)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(filter_out, filter_out, kernel_size=3, stride=1, padding=1, bias=False)
#
#     def forward(self, input):
#         output = self.conv1(input)
#         output = self.bn(output)
#         output = self.relu(output)
#         output = self.conv2(output)
#         output = self.bn(output)
#         output += self.residual(input)
#         output = self.relu(output)
#         return output
#
#
# class _Encoder(nn.Module):
#
#     def __init__(self, filters, downsample):
#         super().__init__()
#
#         assert len(filters) == len(downsample) + 1, "len(filters) should be len(downsample)+1"
#
#         # concatenate resnet blocks
#         self.blocks = nn.ModuleList()
#         for idx in range(len(filters) - 1):
#             self.blocks.append(ResNetBlock(filters[idx], filters[idx + 1], downsample=downsample[idx]))
#
#         # determine output size of encoder
#         res = _resolution_input_image // 2 ** sum(downsample)
#         self.output_size = filters[-1] * res ** 2
#
#     def forward(self, x):
#         for block in self.blocks:
#             x = block(x)
#         return x
#
# class _Bottleneck(nn.Module):
#
#     def __init__(self, features):
#         super().__init__()
#
#         # concatenate dense (linear) layers
#         self.blocks = nn.ModuleList()
#         for idx in range(len(features) - 1):
#             self.blocks.append(nn.Linear(features[idx], features[idx + 1]))
#             self.blocks.append(nn.ReLU())
#
#     def forward(self, output_encoder, input_meta):
#
#         # combine output_encoder and input_meta
#         x = torch.cat((output_encoder.flatten(start_dim=1), input_meta), 1)
#
#         # apply dense layers
#         for layer in self.blocks:
#             x = layer(x)
#
#         return x
#
# class _Decoder(nn.Module):
#     UPSAMPLE = 1
#     DOWNSAMPLE = -1
#
#     def __init__(self, filters, upsample):
#         """
#         ResNet consists out of multiple ResNetBlocks. The number of blocks is equal to len(filters)-1. Each element
#         of the list <filters> is the input size of the ResNetBlock, except the last element which is the output size.
#
#         :param filters: integer list, each element -except the last- is the input size of a ResNetBlock,
#             the last element is the output size of the last ResNetBlock
#         """
#         super().__init__()
#
#         assert len(filters) == len(upsample) + 1, "len(filters) should be len(downsample)+1"
#
#         # concatenate resnet blocks
#         self.blocks = nn.ModuleList()
#         for idx in range(len(filters) - 1):
#             self.blocks.append(__ResNetBlock(filters[idx], filters[idx + 1], upsample=upsample[idx]))
#
#         # decoder input resolution
#         res = _resolution_output_image // 2 ** (sum(upsample))
#
#         # decoder input size
#         self.input_size = filters[0] * res ** 2
#
#         # decoder input shape
#         self.input_shape = [-1, filters[0], res, res]
#
#     def forward(self, x):
#         x = x.reshape(self.input_shape)
#         for block in self.blocks:
#             x = block(x)
#         return x

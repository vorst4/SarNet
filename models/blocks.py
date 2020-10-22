from abc import ABC

import torch
import torch.nn as nn

import settings
from util.timer import Timer
from util.log import Log

_last_channels_out = None
_last_resolution_out = None


class Conv2d(nn.Module, ABC):

    def __init__(self,
                 ci,  # channels input
                 co,  # channels output
                 k=3,  # kernel size
                 s=1,  # stride
                 p=0,  # padding
                 d=1,  # dilation
                 groups=1,
                 bias=False,
                 mode='zeros'
                 ):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=ci,
                                out_channels=co,
                                kernel_size=k,
                                stride=s,
                                padding=p,
                                dilation=d,
                                groups=g,
                                bias=bias,
                                padding_mode=mode)

    def forward(self, x):
        return self.conv2d(x)


class Combine(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x_encoder: torch.tensor, x_meta: torch.tensor):
        return torch.cat((x_encoder.reshape(x_encoder.shape[0], -1), x_meta),
                         dim=1)


class Reshape(nn.Module, ABC):

    def __init__(self, co, ro):
        super().__init__()
        self.shape = (-1, co, ro, ro)

        # set globals
        global _last_channels_out, _last_resolution_out
        _last_channels_out, _last_resolution_out = co, ro

    def forward(self, x):
        return x.reshape(self.shape)


class LinBnHs(nn.Module, ABC):
    def __init__(
            self,
            co: int,  # number of output channels
            ci: int = None,  # number of input channels
    ):
        super().__init__()
        global _last_channels_out
        ci = _last_channels_out if ci is None else ci
        _last_channels_out = co

        self.lin = nn.Linear(ci, co)
        self.bn = nn.BatchNorm1d(num_features=co, **settings.batch_norm)
        self.hs = nn.Hardswish()

    def forward(self, x):
        return self.hs(self.bn(self.lin(x)))


class ConvBnHs(nn.Module, ABC):
    """
    (Transpose) Convolutional filter + batch normalisation + hard swish
    """

    def __init__(
            self,
            ci: int = None,  # input channels
            co: int = None,  # output channels
            k: int = 3,  # kernel size
            s: int = 1,  # stride
            p: int = None,  # padding, if None -> padding is self determined
            po: int = 1,  # output_padding (only applies for transpose conv)
            t: bool = False  # transpose convolution or not
    ):
        super().__init__()
        global _last_channels_out

        assert co is not None, 'number of output channels must be provided'
        assert not (ci is None and _last_channels_out is None)

        # determine padding if needed
        p = int((k - 1) * 0.5) if p is None else p

        # determine input channels if needed
        ci = _last_channels_out if ci is None else ci

        # set last channels out
        _last_channels_out = co

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
        self.bn = nn.BatchNorm2d(num_features=co, **settings.batch_norm)
        self.hs = nn.Hardswish()

    def forward(self, x):
        return self.hs(self.bn(self.conv(x)))


# class RepIResBlock(nn.Module, ABC):
#     """
#     Repeats the IResBlock <n_rep> times, in series.
#
#     1. If downsample is true then only the first repetition is down-sampled,
#        the others are not.
#     2. The first block uses in_channel as the input channel, all other
#        blocks use out_channel as the output channel.
#     """
#
#     def __init__(self,
#                  n_rep: int,
#                  in_channels: int,
#                  out_channels: int,
#                  in_resolution: int,
#                  kernel_size: [3, 5, 7] = 3,
#                  downsample: bool = False,
#                  expand=True,
#                  squeeze_and_excite: bool = True):
#         super().__init__()
#         self.blocks = []
#         resolution = in_resolution // 2 if downsample else in_resolution
#         for idx in range(n_rep):
#             self.blocks.append(IResBlock(
#                 in_channels if idx == 0 else out_channels,
#                 out_channels,
#                 in_resolution if idx == 0 else resolution,
#                 kernel_size,
#                 downsample if idx == 0 else False,
#                 expand,
#                 squeeze_and_excite
#             ))
#
#     def forward(self, x):
#         for block in self.blocks:
#             x = block(x)
#         return x


class IResBlock(nn.Module, ABC):
    """
    Inverse Resnet Block from MobileNet v3, it is an inverted residual block
    with linear bottleneck + squeeze & excitation.

    See MobileNetV3 paper: https://arxiv.org/pdf/1905.02244.pdf

    NOTE: to downsample, use stride = 2

    Arguments:
        ci: number of input channels/filters
        co: number of output channels/filters
        ri: the input resolution -> r = width = height (must be square)
        k: kernel size, 3, 5, 7 are valid options
        downsample = True -> output resolution is ~0.5 times that of the input
        upsample = True -> output resolution is ~2 times that of the input
        expand: if the expansion step should be applied (expansion factor = 6)
        squeeze: if the squeeze and excite step should be applied
            (squeeze factor is 1/4)
    """

    SQUEEZE = 0.25

    EXPANSION = 6

    def __init__(self,
                 co: int,
                 ci: int = None,
                 ri: int = None,
                 k: [3, 5, 7] = 3,
                 downsample: bool = False,
                 upsample: bool = False,
                 expand=True,
                 squeeze: bool = True):

        # misc
        super().__init__()
        assert not (downsample & upsample), "cannot be True at the same time"
        global _last_channels_out, _last_resolution_out

        # If ci/ri not provided: use co/ro of the block that was created last
        ci = _last_channels_out if ci is None else ci
        ri = _last_resolution_out if ri is None else ri

        # determine stride
        s = 2 if downsample or upsample else 1

        # determine output resolution
        if downsample:
            ro = ri // 2
        elif upsample:
            ro = ri * 2
        else:
            ro = ri

        # set globals co and ro
        _last_channels_out = co
        _last_resolution_out = ro

        # determine padding
        p = int((k - 1) * 0.5)

        # determine type of convolutional filter to use
        conv = self.transpose_conv if upsample else nn.Conv2d

        # determine number of 'depth-wise' channels
        ch_depth = self.EXPANSION if expand else 1

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

        # expand narrow input to a wide layer
        if expand:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels=ci,
                          out_channels=ci * ch_depth,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(num_features=ci * ch_depth,
                               **settings.batch_norm),
                nn.Hardswish()
            )

        # depth-wise convolution
        self.depth_conv = nn.Sequential(
            conv(in_channels=ci * ch_depth,
                 out_channels=ci * ch_depth,
                 groups=ci * ch_depth,
                 kernel_size=k,
                 stride=s,
                 padding=p,
                 bias=False),
            nn.BatchNorm2d(num_features=ci * ch_depth, **settings.batch_norm),
            nn.Hardswish()
        )

        # squeeze & excite
        if squeeze:
            self.squeeze = nn.Sequential(
                # squeeze
                nn.AvgPool2d(kernel_size=ro),
                nn.Conv2d(in_channels=ci * ch_depth,
                          out_channels=int(ci * self.SQUEEZE),
                          kernel_size=1),
                nn.Hardswish(),
                nn.Conv2d(in_channels=int(ci * self.SQUEEZE),
                          out_channels=ci * ch_depth,
                          kernel_size=1)
            )
            self.excite = self._Excite()

        # project wide layer to narrow output
        self.project = nn.Sequential(
            nn.Conv2d(in_channels=ci * ch_depth,
                      out_channels=co,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(num_features=co, **settings.batch_norm),
            nn.Hardswish()
        )

        # upsample/downsample identity, if needed
        if upsample or downsample or ci != co:
            conv = nn.ConvTranspose2d if upsample else nn.Conv2d
            self.identity = conv(in_channels=ci,
                                 out_channels=co,
                                 kernel_size=2 if upsample else 1,
                                 stride=s,
                                 padding=0)

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

        # set attributes
        self.expand = expand
        self.upsample = upsample
        self.downsample = downsample
        self.co = co
        self.squeeze_and_excite = squeeze
        self.ci = ci
        self.stride = s
        self.padding = p

    @staticmethod
    def transpose_conv(**kwargs):
        p = 0 if (0.5 * kwargs['kernel_size']).is_integer() else 1
        return nn.ConvTranspose2d(**kwargs, output_padding=p)

    class _Excite:
        def __call__(self, x_squeezed, x):
            return torch.sigmoid(x_squeezed) * x

    def forward(self, x):
        # timer = Timer(Log(None))  # todo: remove
        # timer2 = Timer(Log(None)).start()  # todo: remove

        i = x  # identity

        # apply expansion
        # timer.start()
        if self.expand:
            x = self.expand_conv(x)
        # timer.stop('\t\t\tforwarded through expansion')

        # apply depth-wise convolution
        # timer.start()
        x = self.depth_conv(x)
        # timer.stop('\t\t\tforwarded through depth-wise conv')

        # squeeze & excite
        if self.squeeze_and_excite:
            # timer.start()
            x = self.excite(self.squeeze(x), x)
            # timer.stop('\t\t\tforwarded through squeeze & excite')

        # project the expanded convolution back to a narrow one
        # timer.start()
        x = self.project(x)
        # timer.stop('\t\t\tforwarded through projection')

        # add identity to x, upsample/downsample it if required
        # timer.start()
        if self.upsample or self.downsample or self.ci != self.co:
            # x = x + self.identity(i)
            # timer.stop('\t\t\tforwarded : added (changed) identity')
            # timer2.stop('\t\t\tforwarded: inverse resnet block')
            return x + self.identity(i)
        else:
            # x = x + i
            # timer2.stop('\t\t\tforwarded: inverse resnet block')
            # timer.stop('\t\t\tforwarded: added (unchanged) identity')
            return x + i


class TransposeResNetBlock(nn.Module, ABC):

    def __init__(self, filter_in, filter_out):
        super().__init__()
        self.identity = nn.Sequential(
            nn.ConvTranspose2d(filter_in, filter_out, kernel_size=2,
                               stride=2, padding=0, bias=False),
            nn.BatchNorm2d(filter_out))
        self.residual = nn.Sequential(
            nn.ConvTranspose2d(filter_in, filter_out, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filter_out),
            nn.ReLU(),
            nn.Conv2d(filter_out, filter_out, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(filter_out))
        self.relu = nn.ReLU()

    def forward(self, x):
        r = self.residual(x)
        i = self.identity(x)
        x = r + i
        x = self.relu(x)
        return x


class ResNetBlock(nn.Module, ABC):

    def __init__(self, ci, co):
        super().__init__()
        self.identity = nn.Sequential(
            Conv2d(ci=ci, co=co, k=2, s=2, p=0),
            nn.BatchNorm2d(co)
        )
        self.residual = nn.Sequential(
            Conv2d(ci=ci, co=co, k=4, s=2, p=1),
            nn.BatchNorm2d(co),
            nn.ReLU(),
            Conv2d(ci=co, co=co, k=3, s=1, p=1),
            nn.BatchNorm2d(co),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        r = self.residual(x)
        i = self.identity(x)
        x = r + i
        x = self.relu(x)
        return x


class InverseResidualEncoder(nn.Module, ABC):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # Use just a convolutional layer for the first layer
            IResBlock(ci=3, co=32, ri=64, k=3, expand=False, squeeze=False),
            IResBlock(ci=32, co=64, ri=64, k=3, expand=False, downsample=True),
            IResBlock(ci=64, co=96, ri=32, k=5, downsample=True),
            IResBlock(ci=96, co=128, ri=16, k=3, downsample=True),
            IResBlock(ci=128, co=256, ri=8, k=5, downsample=True),
            IResBlock(ci=256, co=512, ri=4, k=3, downsample=True),
            IResBlock(ci=512, co=1024, ri=2, k=3, downsample=True),
        )

    def forward(self, x):
        return self.encoder(x)

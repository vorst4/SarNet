from abc import ABC

import torch
import torch.nn as nn
from typing import Any, Union

import settings
from util.timer import Timer
from util.log import Log

_last_channels_out = 3  # permittivity, conductivity, density
_last_resolution_out = settings.IMG_RESOLUTION


class Forward(nn.Module, ABC):
    @staticmethod
    def forward(x):
        return x


class Upsample(nn.Upsample, ABC):
    def __init__(self):
        super().__init__(scale_factor=2, mode='bicubic', align_corners=False)


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
    """
    ci: channels input
    co: channels output
    """

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


class LinBnHsDo(LinBnHs, ABC):
    """
    dr : dropout rate
    """

    def __init__(self, co: int, ci: int, dr: float):
        super().__init__(co=co, ci=ci)
        self.dr = nn.Dropout(p=dr)

    def forward(self, x):
        return self.dr(self.hs(self.bn(self.lin(x))))


class ConvBnHs(nn.Module, ABC):
    """
    (Transpose) Convolutional filter + batch normalisation + hard swish

    ci: input channels
    co: output channels
    k: kernel size
    s: stride
    p: padding
    po: output padding (only for tranpose convolutions)
    t: to use transpose convolutions or not
    """

    def __init__(
            self,
            ci: int = None,
            co: int = None,
            k: int = 3,
            s: int = 1,
            p: int = None,
            po: int = 1,
            t: bool = False
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


class ConvBnHsDr(ConvBnHs, ABC):
    def __init__(
            self,
            ci: int,
            co: int,
            k: int,
            s: int = 1,
            p: int = 0,
            dr: float = settings.dropout_rate,
            po: int = 0,
            t: bool = False
    ):
        super().__init__(ci=ci, co=co, k=k, s=s, p=p, po=po, t=t)
        self.dr = nn.Dropout2d(p=dr)

    def forward(self, x):
        return self.dr(self.hs(self.bn(self.conv(x))))


class InvResBlock(nn.Module, ABC):
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
                 upsample_mode: str = 'transpose',
                 dropout: float = settings.dropout_rate,
                 expand=True,
                 squeeze: bool = True):

        # misc
        super().__init__()
        assert not (downsample & upsample), "cannot be True at the same time"
        assert (upsample_mode == 'transpose' or upsample_mode == 'bicubic'), \
            'upsample_mode must either be "transpose" or "bicubic"'
        global _last_channels_out, _last_resolution_out

        # If ci/ri not provided: use co/ro of the block that was created last
        ci = _last_channels_out if ci is None else ci
        ri = _last_resolution_out if ri is None else ri

        # set transpose/bicubic
        transpose = True if upsample_mode is 'transpose' else False
        bicubic = True if upsample_mode is 'bicubic' else False

        # determine stride
        s = 2 if downsample or (upsample and transpose) else 1

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

        # determine type of convolutional filter to use during depth-wise conv
        conv = self.transpose_conv if (upsample and transpose) else nn.Conv2d

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

        # upsample (if bicubic)
        bicubic_upsample = Upsample() if bicubic else Forward()

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

        # dropout (adds stochastic depth)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

        # upsample/downsample identity, if needed
        if upsample or downsample or ci != co:
            conv = nn.ConvTranspose2d if upsample and transpose else nn.Conv2d
            self.identity = nn.Sequential(
                conv(in_channels=ci,
                     out_channels=co,
                     kernel_size=2 if upsample and transpose else 1,
                     stride=s,
                     padding=0),
                Upsample() if bicubic else Forward()
            )
        else:
            self.identity = Forward()

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

        # ATTRIBUTES
        self.expand = expand
        self.bicubic = bicubic
        self.bicubic_upsample = bicubic_upsample
        self.downsample = downsample
        self.co = co
        self.squeeze_and_excite = squeeze
        self.ci = ci
        self.stride = s
        self.padding = p
        self.kernel_size = k

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

        # bicubic upsample
        # timer.start()
        if self.bicubic:
            x = self.bicubic_upsample(x)
        # timer.stop('\t\t\tforwarded through bicubic upsample')

        # squeeze & excite
        if self.squeeze_and_excite:
            # timer.start()
            x = self.excite(self.squeeze(x), x)
            # timer.stop('\t\t\tforwarded through squeeze & excite')

        # project the expanded convolution back to a narrow one
        # timer.start()
        x = self.project(x)
        # timer.stop('\t\t\tforwarded through projection')

        # dropout
        # timer.start()
        if self.dropout is not None:
            x = self.dropout(x)
        # timer.stop('\t\t\tForwarded through dropout')

        # add identity to x, upsample/downsample it if required
        # timer.start()
        # x = x + self.identity(i)
        # timer.stop('\t\t\tforwarded : added (changed) identity')
        # timer2.stop('\t\t\tforwarded: inverse resnet block')
        # print(x)
        return x + self.identity(i)

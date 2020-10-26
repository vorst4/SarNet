"""
Every architecture/model that exists should be imported in this file,
because 'util.design' will look here for all the existing models.
"""
from .convae1 import ConvAE1
from .convae2 import ConvAE2
from .denseae1 import DenseAE1
from .resnetae1 import ResNetAE1
from .resnetae2 import ResNetAE2
from .convunet import ConvUNet
from .resunet import ResUNet
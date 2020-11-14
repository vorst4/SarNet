"""
Every architecture/model that exists should be imported in this file,
because 'util.design' will look here for all the existing models.
"""
from .sarnet_l import SarNetL
from .sarnet_c import SarNetC
from .sarnet_r import SarNetRN, SarNetRS
from .sarnet_m import SarNetM
from .sarnet_rv import SarNetRV, SarNetRV2, SarNetRV3

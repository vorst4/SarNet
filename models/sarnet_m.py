from .sarnet_r import SarNetRS, SarNetRN
from abc import ABC


class SarNetMS(SarNetRS, ABC):
    lr_ideal = 1e-5

    def __init__(self):
        super().__init__(sq=True, ex=True)


class SarNetM(SarNetRN, ABC):
    lr_ideal = 1e-5

    def __init__(self):
        super().__init__(sq=True, ex=True)

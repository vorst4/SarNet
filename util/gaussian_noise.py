import numpy as np


class GaussianNoise:
    """
    Data augmentation method that adds Gaussian noise
    """
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + self.std * np.random.randn(tensor.size()) + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={}, std={})'.format(self.mean, self.std)

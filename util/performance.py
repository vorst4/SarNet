from typing import Tuple

import numpy as np
import settings


class Performance:

    def __init__(self, n_validation_samples):
        self.n_validation_samples = n_validation_samples
        self.mse_train = _MseTrain()
        self.mse_valid = _MseValid()
        self.loss_valid = _MseSampleValid(n_validation_samples)

    def to_dict(self):
        return {
            'mse_train': self.mse_train.to_dict(),
            'mse_valid': self.mse_valid.to_dict(),
            'loss_valid': self.loss_valid.to_dict(),
            'n_validation_samples': self.n_validation_samples,
        }

    @classmethod
    def load(cls, dictionary):
        obj = cls(dictionary['n_validation_samples'])
        obj.mse_train = _MseTrain.load(dictionary['mse_train'])
        obj.mse_valid = _MseValid.load(dictionary['mse_valid'])
        return obj


class _List:

    def __init__(self, x_label: str, y_label: str, size: int):
        self.x_label = x_label
        self.y_label = y_label
        self.size = size
        self.x = [-1.0] * size
        self.y = [-1.0] * size
        self.idx = 0

    @staticmethod
    def load(cls, dictionary: dict):
        obj = cls()
        obj.x = dictionary['x']
        obj.y = dictionary['y']
        obj.idx = dictionary['idx']
        return obj

    def to_dict(self):
        return {
            'x_label': self.x_label,
            'y_label': self.y_label,
            'x': self.x,
            'y': self.y,
            'size': self.size,
            'idx': self.idx,
        }

    def append(self, x, y):
        self.x[self.idx] = float(x)
        self.y[self.idx] = float(y)
        self.idx += 1

    def allocate(self, n):
        """
        Allocate 'n' empty elements in the x and y array.
        """

        # current number of elements
        n0 = len(self.x)

        # return if empty elements to be allocated is less than current
        # number of empty elements
        if n <= n0:
            return

        # allocate empty elements
        self.x.extend([-1] * (n - n0))
        self.y.extend([-1] * (n - n0))

    def __call__(self, idx=None):
        if idx is None:
            return self.x[:self.idx], self.y[:self.idx]
        else:
            idx = self.idx + idx if idx < 0 else idx
            if idx >= self.idx or idx < 0:
                raise ValueError('index error')
            return self.x[idx], self.y[idx]

    def __len__(self):
        return self.idx

    def __str__(self):
        return str(self.to_dict())


class _DoubleList:

    def __init__(self,
                 x_label: str,
                 y_label: str,
                 z_label: str,
                 shape: Tuple[int, int]):
        self.x_label = x_label
        self.y_label = y_label
        self.z_label = z_label
        self.shape = shape
        self.x = [0.0] * shape[1] * shape[0]
        self.y = [0.0] * shape[1] * shape[0]
        self.z = [0.0] * shape[1] * shape[0]
        self.idx = 0

    @staticmethod
    def load(cls, dictionary: dict):
        obj = cls(dictionary['x_label'],
                  dictionary['y_label'],
                  dictionary['z_label'],
                  dictionary['size'])
        obj.x = dictionary['x']
        obj.y = dictionary['y']
        obj.z = dictionary['z']
        obj.idx = dictionary['idx']
        return obj

    def to_dict(self):
        return {
            'x_label': self.x_label,
            'y_label': self.y_label,
            'z_label': self.z_label,
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'size': self.shape,
            'idx': self.idx,
        }

    def append(self, x, y, z):
        if not isinstance(y, list):
            self.x[self.idx] = float(x)
            self.y[self.idx] = float(y)
            self.z[self.idx] = float(z)
            self.idx += 1
        else:
            ids = slice(self.idx, self.idx + len(y), 1)
            self.x[ids] = [x] * len(y)
            self.y[ids] = np.array(y, dtype=float).tolist()
            self.z[ids] = np.array(z, dtype=float).tolist()
            self.idx += int(len(y))

    def allocate(self, n):
        """
        Allocate 'n' empty elements in the x, y, z arrays.
        """

        # current number of elements
        n0 = len(self.x)

        # return if elements to be allocated is less than current
        # number of empty elements
        if n <= n0:
            return

        # allocate empty elements
        self.x.extend([-1] * (n - n0))
        self.y.extend([-1] * (n - n0))
        self.z.extend([-1] * (n - n0))

    def __call__(self):
        return self.x[:self.idx], self.y[:self.idx], self.z[:self.idx]

    def __len__(self):
        return self.idx

    def __str__(self):
        return str(self.to_dict())


class _MseTrain(_List):

    def __init__(self):
        super().__init__('epochs',
                         'MSE_train',
                         settings.epochs * settings.n_subsets)

    def append(self, epoch, mse_train):
        super().append(epoch, mse_train)

    @classmethod
    def load(cls, dictionary):
        return super().load(cls, dictionary)


class _MseValid(_List):
    def __init__(self):
        super().__init__('epochs',
                         'MSE_valid',
                         settings.epochs * settings.n_subsets)

    def append(self, epoch, mse_valid):
        super().append(epoch, mse_valid)

    @classmethod
    def load(cls, dictionary):
        return super().load(cls, dictionary)


class _MseSampleValid(_DoubleList):
    """
    MSE of each individual sample from the validation set, per epoch.
    """
    def __init__(self, n_validation_samples):
        super().__init__(x_label='epoch',
                         y_label='mse',
                         z_label='sample_idx',
                         shape=(settings.epochs * settings.n_subsets,
                                n_validation_samples),
                         )
        self.n_validation_samples = n_validation_samples

    def append(self, epoch, mse_sample, sample_idx):
        super().append(epoch, mse_sample, sample_idx)

    @classmethod
    def load(cls, dictionary):
        return super().load(cls, dictionary)

    def get_shape(self) -> dict:
        return {
            'len_epochs': self.shape[0],
            'n_validation_samples': self.shape[1],
        }

    def allocate(self, len_epochs):
        super().allocate(len_epochs * self.n_validation_samples)

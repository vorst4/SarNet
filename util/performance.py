from typing import Tuple

import numpy as np
import settings


class Performance:

    def __init__(self, n_validation_samples):
        self.n_validation_samples = n_validation_samples
        self.loss_train = _LossTrain()
        self.loss_valid = _LossValid()
        self.mse_train = _MseTrain()
        self.mse_valid = _MseValid()
        self.mape5_train = _Mape5Train()
        self.mape5_valid = _Mape5Valid()
        self.mape5i = _Mape5i(n_validation_samples)

    @classmethod
    def load(cls, dictionary: dict):
        obj = cls(dictionary['n_validation_samples'])
        obj.loss_train = _LossTrain.load(dictionary['loss_train'])
        obj.loss_valid = _LossValid.load(dictionary['loss_valid'])
        obj.mse_train = _MseTrain.load(dictionary['mse_train'])
        obj.mse_valid = _MseValid.load(dictionary['mse_valid'])
        obj.mape5_train = _Mape5Train.load(dictionary['mape5_train'])
        obj.mape5_valid = _Mape5Valid.load(dictionary['mape5_valid'])
        obj.mape5i = _Mape5i.load(dictionary['mape5i'])
        return obj

    def to_dict(self):
        """
        create dictionary containing all attributes
        """
        dictionary = {}
        for a in dir(self):
            if a[0] is not '_':
                a_type = str(type(getattr(self, a)))
                if 'util.performance' in a_type:
                    dictionary[a] = getattr(self, a).to_dict()
                elif 'int' in a_type or 'float' in a_type:
                    dictionary[a] = getattr(self, a)
        return dictionary


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
        # todo: allocating x,y,z (below) will result in an out of memory
        #  error on the server with var _MseSampleValid. this needs to be
        #  fixed. Maybe: save the data each epoch instead of keeping it in
        #  memory
        #  maybe it's fixed now?
        self.x = [0.0] * shape[1] * shape[0]
        self.y = [0.0] * shape[1] * shape[0]
        self.z = [0.0] * shape[1] * shape[0]
        self.idx = 0

    @staticmethod
    def load(cls, dictionary: dict):
        obj = cls(dictionary['size'][1] / 0.05)
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


class _LossTrain(_List):

    def __init__(self):
        super().__init__('epochs',
                         'Loss_train',
                         settings.epochs * settings.dataset.n_subsets)

    def append(self, epoch, loss_train):
        super().append(epoch, loss_train)

    @classmethod
    def load(cls, dictionary):
        return super().load(cls, dictionary)


class _LossValid(_List):
    def __init__(self):
        super().__init__('epochs',
                         'Loss_valid',
                         settings.epochs * settings.dataset.n_subsets)

    def append(self, epoch, loss_valid):
        super().append(epoch, loss_valid)

    @classmethod
    def load(cls, dictionary):
        return super().load(cls, dictionary)


class _MseTrain(_List):

    def __init__(self):
        super().__init__('epochs', 'MSE_train',
                         settings.epochs * settings.dataset.n_subsets)

    def append(self, epoch, mse_train):
        super().append(epoch, mse_train)

    @classmethod
    def load(cls, dictionary):
        return super().load(cls, dictionary)


class _MseValid(_List):

    def __init__(self):
        super().__init__('epochs', 'MSE_valid',
                         settings.epochs * settings.dataset.n_subsets)

    def append(self, epoch, mse_valid):
        super().append(epoch, mse_valid)

    @classmethod
    def load(cls, dictionary):
        return super().load(cls, dictionary)


class _Mape5Train(_List):

    def __init__(self):
        super().__init__('epochs', 'MAPE5_train',
                         settings.epochs * settings.dataset.n_subsets)

    def append(self, epoch, mape5_train):
        super().append(epoch, mape5_train)

    @classmethod
    def load(cls, dictionary):
        return super().load(cls, dictionary)


class _Mape5Valid(_List):

    def __init__(self):
        super().__init__('epochs', 'MAPE5_valid',
                         settings.epochs * settings.dataset.n_subsets)

    def append(self, epoch, mape5_valid):
        super().append(epoch, mape5_valid)

    @classmethod
    def load(cls, dictionary):
        return super().load(cls, dictionary)


class _Mape5i(_DoubleList):
    """
    MSE of each individual sample from the validation set, per epoch.
    """

    def __init__(self, n_validation_samples):
        n_mape5_samples = int(0.05 * n_validation_samples)
        super().__init__(x_label='epoch',
                         y_label='MAPE5i',
                         z_label='sample_idx',
                         shape=(settings.epochs, n_mape5_samples),
                         )
        self.n_mape5_samples = n_mape5_samples

    def append(self, epoch, mape5i, sample_idx):
        super().append(epoch, mape5i, sample_idx)

    @classmethod
    def load(cls, dictionary):
        return super().load(cls, dictionary)

    def get_shape(self) -> dict:
        return {
            'len_epochs': self.shape[0],
            'n_mape5_samples': self.shape[1],
        }

    def allocate(self, len_epochs):
        super().allocate(len_epochs * self.n_mape5_samples)

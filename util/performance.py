from typing import Tuple

import numpy as np


class PerformanceParameter2D:

    def __init__(self, x_label: str, y_label: str, size: int):
        self.x_label = x_label
        self.y_label = y_label
        self.size = size
        self.x = [-1.0] * size
        self.y = [-1.0] * size
        self.idx = 0

    @classmethod
    def load(cls, dictionary: dict):
        obj = cls(dictionary['x_label'], dictionary['y_label'],
                  dictionary['size'])
        obj.x = dictionary['x']
        obj.y = dictionary['y']
        obj.idx = dictionary['idx']
        return obj

    def to_dict(self):
        return {'x_label': self.x_label,
                'y_label': self.y_label,
                'x': self.x,
                'y': self.y,
                'size': self.size,
                'idx': self.idx}

    def append(self, x, y):
        self.x[self.idx] = float(x)
        self.y[self.idx] = float(y)
        self.idx += 1

    def allocate(self, n):
        """
        Allocate 'n' empty elements in the x and y array.
        """

        # current number of empty elements
        n0 = len(self)

        # return if empty elements to be allocated is less than current number of empty elements
        if n <= n0:
            return

        # allocate empty elements
        self.x.extend([-1] * (n - n0))
        self.y.extend([-1] * (n - n0))

    def __call__(self):
        return self.x[:self.idx], self.y[:self.idx]

    def __len__(self):
        return self.idx

    def __str__(self):
        return str(self.to_dict())


class PerformanceParameter3D:

    def __init__(self, x_label: str, y_label: str, z_label: str,
                 sizes: Tuple[int, int]):
        self.x_label = x_label
        self.y_label = y_label
        self.z_label = z_label
        self.sizes = sizes
        self.x = [0.0] * sizes[0] * sizes[1]
        self.y = [0.0] * sizes[0] * sizes[1]
        self.z = [0.0] * sizes[0] * sizes[1]
        self.idx = 0

    @classmethod
    def load(cls, dictionary: dict):
        obj = cls(dictionary['x_label'], dictionary['y_label'],
                  dictionary['z_label'], dictionary['size'])
        obj.x = dictionary['x']
        obj.y = dictionary['y']
        obj.z = dictionary['z']
        obj.idx = dictionary['idx']
        return obj

    def to_dict(self):
        return {'x_label': self.x_label,
                'y_label': self.y_label,
                'z_label': self.z_label,
                'x': self.x,
                'y': self.y,
                'z': self.z,
                'size': self.sizes,
                'idx': self.idx}

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

        # current number of empty elements
        n0 = len(self)

        # return if empty elements to be allocated is less than current number of empty elements
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

"""
This module contains all classes that are relevant to the data. Such as
dataset, transformations of the dataset, subsets, etc.

NOTE: each sample is returned as a dictionary, not a class. This is because
dictionaries are considerably faster in python, which has a major impact
since there are ~10 million samples
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas
import torch
import torch.utils.data
import torchvision.transforms
import torchvision.transforms.functional as func
from PIL import Image

N_ANTENNAS = 12

_img_to_tensor = torchvision.transforms.ToTensor()


class IDX:
    INDEX = 0
    INPUT_IMG = slice(1, 4)  # [1, 2, 3]
    INPUT_META = slice(4, 4 + 2 * N_ANTENNAS)  # [4, 5, ..., 27, 28]
    OUTPUT = -1  # 26


class KEY:
    INDEX = 'index'
    INPUT_IMG = 'input_img'
    INPUT_META = 'input_meta'
    OUTPUT = 'output'


def _list(slice_: slice) -> []:
    return list(range(*slice_.indices(slice_.stop)))


def _load_sample(data):
    return {
        KEY.INPUT_IMG: [Image.open(path) for path in data[IDX.INPUT_IMG]],
        KEY.INPUT_META: torch.tensor(data[IDX.INPUT_META]),
        KEY.OUTPUT: Image.open(data[IDX.OUTPUT]),
        KEY.INDEX: data[IDX.INDEX]
    }


class Dataset:
    """
    This dataset is implemented according to the map-style dataset of PyTorch
      https://pytorch.org/docs/stable/data.html#map-style-datasets

    The dataset.csv file should look as follows, but without spaces. Output_img
    and input_img are the relative paths to the images
      _________________________________________________________________________
      │      output_img      ;      input_img      ;  amplitude_1 ; ...
      │ "output/0000001.png" ; "input/0000001.png" ;       1      ; ...
      │ "output/0000002.png" ; "input/0000002.png" ;       1      ; ...
      │        ...           ;        ...          ;      ...     ; ...

    NOTE: if a different delimiter is used for the csv files, it can be
    passed as  an argument when creating instance of this object. However,
    all csv files must have the same delimiter

    The directory structure that corresponds to the example above is as follows
      dir
        ├─ input
        │  ├─ 0000001.png
        │  ├─ 0000002.png
        │  └─ ...
        ├─ output
        │  ├─ 0000001.png
        │  ├─ 0000003.png
        │  └─ ...
        └─ dataset.csv
    """

    KEY = KEY
    IDX = IDX

    ANGLES = tuple(torch.arange(0, 360, 30))

    def __init__(self,
                 dataset_file: Union[str, Path],
                 train_pct: float,
                 n_subsets: int = 1,
                 csv_delimiter: str = ';',
                 n_max: int = None,
                 trans_train=None,
                 trans_val=None,
                 shuffle_train=True,
                 shuffle_valid=False,
                 ):
        """
        Create instance of custom map-style PyTorch Dataset

        :param dataset_file: <str> or <pathlib.Path> of the dataset csv file
        :param csv_delimiter: (default: ';') delimiter used in the dataset
        csv file
        """

        # set attributes
        self.train_pct = train_pct
        self.n_subsets = n_subsets
        self.shuffle_train = shuffle_train
        self.shuffle_valid = shuffle_valid

        # if dir is a <str>, create a pathlib.Path from it
        if isinstance(dataset_file, str):
            dataset_file = Path().cwd().joinpath(dataset_file)

        # read dataset.csv into a nested list
        self.dataset = pandas.read_csv(
            dataset_file,
            delimiter=csv_delimiter,
            nrows=n_max,
        ).values.tolist()

        # set dataset length
        self.len = len(self.dataset)

        # set transformation functions, images must always be converted to
        #   tensor
        if trans_train is None:
            trans_train = ToTensor()
        if trans_val is None:
            trans_val = ToTensor()

        # get directory in which dataset_file is located
        directory = dataset_file.parents[0]

        # change relative path in dataset to full path
        for idx1 in range(self.len):
            for idx2 in (*_list(IDX.INPUT_IMG), IDX.OUTPUT):
                self.dataset[idx1][idx2] = str(
                    directory.joinpath(self.dataset[idx1][idx2])
                )

        # determine ids for validation/training dataset
        self.ids_train = np.arange(0, int(len(self) * self.train_pct / 100))
        self.ids_valid = np.arange(int(len(self) * self.train_pct / 100),
                                   len(self))

        # determine ids of training/validation subsets, and shuffle them
        self.ids_train_subset = np.array_split(self.ids_train, self.n_subsets)
        self.ids_valid_subset = np.array_split(self.ids_valid, self.n_subsets)
        self.shuffle()

        # create validation/training datasets
        self.training = []
        self.validation = []
        for ids_t, ids_v in zip(self.ids_train_subset, self.ids_valid_subset):
            self.training.append(_Subset(self, ids_t, trans_train))
            self.validation.append(_Subset(self, ids_v, trans_val))

    def __len__(self):
        return self.len

    def shuffle(self):
        """
        This function shuffles the WHOLE training/validation dataset,
        NOT each individual training/validation subset.

        NOTE: the training/validation dataset are still kept separate.
        """
        if self.shuffle_train:
            np.random.shuffle(self.ids_train)
        if self.shuffle_valid:
            np.random.shuffle(self.ids_valid)


class _Subset(torch.utils.data.Dataset):
    """
    Subset of dataset, that uses transformations
    """

    KEY = KEY
    IDX = IDX

    def __init__(self, ds_obj, indices, transform):
        self.ds_obj = ds_obj
        self.dataset = ds_obj.dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(_load_sample(self.dataset[self.indices[idx]]))

    def __len__(self):
        return len(self.indices)


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, sample):
        for idx, img in enumerate(sample[KEY.INPUT_IMG]):
            sample[KEY.INPUT_IMG][idx] = func.to_tensor(img)
        sample[KEY.INPUT_IMG] = torch.cat(sample[KEY.INPUT_IMG], dim=0)
        sample[KEY.OUTPUT] = func.to_tensor(sample[KEY.OUTPUT])
        return sample


class RandomRotation:
    """"
    Applies a random discrete rotation to the sample
    """

    def __init__(self,
                 resample=False,
                 expand=False,
                 center=None,
                 fill=None):
        self.degrees = tuple(np.arange(0, 360, 30))
        self.kwargs = {
            'resample': resample,
            'expand': expand,
            'center': center,
            'fill': fill
        }

    def random_degree(self):
        return self.degrees[int(torch.randint(len(self.degrees), [1]))]

    def __call__(self, sample):
        # apply random rotation to sample and return it
        idx = int(torch.randint(len(self.degrees), [1]))
        rand_degree = self.degrees[idx]

        # input images
        for idx, img in enumerate(sample[KEY.INPUT_IMG]):
            sample[KEY.INPUT_IMG][idx] = func.rotate(
                img, self.degrees[idx], **self.kwargs
            )

        # meta-data
        sample[KEY.INPUT_META] = torch.roll(sample[KEY.INPUT_META], 2 * idx)

        # output img
        sample[KEY.OUTPUT] = func.rotate(
            sample[KEY.OUTPUT], rand_degree, **self.kwargs
        )

        return sample


class RandomVerticalFlip:
    """
    Randomly applies a horizontal flip to the sample (50% chance).

    NOTE: MIRROR AXIS IS VERTICAL AXIS, SO LEFT TO RIGHT
    """

    def __call__(self, sample):
        # don't flip sample if below 50%
        if torch.rand(1) < 0.5:
            return sample

        # else, flip sample and return it
        #   flip input images
        for idx, img in enumerate(sample[KEY.INPUT_IMG]):
            sample[KEY.INPUT_IMG][idx] = func.vflip(img)

        #   flip input meta-data
        # todo: do this dynamically in a more elegant way
        sample[KEY.INPUT_META] = torch.tensor([
            #       Amplitude                     phase
            sample[KEY.INPUT_META][0], sample[KEY.INPUT_META][1],
            sample[KEY.INPUT_META][22], sample[KEY.INPUT_META][23],
            sample[KEY.INPUT_META][20], sample[KEY.INPUT_META][21],
            sample[KEY.INPUT_META][18], sample[KEY.INPUT_META][19],
            sample[KEY.INPUT_META][16], sample[KEY.INPUT_META][17],
            sample[KEY.INPUT_META][14], sample[KEY.INPUT_META][15],
            sample[KEY.INPUT_META][12], sample[KEY.INPUT_META][13],
            sample[KEY.INPUT_META][10], sample[KEY.INPUT_META][11],
            sample[KEY.INPUT_META][8], sample[KEY.INPUT_META][9],
            sample[KEY.INPUT_META][6], sample[KEY.INPUT_META][7],
            sample[KEY.INPUT_META][4], sample[KEY.INPUT_META][5],
            sample[KEY.INPUT_META][2], sample[KEY.INPUT_META][3],
        ])

        #   flip output img
        sample[KEY.OUTPUT] = func.vflip(sample[KEY.OUTPUT])

        return sample


class Normalize:
    """
    Normalize sample
    """

    def __init__(self,
                 mean_input_img=[0.5, 0.5, 0.5],
                 std_input_img=[0.5, 0.5, 0.5],
                 mean_input_meta=0.5,
                 std_input_meta=0.5,
                 mean_output=0.5,
                 std_output=0.5):
        self.mean_input_img = mean_input_img
        self.std_input_img = std_input_img
        self.mean_input_meta = mean_input_meta
        self.std_input_meta = std_input_meta
        self.mean_output = mean_output
        self.std_output = std_output

    def __call__(self, sample):
        func.normalize(tensor=sample[KEY.INPUT_IMG],
                       mean=self.mean_input_img,
                       std=self.std_input_img,
                       inplace=True)
        sample[KEY.INPUT_META] = \
            (sample[KEY.INPUT_META] - self.mean_input_meta) / \
            self.std_input_meta
        func.normalize(tensor=sample[KEY.OUTPUT],
                       mean=self.mean_output,
                       std=self.std_output,
                       inplace=True)
        return sample

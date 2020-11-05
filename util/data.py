"""
This module contains all classes that are relevant to the data. Such as
dataset, transformations of the dataset, subsets, etc.

NOTE: each sample is returned as a dictionary, not a class. This is because
dictionaries are considerably faster in python, which has a major impact
since there are ~10 million samples
"""

from pathlib import Path
from typing import Union, Tuple, List

import numpy as np
import pandas
import torch
import torch.utils.data
import torchvision.transforms
import torchvision.transforms.functional as func
from PIL import Image
from zipfile import ZipFile
from util.timer import Timer
from util.log import Log
from util.base_obj import BaseObj
from torch.utils.data import DataLoader
from dataclasses import dataclass

N_ANTENNAS = 12  # todo: don't hardcode this


class IDX:
    INDEX = 0
    INPUT_IMGS = slice(1, 4)  # [1, 2, 3]
    INPUT_META = slice(4, 4 + 2 * N_ANTENNAS)  # [4, 5, ..., 27, 28]
    OUTPUT = -1  # 26


class KEY:
    INDEX = 'index'
    INPUT_IMGS = 'input_img'
    INPUT_META = 'input_meta'
    OUTPUT = 'output'


def _list(slice_: slice) -> []:
    return list(range(*slice_.indices(slice_.stop)))


def _append_sample(dataset, idx, zipfile, index, input_imgs, input_meta,
                   output):
    # add sample-index
    dataset[KEY.INDEX][idx] = int(index)

    # add input img
    for idx_img, img in enumerate(input_imgs):
        dataset[KEY.INPUT_IMGS][idx, idx_img, :, :] = 255 * func.to_tensor(
            Image.open(zipfile.open(img))
        )

    # add input meta
    dataset[KEY.INPUT_META][idx, :] = torch.tensor(
        # convert to uint8 range
        [255 * float(a) for a in input_meta]
    )

    # add output img
    dataset[KEY.OUTPUT][idx, :, :, :] = 255 * func.to_tensor(
        Image.open(zipfile.open(output))
    )


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

    class Settings(BaseObj):
        def __init__(self,
                     file: Union[str, Path],
                     img_resolution: int = 32,
                     max_samples: int = None,
                     train_pct: float = 90,
                     n_subsets: int = 1,
                     batch_size: int = 32,
                     shuffle_train=True,
                     shuffle_valid=False,
                     csv_delimiter: str = ';',
                     dtype=torch.float32,
                     ):
            """
            :param file: pathlib.Path or str of the dataset zip file
            :param max_samples: maximum number of samples from dataset that
                are to be loaded and used (convenient for debugging/testing).
                If None is passed, the number of samples in the dataset are
                counted, note however that this is pretty slow.
            :param train_pct: percentage of dataset to be used for training
            :param n_subsets: running 1 epoch with the complete dataset can
                take a very long time, meaning it takes a long time before
                meaning-full results are obtained. The dataset can therefore
                be split into subsets such that results are obtained every
                < 1/n_subsets > epochs
            :param batch_size: batch size of each dataloader
            :param shuffle_train: whether to shuffle the training set or
                not, call method 'shuffle()' the shuffle the set
            :param shuffle_valid: whether to shuffle the validation set or
                not, call method 'shuffle()' the shuffle the set
            :param csv_delimiter: delimiter used in csv file
            """
            super().__init__(indent=' ' * 2)

            # if none is given as max_samples, count the number of samples
            #   present in the dataset. Note that this is pretty slow
            if max_samples is None:
                with ZipFile(file, 'r') as zipfile:
                    with zipfile.open('dataset.csv') as file:
                        for idx, _ in enumerate(file.readlines()):
                            pass
                        max_samples = idx

            # ATTRIBUTES
            self.file = file
            self.max_samples = max_samples
            self.img_resolution = img_resolution
            self.train_pct = train_pct
            self.n_subsets = n_subsets
            self.batch_size = batch_size
            self.shuffle_train = shuffle_train
            self.shuffle_valid = shuffle_valid
            self.csv_delimiter = csv_delimiter
            self.dtype = dtype

    def __init__(self, settings: Settings, trans_train=None, trans_valid=None):
        """
        Create instance of custom map-style PyTorch Dataset
        """

        # if dir is a <str>, create a pathlib.Path from it
        if isinstance(settings.file, str):
            settings.file = Path().cwd().joinpath(settings.file)

        # allocate memory for dataset
        #   using float results in ~100 GB of memory being used for the
        #   current dataset, hence uint8 is used instead where possible.
        #   This reduces the memory cost to only <...> GB for the complete
        #   dataset. Furthermore, pre-allocating the data results in much
        #   faster load times.  Pandas took 3h 45min to load the whole
        #   dataset, this function only takes <...>.
        #
        #   Note: the dataset is a zip file, this is because reading from 1
        #   zip container is faster than reading millions of separate image
        #   files.
        dataset = {
            KEY.INDEX: torch.empty(settings.max_samples, dtype=torch.int32),
            KEY.INPUT_IMGS: torch.empty((settings.max_samples,
                                         3,
                                         settings.img_resolution,
                                         settings.img_resolution),
                                        dtype=torch.uint8),
            KEY.INPUT_META: torch.empty((settings.max_samples,
                                         2 * N_ANTENNAS),
                                        dtype=torch.uint8),
            KEY.OUTPUT: torch.empty((settings.max_samples,
                                     1,
                                     settings.img_resolution,
                                     settings.img_resolution),
                                    dtype=torch.uint8),
        }

        # load dataset
        with ZipFile(settings.file, 'r') as zipfile:
            with zipfile.open('dataset.csv') as file:
                # skip header
                next(file)

                # read every line
                for idx, byte_array in enumerate(file.readlines()):
                    # break if max_lines is reached
                    if idx >= settings.max_samples:
                        break

                    # convert bytes to string and split string at delimiter
                    array = byte_array.decode('utf-8') \
                        .split(settings.csv_delimiter)

                    # append sample to dataset
                    _append_sample(
                        dataset=dataset,
                        idx=idx,
                        zipfile=zipfile,
                        index=array[IDX.INDEX],
                        input_imgs=array[IDX.INPUT_IMGS],
                        input_meta=array[IDX.INPUT_META],
                        output=array[IDX.OUTPUT].replace('\n', '')
                    )

        # number of samples that were read
        n_samples = idx

        # reduce dataset if too much space was allocated
        if n_samples < settings.max_samples:
            dataset[KEY.INDEX] = dataset[KEY.INDEX][:n_samples]
            dataset[KEY.INPUT_IMGS] = \
                dataset[KEY.INPUT_IMGS][:n_samples, :, :, :]
            dataset[KEY.INPUT_META] = dataset[KEY.INPUT_META][:n_samples, :]
            dataset[KEY.OUTPUT] = dataset[KEY.OUTPUT][:n_samples, :, :]

        # set transformation functions
        #   Note: if no transformation is provided, ToTensor is used,
        #   since images must always be converted to a tensor
        if trans_train is None:
            trans_train = Forward()
        if trans_valid is None:
            trans_valid = Forward()

        # determine ids for validation/training dataset
        # n_train = int(n_samples * settings.train_pct / 100)
        # ids_train = torch.arange(0, n_train, dtype=torch.int32)
        # ids_valid = torch.arange(n_train, n_samples, dtype=torch.int32)
        # ids_train = np.arange(0, int(n_samples * settings.train_pct / 100))
        # ids_valid = np.arange(int(n_samples * settings.train_pct / 100),
        #                       n_samples)

        # ids_train_subset = torch.split(ids.train, settings.n_subsets)
        # ids_valid_subset = torch.split(ids.train, settings.n_subsets)

        # create train/valid subsets
        ds_train = []
        ds_valid = []
        for idx_subset in range(settings.n_subsets):
            ds_train.append(_Subset(self, 'train', idx_subset, trans_train))
            ds_valid.append(_Subset(self, 'valid', idx_subset, trans_valid))

        # ATTRIBUTES
        self.settings = settings
        self.dataset = dataset
        self._n_train = int(n_samples * settings.train_pct / 100)
        self._n_samples = n_samples
        self.indices = self._generate_indices()

        # create training/validation dataloaders
        dataloaders_train = []
        dataloaders_valid = []
        for ds_t, ds_v in zip(ds_train, ds_valid):
            dataloaders_train.append(
                DataLoader(ds_t, settings.batch_size, shuffle=True)
            )
            dataloaders_valid.append(
                DataLoader(ds_v, settings.batch_size, shuffle=True)
            )

        # ATTRIBUTES
        self.dataloaders_train: List[DataLoader] = dataloaders_train
        self.dataloaders_valid: List[DataLoader] = dataloaders_valid


    def _generate_indices(self):
        """
        This method uses the first 'n_train' number of samples from the
        dataset as the training set, and the remaining number of samples as
        the validation set. These indices are then shuffled (note: they are
        shuffled INDEPENDENTLY for the train/valid set), then each set is
        split further into subsets.
        """

        # generate ids for train/valid set
        #   use the first n_train for the training set, and the remaining
        #   ones for the validation set
        ids_train = np.arange(0, self._n_train)
        ids_valid = np.arange(self._n_train, self._n_samples)

        # shuffle them if desired (shuffle is done inplace)
        if self.settings.shuffle_train:
            np.random.shuffle(ids_train)
        if self.settings.shuffle_valid:
            np.random.shuffle(ids_valid)

        return {
            'train': np.array_split(ids_train, self.settings.n_subsets),
            'valid': np.array_split(ids_valid, self.settings.n_subsets)
        }

    def shuffle(self):
        """
        Simply generate new random indices to shuffle them
        """
        self.indices = self._generate_indices()

    def get(self, idx):
        ds = self.dataset
        dt = self.settings.dtype
        return {
            KEY.INPUT_IMGS: ds[KEY.INPUT_IMGS][idx, :, :, :].type(dt) / 255,
            KEY.INPUT_META: ds[KEY.INPUT_META][idx, :].type(dt) / 255,
            KEY.OUTPUT: ds[KEY.OUTPUT][idx, :, :].type(dt) / 255,
            KEY.INDEX: ds[KEY.INDEX][idx]
        }

    def __len__(self):
        return self._n_samples


class _Subset(torch.utils.data.Dataset):
    """
    Subset of dataset, that uses transformations
    """

    KEY = KEY
    IDX = IDX

    def __init__(self, dataset: Dataset, stype, idx_subset, transform):
        self.dataset = dataset
        self.stype = stype
        self.idx_subset = idx_subset
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.dataset.get(
            self.dataset.indices[self.stype][self.idx_subset][idx])
        )

    def __len__(self):
        return len(self.dataset.indices[self.stype][self.idx_subset])


class Forward:
    def __call__(self, x):
        return x


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, sample):
        for idx, img in enumerate(sample[KEY.INPUT_IMGS]):
            sample[KEY.INPUT_IMGS][idx] = func.to_tensor(img)
        sample[KEY.INPUT_IMGS] = torch.cat(sample[KEY.INPUT_IMGS], dim=0)
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
        # todo: degrees should be defined through argument
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
        for idx, img in enumerate(sample[KEY.INPUT_IMGS]):
            sample[KEY.INPUT_IMGS][idx] = func.rotate(
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
        for idx, img in enumerate(sample[KEY.INPUT_IMGS]):
            sample[KEY.INPUT_IMGS][idx] = func.vflip(img)

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
                 mean_input_img: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                 std_input_img: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                 mean_input_meta: float = 0.5,
                 std_input_meta: float = 0.5,
                 mean_output: float = 0.5,
                 std_output: float = 0.5):
        # ATTRIBUTES
        self.mean_input_img = mean_input_img
        self.std_input_img = std_input_img
        self.mean_input_meta = mean_input_meta
        self.std_input_meta = std_input_meta
        self.mean_output = mean_output
        self.std_output = std_output

    def __call__(self, sample):
        func.normalize(tensor=sample[KEY.INPUT_IMGS],
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

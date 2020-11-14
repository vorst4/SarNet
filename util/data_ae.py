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
from util.data import KEY, IDX, Forward, _Subset
import os

N_ANTENNAS = 12  # todo: don't hardcode this


class DatasetAE:
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
        r = settings.img_resolution
        dataset = torch.zeros((settings.max_samples, 3, r, r),
                              dtype=torch.uint8)

        # load property maps
        with ZipFile(settings.file, 'r') as zipfile:

            paths = zipfile.namelist()
            idx_max = 0
            for path in paths:
                # skip file if not an property map
                if 'input' not in path:
                    continue

                # skip model
                if 'model' in path:
                    continue

                # add conductivity map
                if 'conductivity' in path:
                    idx = int(path.split('_', 1)[1].split('.', 1)[0])
                    if idx > idx_max:
                        idx_max = idx
                    dataset[idx, 1, :, :] = 255 * func.to_tensor(
                        Image.open(zipfile.open(path))
                    )
                    continue

                if 'density' in path:
                    idx = int(path.split('_', 1)[1].split('.', 1)[0])
                    if idx > idx_max:
                        idx_max = idx
                    dataset[idx, 2, :, :] = 255 * func.to_tensor(
                        Image.open(zipfile.open(path))
                    )
                    continue

                if 'permittivity' in path:
                    idx = int(path.split('_', 1)[1].split('.', 1)[0])
                    if idx > idx_max:
                        idx_max = idx
                    dataset[idx, 0, :, :] = 255 * func.to_tensor(
                        Image.open(zipfile.open(path))
                    )
                    continue

        # limit dataset to idx_max
        n_samples = idx_max
        dataset = dataset[:idx, :]

        # set transformation functions
        #   Note: if no transformation is provided, ToTensor is used,
        #   since images must always be converted to a tensor
        if trans_train is None:
            trans_train = Forward()
        if trans_valid is None:
            trans_valid = Forward()

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
            KEY.INPUT_IMGS: ds[idx, :, :, :].type(dt) / 255,
            KEY.INPUT_META: torch.zeros((2 * N_ANTENNAS)),
            KEY.OUTPUT: ds[idx, :, :, :].type(dt) / 255,
            KEY.INDEX: idx
        }

    def __len__(self):
        return self._n_samples

from pathlib import Path
from typing import Union, List

import pandas
import torch
import torch.utils.data
from PIL import Image
from torchvision.transforms import ToTensor

N_ANTENNAS = 12

_img_to_tensor = ToTensor()


def _load_img(path: Union[str, List[str]]):
    """
    Loads 1 or multiple images located at the given path(s) and returns
    these as a tensor with size [n_images, px_width, px_height]
    """
    if isinstance(path, list):
        return torch.cat(
            [_img_to_tensor(Image.open(p)) for p in path], dim=0
        )
    else:
        return _img_to_tensor(Image.open(path))


def _list(slice_: slice) -> []:
    return list(range(*slice_.indices(slice_.stop)))


class IDX:
    INDEX = 0
    INPUT_IMG = slice(1, 4)  # [1, 2, 3]
    INPUT_META = slice(4, 4 + 2 * N_ANTENNAS)  # [4, 5, ..., 27, 28]
    OUTPUT = -1  # 26


class KEY:
    INPUT_IMG = 'input_img'
    INPUT_META = 'input_meta'
    OUTPUT = 'output'


class Dataset(torch.utils.data.Dataset):
    """
    This dataset is implemented according to the map-style dataset of PyTorch
      https://pytorch.org/docs/stable/data.html#map-style-datasets

    The dataset.csv file should look as follows, but without spaces. Output_img
    and input_img are the relative paths to the images
      ___________________________________________________________________________
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

    def __init__(self,
                 dataset_file: Union[str, Path],
                 csv_delimiter: str = ';',
                 n_max: int = None):
        """
        Create instance of custom map-style PyTorch Dataset

        :param dataset_file: <str> or <pathlib.Path> of the dataset csv file
        :param csv_delimiter: (default: ';') delimiter used in the dataset
        csv file
        """

        # if dir is a <str>, create a pathlib.Path from it
        if isinstance(dataset_file, str):
            dataset_file = Path().cwd().joinpath(dataset_file)

        # read dataset.csv into a nested list
        self.dataset = pandas.read_csv(
            dataset_file,
            delimiter=csv_delimiter
        ).values.tolist()

        if n_max is not None:
            self.dataset = self.dataset[:n_max]

        # set length
        self.len = len(self.dataset)

        # get directory in which dataset_file is located
        directory = dataset_file.parents[0]

        # change relative path in dataset to full path
        for idx1 in range(self.len):
            for idx2 in (*_list(IDX.INPUT_IMG), IDX.OUTPUT):
                self.dataset[idx1][idx2] = str(
                    directory.joinpath(self.dataset[idx1][idx2])
                )

        # tensor shapes
        tensors = self.__getitem__(0)
        self.shapes = {
            KEY.INPUT_IMG: tuple(tensors[KEY.INPUT_IMG].shape),
            KEY.INPUT_META: tuple(tensors[KEY.INPUT_META].shape),
            KEY.OUTPUT: tuple(tensors[KEY.OUTPUT].shape),
        }

    def __getitem__(self, idx):
        return {
            KEY.INPUT_IMG: _load_img(self.dataset[idx][IDX.INPUT_IMG]),
            KEY.INPUT_META: torch.tensor(self.dataset[idx][IDX.INPUT_META]),
            KEY.OUTPUT: _load_img(self.dataset[idx][IDX.OUTPUT]),
        }

    def __len__(self):
        return self.len

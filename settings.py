import os

from util.progress import Progress
from util.data import Dataset
from util.log import Log
import torch

RUNNING_ON_DESKTOP = os.name == 'nt'  # nt: Windows , posix: Linux
IMG_RESOLUTION = 32  # todo: obtain this from dynamically from dataset
N_ANTENNAS = 12  # todo: obtain this from dynamically from dataset

# -----------------------------------------------------------------------------
# SETTINGS WINDOWS
if RUNNING_ON_DESKTOP:

    MAX_SAMPLES = 70399

    # dataset settings
    dataset = Dataset.Settings(
        file='dataset_sar.zip',
        max_samples=MAX_SAMPLES,
        train_pct=90,
        n_subsets=100,
        shuffle_train=True,
        shuffle_valid=True,
    )

    log = Log.Settings(
        directory='output',
        save_log=False,
    )

    dropout_rate = 0.2

    # log the timer or not
    log_timer = False

    # batch size (int)
    batch_size = 8

    # batch normalisation settings
    batch_norm = {'momentum': 0.99, 'eps': 1e-3}

    # settings learning-rate tuner
    # beta1s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
    epochs = 10

    # progress logging settings
    progress = Progress.Settings(
        print_=True,
        lossplot=False,
        preview=False,
        load_design=False,
        save_design=True,
        path='C:/Users/Dennis/Documents/desktop_resnet_output',
        save_lossplot=True,
        save_preview=True,
    )

# -----------------------------------------------------------------------------
# SETTINGS SERVER
else:

    MAX_SAMPLES = 10236800

    # dataset
    dataset = Dataset.Settings(
        file='dataset_sar.zip',
        max_samples=MAX_SAMPLES // 100,
        train_pct=90,
        n_subsets=1000,
        shuffle_train=True,
        shuffle_valid=True,
    )

    # log
    log = Log.Settings(
        directory='output',
        save_log=True,
    )

    dropout_rate = 0.2

    # log the timer or not
    log_timer = False

    # batch size (int)
    batch_size = 128

    # path of dataset (str): the path at which the dataset csv file is located.
    path_dataset = 'dataset_sar.zip'

    # batch normalisation settings
    batch_norm = {'momentum': 0.99, 'eps': 1e-3}

    # settings learning-rate tuner
    # beta1s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
    epochs = 10

    # progress logging settings
    progress = Progress.Settings(
        save_design=True,
        load_design=False,
        path='/home/tue/s111167/trained_models',
        save_lossplot=True,
        save_preview=True,
    )

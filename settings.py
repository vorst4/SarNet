import os

from util.progress import Progress
from util.data import Dataset
import torch

is_running_on_desktop = os.name == 'nt'  # nt: Windows , posix: Linux

# -----------------------------------------------------------------------------
# SETTINGS WINDOWS
if is_running_on_desktop:

    MAX_SAMPLES = 70399

    # dataset settings
    dataset = Dataset.Settings(
        file='dataset_sar.zip',
        max_samples=1000,
        train_pct=90,
        n_subsets=100,
        shuffle_train=True,
        shuffle_valid=True,
    )

    dropout_rate = 0.5

    # directory of log (str): the directory in which the log (.txt) is saved
    directory_log = 'output'

    # save log (bool): if the log needs to be saved or not
    save_log = False

    # log the timer or not
    log_timer = False

    # img resolution that is used
    img_resolution = 32  # todo: obtain this from dynamically from dataset

    # batch size (int)
    batch_size = 8

    n_antennas = 12  # todo: obtain this from dynamically from dataset

    # batch normalisation settings
    batch_norm = {'momentum': 0.99, 'eps': 1e-3}

    # settings learning-rate tuner
    beta1s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
    epochs = 20

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

    # dataset settings
    dataset = Dataset.Settings(
        file='dataset_sar.zip',
        max_samples=None,
        train_pct=90,
        n_subsets=1000,
        shuffle_train=True,
        shuffle_valid=True,
    )

    dropout_rate = 0.5

    # directory of log (str): the directory in which the log (.txt) is saved
    directory_log = 'output'

    # save log (bool): if the log needs to be saved or not
    save_log = True

    # log the timer or not
    log_timer = False

    # batch size (int)
    batch_size = 128 * 2

    n_antennas = 12

    # img resolution that is used
    img_resolution = 32

    # path of dataset (str): the path at which the dataset csv file is located.
    path_dataset = 'dataset_sar.zip'

    # batch normalisation settings
    batch_norm = {'momentum': 0.99, 'eps': 1e-3}

    # settings learning-rate tuner
    # beta1s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
    epochs = 20

    # progress logging settings
    progress = Progress.Settings(
        save_design=True,
        load_design=False,
        path='/home/tue/s111167/trained_models',
        save_lossplot=True,
        save_preview=True,
    )

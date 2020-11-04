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
        # file='dataset_msf.zip',
        max_samples=100,
        train_pct=90,
        n_subsets=10,
        shuffle_train=True,
        shuffle_valid=True,
    )

    log = Log.Settings(
        directory=None,
        save_log=True,
    )

    dropout_rate = 0.5

    # log the timer or not
    log_timer = False

    # batch size (int)
    batch_size = 128

    # batch normalisation settings
    batch_norm = {'momentum': 0.99, 'eps': 1e-3}

    # settings learning-rate tuner
    # beta1s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
    epochs = 1

    # progress logging settings
    progress = Progress.Settings(
        print_=True,
        lossplot=False,
        preview=False,
        load_design=False,
        save_design=True,
        path='C:/Users/Dennis/Documents/desktop_resnet_output/' +
             Log.date_time(),
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
        # file='dataset_msf.zip',
        max_samples=MAX_SAMPLES,
        train_pct=90,
        n_subsets=1000,
        shuffle_train=True,
        shuffle_valid=True,
    )

    # log
    log = Log.Settings(
        directory=None,
        save_log=True,
    )

    dropout_rate = 0.5

    # log the timer or not
    log_timer = False

    # batch size (int)
    batch_size = 128

    # batch normalisation settings
    batch_norm = {'momentum': 0.99, 'eps': 1e-3}

    # settings learning-rate tuner
    # beta1s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
    epochs = 20

    # progress logging settings
    progress = Progress.Settings(
        save_design=True,
        load_design=False,
        path='/home/tue/s111167/trained_models/' + Log.date_time(),
        save_lossplot=True,
        save_preview=True,
    )


def info(indent: str = ' ' * 2):
    s = ''
    for key, att in globals().items():
        if (
                key[0] != '_'
                and 'module' not in str(att)
                and 'class' not in str(att)
                and 'function' not in str(att)
        ):
            s += '\n%s%s: %s' % (indent, key, str(att).replace('\n',
                                                               '\n'+indent))

    return s[1:]

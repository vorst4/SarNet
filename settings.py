import os

from util.progress import Progress

is_running_on_desktop = os.name == 'nt'  # nt: Windows , posix: Linux

# -----------------------------------------------------------------------------
# SETTINGS WINDOWS
if is_running_on_desktop:
    # directory of log (str): the directory in which the log (.txt) is saved
    directory_log = 'output'

    # save log (bool): if the log needs to be saved or not
    save_log = False

    # log the timer or not
    log_timer = False

    # train percentage (float): percentage of the dataset that is to be used
    #   for training, the rest is used for evaluation.
    train_pct = 50  # in percent

    # batch size (int)
    batch_size = 16

    # number of train/valid subsets
    #   The evaluation set (can be) split into N unique subsets that are
    #   evaluated at equal intervals during training. Reason is that 1 epoch
    #   takes a very long time, during which a lot of change can occur.
    #   Thus, evaluating the dataset at multiple times during an epoch can
    #   provide more information on how the network is improving during
    #   training.
    n_subsets = 10

    # path of dataset (str): the path at which the dataset csv file is located.
    path_dataset = 'C:/Users/Dennis/Documents/dataset_sar/dataset.csv'

    # length of dataset (float): maximum length (in samples) of the dataset
    #   to be used, set to None to use the whole dataset
    len_dataset = 100

    # batch normalisation settings
    batch_norm = {'momentum': 0.99, 'eps': 1e-3}

    # settings learning-rate tuner
    beta1s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
    epochs = 20

    # progress logging settings
    progress_settings = Progress.Settings(
        print_=True,
        lossplot=False,
        preview=False,
        load_design=True,
        save_design=True,
        path='C:/Users/Dennis/Documents/desktop_resnet_output',
        save_lossplot=True,
        save_preview=True,
    )

# -----------------------------------------------------------------------------
# SETTINGS SERVER
else:
    # directory of log (str): the directory in which the log (.txt) is saved
    directory_log = 'output'

    # save log (bool): if the log needs to be saved or not
    save_log = True

    # log the timer or not
    log_timer = False

    # train percentage (float): percentage of the dataset that is to be used
    #   for training, the rest is used for evaluation.
    train_pct = 90  # in percent

    # batch size (int)
    batch_size = 128 * 2

    # number of train/valid subsets
    #   The evaluation set (can be) split into N unique subsets that are
    #   evaluated at equal intervals during training. Reason is that 1 epoch
    #   takes a very long time, during which a lot of change can occur.
    #   Thus, evaluating the dataset at multiple times during an epoch can
    #   provide more information on how the network is improving during
    #   training.
    n_subsets = 10

    # path of dataset (str): the path at which the dataset csv file is located.
    path_dataset = 'dataset_sar/dataset.csv'

    # length of dataset (float): maximum length (in samples) of the dataset
    #   to be used, set to None to use the whole dataset
    len_dataset = 100000

    # batch normalisation settings
    batch_norm = {'momentum': 0.99, 'eps': 1e-3}

    # settings learning-rate tuner
    # beta1s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
    epochs = 20

    # progress logging settings
    progress_settings = Progress.Settings(
        save_design=True,
        load_design=False,
        path='/home/tue/s111167/trained_models_gpu',
        save_lossplot=True,
        save_preview=True,
    )

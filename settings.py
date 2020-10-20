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
    log_timer = True

    # train percentage (float): percentage of the dataset that is to be used
    #   for training, the rest is used for evaluation.
    train_pct = 90  # in percent

    # batch size (int)
    batch_size = 20

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
    len_dataset = 10000

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
        load_design=False,
        save_design=False,
        save_interval=1,
        path='C:/Users/Dennis/Documents/tmp',
        save_lossplot=False,
        save_preview=False,
    )

# -----------------------------------------------------------------------------
# SETTINGS SERVER
else:
    # directory of log (str): the directory in which the log (.txt) is saved
    directory_log = 'output'

    # save log (bool): if the log needs to be saved or not
    save_log = True

    # log the timer or not
    log_timer = True

    # train percentage (float): percentage of the dataset that is to be used
    #   for training, the rest is used for evaluation.
    train_pct = 98  # in percent

    # batch size (int)
    batch_size = 100

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
    len_dataset = None

    # batch normalisation settings
    batch_norm = {'momentum': 0.99, 'eps': 1e-3}

    # settings learning-rate tuner
    # beta1s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
    epochs = 20

    # progress logging settings
    progress_settings = Progress.Settings(
        save_design=False,
        load_design=False,
        save_interval=15,
        path='/home/tue/s111167/trained_models_gpu',
        save_lossplot=False,
        save_preview=False,
    )

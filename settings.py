import os

from util.progress import Progress

is_running_on_desktop = os.name == 'nt'  # nt: Windows , posix: Linux

# -----------------------------------------------------------------------------
# SETTINGS WINDOWS
if is_running_on_desktop:
    # misc
    directory_log = 'output'
    save_log = False
    train_pct = 90  # in percent
    path_dataset = 'C:/Users/Dennis/Documents/dataset_sar/dataset.csv'
    batch_size = 20
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
        load_design=True,
        save_design=True,
        save_interval=1,
        path='C:/Users/Dennis/Documents/tmp',
        save_lossplot=True,
        save_preview=True,
    )

# -----------------------------------------------------------------------------
# SETTINGS SERVER
else:
    # misc
    directory_log = 'output'
    save_log = True
    train_pct = 90  # in percent
    batch_size = 100
    path_dataset = 'dataset_sar/dataset.csv'
    len_dataset = None

    # batch normalisation settings
    batch_norm = {'momentum': 0.99, 'eps': 1e-3}

    # settings learning-rate tuner
    beta1s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
    epochs = 20

    # progress logging settings
    progress_settings = Progress.Settings(
        save_design=True,
        load_design=True,
        save_interval=15,
        path='/home/tue/s111167/trained_models',
        save_lossplot=True,
        save_preview=True,
    )

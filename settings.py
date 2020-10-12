import os
from util.progress import Progress

# misc
is_running_on_desktop = os.name == 'nt'  # nt: Windows , posix: Linux
directory_log = 'output'
if is_running_on_desktop:
    save_log = False
else:
    save_log = True

# settings dataset
train_pct = 90  # in percent
if is_running_on_desktop:
    path_dataset = 'C:/Users/Dennis/Documents/dataset_sar/dataset.csv'
    batch_size = 30
    len_dataset = 10
else:
    batch_size = 302
    len_dataset = None

# settings learning-rate tuner
beta1s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
epochs = 20

# settings of progress tracking
if is_running_on_desktop:
    progress_settings = Progress.Settings(
        print=True,
        lossplot=False,
        preview=False,
        load_design=True,
        save_design=True,
        path='C:/Users/Dennis/Documents/tmp',
        save_lossplot=True,
        save_preview=True,
    )
else:
    progress_settings = Progress.Settings(
        save_design=True,
        load_design=True,
        path='/home/tue/s111167/trained_models',
        save_lossplot=True,
        save_preview=True,
    )


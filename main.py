import argparse
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as transform
from torch.utils.data import DataLoader

import settings
from util.data import Dataset, RandomRotation, RandomVerticalFlip, \
    Normalize, ToTensor
from util.design import Design
from util.log import Log
from util.timer import Timer


# -------------------------------- ARGUMENTS -------------------------------- #
# On the server, several parameters should be passed when running the
# script, this is not necessary when running it on the desktop
if settings.is_running_on_desktop:
    partition_id = 0
    job_id = 1
    n_cpus = 4
    n_gpus = 1
else:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition_id", help="server partition id", type=int)
    parser.add_argument('--job_id', help='job id', type=int)
    parser.add_argument('--n_cpus', help='number of cpus assigned', type=int)
    parser.add_argument('--n_gpus', help='number of gpus assigned', type=int)
    partition_id = parser.parse_args().partition_id
    job_id = parser.parse_args().job_id
    n_cpus = parser.parse_args().n_cpus
    n_gpus = parser.parse_args().n_gpus

# ----------------------------------- MISC ---------------------------------- #
# models to be run
modelname = ['ResNetAE1', 'ConvAE1', 'DenseAE1'][job_id]

# set/create root path from modelname
settings.progress_settings.path += '/' + modelname
if not Path(settings.progress_settings.path).exists():
    Path(settings.progress_settings.path).mkdir()

# initialize log
log = Log(directory=settings.directory_log,
          prefix='log_%i_' % job_id,
          save_log=settings.save_log)

# print memory usage
log.logprint('memory usage (cpu): ' + log.cpu_memory_usage())

# create timer object
timer = Timer(log)

log.logprint(log.memory_usage())

# --------------------------------- DATASET --------------------------------- #

# log
log.logprint('setting up dataset')

# transformation functions to be applied to training and validation dataset
trans_train = transform.Compose([
    RandomRotation(),
    RandomVerticalFlip(),
    ToTensor(),
    Normalize(),
])
trans_val = transform.Compose([
    ToTensor(),
    Normalize(),
])

# dataset
timer.start()
ds = Dataset(
    dataset_file=settings.path_dataset,
    train_pct=settings.train_pct,
    n_subsets=settings.n_subsets,
    n_max=settings.len_dataset,
    trans_train=trans_train,
    trans_val=trans_val
)
ds_train, ds_val = ds.training, ds.validation
timer.stop('created dataset')

# create dataloader for both sets
timer.start()
dl_train = []
dl_val = []
for ds_t, ds_v in zip(ds_train, ds_val):
    dl_train.append(
        DataLoader(ds_t, settings.batch_size, shuffle=True)
    )
    dl_val.append(
        DataLoader(ds_v, settings.batch_size, shuffle=False)
    )
timer.stop('created dataloaders')

# log memory usage
log.logprint('...done, memory usage (cpu): ' + log.cpu_memory_usage())

# --------------------------------- DESIGN ---------------------------------- #

# loss function
lf = torch.nn.MSELoss()

# model & optimizer
log.logprint('initializing model...')
model = Design.get_model_from_name(modelname)
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-4
)
log.logprint('...done')

# design
log.logprint('creating design...')
timer.start()
design = Design().create(
    dl_train, dl_val, model, lf, optimizer, None, settings.epochs, log
)
timer.stop('created design')
log.logprint('...done')

# ---------------------------------- START ---------------------------------- #
# start training
log.logprint('--- training start --- ')
design.train(settings.progress_settings)
log.logprint('--- training end ---')

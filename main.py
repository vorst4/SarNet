import argparse
from pathlib import Path
import torch
import torch.nn as nn
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
if settings.RUNNING_ON_DESKTOP:
    partition_id = 0
    job_id = 0
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

# choose learning rate & model, based on job & partition id
lr = 1e-4
modelname = ['SarNetLN',
             'SarNetLS',
             'SarNetCN',
             'SarNetCS',
             'SarNetRN',
             'SarNetRS',
             'SarNetMN',
             'SarNetMS'][job_id]
modelname = 'SarNetRV'
lr = [1e-6, 1e-7][job_id]

# set/create root path from modelname
settings.progress.path = str(Path(str(settings.progress.path)).joinpath(
    '%s_%s_d%s_s%i_j%i' % (settings.ds, modelname, Log.date_time(),
                           partition_id, job_id)
))
if not Path(settings.progress.path).exists():
    Path(settings.progress.path).mkdir()

# initialize log
if settings.log.directory is None:
    settings.log.directory = str(settings.progress.path)
settings.log.filename_prefix += modelname + '_'
log = Log(settings.log, job_id=job_id, server_id=partition_id,
          n_cpus=n_cpus, n_gpus=n_gpus)

# log settings
log('settings:\n' + settings.info())

# print memory usage
log('memory usage (cpu): ' + log.cpu_memory_usage())

# create timer object
timer = Timer(log)

# ------------------------------ DATASET ------------------------------ #

# log
log('setting up dataset')

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

# create dataset and obtain DataLoader objects for train/valid set
timer.start()
ds = Dataset(settings.dataset,
             # trans_train=trans_train,  # todo: check if these are correct
             # trans_val=trans_val  # todo: check if these are correct
             )
log('  dataset file: %s' % str(ds.settings.file))
dls_train, dls_valid = ds.dataloaders_train, ds.dataloaders_valid
timer.stop('created dataset')

# log memory usage
log('...done, memory usage (cpu): ' + log.cpu_memory_usage())

# ------------------------------ DESIGN ------------------------------- #

# loss function use BCE for variational AE and MSE for the others
# lf = nn.BCELoss(reduction='sum') if 'V' in modelname else nn.MSELoss()
lf = nn.MSELoss()
log('using loss function: %s' % str(lf))

# model & optimizer
log('initializing model...')
model = Design.get_model_from_name(modelname)
log('...done')

optimizer = torch.optim.Adam(
    model.parameters(), lr=lr
)
log('using optimizer: \n  %s' % str(optimizer).replace('\n', '\n  '))

# design
log('creating design...')
timer.start()
design = Design().create(
    dls_train, dls_valid, model, lf, optimizer, None, settings.epochs, log
)
timer.stop('created design')
log('...done')

# ------------------------------- START ------------------------------- #
# start training
log('--- training start --- ')
design.train(settings.progress)
log('--- training end ---')

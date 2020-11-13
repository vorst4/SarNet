import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as transform
from torch.utils.data import DataLoader

import settings
import util.data as data
from util.design import Design
from util.log import Log
from util.timer import Timer

from util.report import Report

# Report()
# exit()

# -------------------------------- ARGUMENTS -------------------------------- #
# On the server, several parameters should be passed when running the
# script, this is not necessary when running it on the desktop
if settings.RUNNING_ON_DESKTOP:
    partition_id = 1
    job_id = 5
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
# lr = 1e-7
modelname = [
    'SarNetL',  # 0
    'SarNetC',  # 1
    'SarNetRN',  # 2
    'SarNetRS',  # 3
    'SarNetM',  # 4
    'SarNetRV',  # 5
    'SarNetRN',  # 6  , including data augmentations
][job_id]

# set/create root path from modelname
settings.progress.path = settings.progress.add_subdir(settings.progress.path,
                                                      modelname,
                                                      partition_id,
                                                      job_id)

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

# transformation functions to be applied to training data
if job_id is 6:
    trans_train = transform.Compose([
        data.RandomRotation90(),
        data.RandomVerticalFlip(),
    ])
else:
    trans_train = None

# create dataset and obtain DataLoader objects for train/valid set
timer.start()
ds = data.Dataset(settings.dataset, trans_train=trans_train)
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

# optimizer
lr_init = settings.learning_rate.initial
lr_init = model.lr_ideal if lr_init is None else lr_init
optimizer = torch.optim.Adam(
    model.parameters(), lr=lr_init
)
log('using optimizer: \n  %s' % str(optimizer).replace('\n', '\n  '))

# learning rate scheduler
lr_sched = torch.optim.lr_scheduler.StepLR(
    optimizer=optimizer,
    step_size=settings.learning_rate.step_size,
    gamma=settings.learning_rate.gamma,
)

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

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import settings
from util.dataset import Dataset
from util.design import Design
from util.log import Log

# on the server, the partition id is passed as an argument
if settings.is_running_on_desktop:
    partition_id = 2
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

# models to be run
modelname = ['ResNetAE1', 'ConvAE1', 'DenseAE1'][job_id]

# set/create root path from modelname
settings.progress_settings.path += '/' + modelname
if not Path(settings.progress_settings.path).exists():
    Path(settings.progress_settings.path).mkdir()

# create log
log = Log(directory=settings.directory_log,
          prefix='log_%i_' % job_id,
          save_log=settings.save_log)

# setup dataset (ds) and dataloader (dl)
ds = Dataset(settings.path_dataset, n_max=settings.len_dataset)
n_train = int(settings.train_pct / 100 * len(ds))
n_val = int((100 - settings.train_pct) / 100 * len(ds))
dl_train = DataLoader(ds,
                      batch_size=settings.batch_size,
                      sampler=SubsetRandomSampler(range(n_train)))
dl_val = DataLoader(ds,
                    batch_size=settings.batch_size,
                    sampler=SubsetRandomSampler(range(n_train, len(ds))))

# loss function
lf = torch.nn.MSELoss()

# model & optimizer
model = Design.get_model_from_name(modelname)
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-4
)

# design
design = Design().create(
    dl_train, dl_val, model, lf, optimizer, None, settings.epochs, log
)

# start training
design.train(settings.progress_settings)

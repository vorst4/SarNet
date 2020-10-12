from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import pathlib
from pathlib import Path
import settings
from util.design import Design
from util.log import Log
from util.dataset import Dataset
import numpy as np

# modelname = Design.get_available_modelnames()[0]
modelname = 'ResNetAE1'

log = Log(settings.directory_log, save_log=settings.save_log)

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

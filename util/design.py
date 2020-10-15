import abc
import json
import os
from pathlib import Path
from time import time
from typing import Union, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader, Subset

import models
from .dataset import Dataset
from .log import Log
from .performance import PerformanceParameter2D, PerformanceParameter3D
from .progress import Progress


class Design:
    """
    The 'design' combines the model, optimiser, loss_function, data loaders,
     training etc.

    After initializing this object, either call method 'create' or method
    'load', depending on if you want to create
    a new design or load an existing one.
    development note: Python does not support multiple __init__ functions,
    and using **kwargs severely hinders the
                      readability of the code. Hence, creating/loading is
                      done separately from __init__

    NOTE: The training and validation loss are stored in an array. The
    memory for this array is allocated during
          __init__. When calling the append_loss function, the element is
          actually written to a certain index,
          not appended. Reason for this is that appending requires the array
          to be reallocated in the memory,
          resulting in a performance dip when large arrays are involved.
    """

    dtype = torch.float32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _torch_seed = 123456789

    def __init__(self):
        self.n_phase = 360
        self._dl_train = None
        self._dl_val = None
        self._model = None
        self._loss_function = None
        self._optimizer = None
        self._lr_sched = None
        self._log: Log = None
        self._epoch_start = None
        self._epoch_current = None
        self._epoch_stop = None
        self._train_mse_per_epoch = None
        self._valid_mse_per_epoch = None
        self._valid_mse_per_phase_per_epoch = None
        self._valid_tae_per_phase_per_epoch = None
        self._timer = None
        self.saved_after_epoch = None
        self.n_batches = None
        self.idx_batch = None
        torch.manual_seed(self._torch_seed)

    def create(self,
               dataloader_training: torch.utils.data.DataLoader,
               dataloader_validation: torch.utils.data.DataLoader,
               model: nn.Module,
               loss_function: nn.Module,
               optimizer: torch.optim,
               lr_scheduler: torch.optim.lr_scheduler,
               epochs: int,
               log: Log):
        self._dl_train = dataloader_training
        self._dl_val = dataloader_validation
        self._model = model
        self._loss_function = loss_function
        self._optimizer = optimizer
        self._lr_sched = lr_scheduler
        self._log = log
        self._epoch_start = 0
        self._epoch_current = 0
        self._epoch_stop = epochs

        # initialize performance parameters
        self._train_mse_per_epoch = PerformanceParameter2D('mse_train',
                                                           'epoch', epochs)
        self._valid_mse_per_epoch = PerformanceParameter2D('mse_valid',
                                                           'epoch', epochs)
        self._valid_mse_per_phase_per_epoch = \
            PerformanceParameter3D('mse_valid', 'phase', 'epoch',
                                   (self.n_phase, epochs))
        self._valid_tae_per_phase_per_epoch = \
            PerformanceParameter3D('tae_valid', 'phase', 'epoch',
                                   (self.n_phase, epochs))

        # log
        log.logprint('Created Design')
        log.logprint('  model: ' + str(type(self._model)))
        log.logprint('  device: ' + str(self.device))
        log.logprint(
            '  dataset size training: %i' % len(self._dl_train.dataset))
        log.logprint(
            '  dataset size validation: %i' % len(self._dl_val.dataset))
        log.logprint('  batch size: %i' % self._dl_train.batch_size)
        n_parameters = sum(
            parameter.numel() for parameter in self._model.parameters())
        log.logprint('  number of model parameters: %s' %
                     '{:,}'.format(n_parameters))
        log.logprint('  memory usage (cpu): ' + log.memory_usage())
        if torch.cuda.is_available():
            log.logprint('  memory usage (gpu): \n' +
                         torch.cuda.memory_summary())

        return self

    def get_train_mse_per_epoch(self) -> list:
        """
        Get epochs -corresponding to the calculated loss-, obtained during
        training
        """
        return self._train_mse_per_epoch()

    def get_valid_mse_per_epoch(self) -> list:
        """
        Get calculated loss per epoch, obtained during training
        """
        return self._valid_mse_per_epoch()

    def get_valid_mse_per_phase_per_epoch(self) -> list:
        """
        Get epochs -corresponding to the calculated loss-, obtained during
        validation
        """
        return self._valid_mse_per_phase_per_epoch()

    def get_valid_tae_per_phase_per_epoch(self) -> list:
        """
        Get epochs -corresponding to the calculated loss-, obtained during
        validation
        """
        return self._valid_tae_per_phase_per_epoch()

    def get_epoch_stop(self) -> int:
        """
        Get epoch at which the training stops
        """
        return self._epoch_stop

    def get_epoch_current(self) -> int:
        """
        Get epoch at which the training stops
        """
        return self._epoch_current

    @staticmethod
    def get_available_modelnames() -> Tuple[str]:
        """
        returns modelnames of models that are available
        """
        modelnames = []
        for att in dir(models):
            if att[0] != '_' and isinstance(getattr(models, att), abc.ABCMeta):
                modelnames.append(att)
        return tuple(modelnames)

    @classmethod
    def get_ideal_learning_rates(cls) -> dict:
        lrs = {}
        for name in cls.get_available_modelnames():
            model = getattr(models, name)
            lr = None
            if hasattr(model, 'lr_ideal'):
                lr = model.lr_ideal
            lrs[name] = lr
        return lrs

    @classmethod
    def get_model_from_name(cls, name: [str]) -> nn.Module:
        """
        if given name is an existing model, the model is returned
        """
        if name not in cls.get_available_modelnames():
            raise ValueError(
                'ERROR: model name: "%s" is not one of the available '
                'models: %s' %
                (name, str(cls.get_available_modelnames())))
        return getattr(models, name)()

    @classmethod
    def get_models_from_names(cls, names: List[str]) -> List[nn.Module]:
        """
        If given list of names are models, list of models is returned
        """
        models_ = []
        for name in names:
            models_.append(cls.get_model_from_name(name))
        return models_

    @classmethod
    def get_default_dataloaders(
            cls,
            path_dataset: Union[str, Path],
            batch_size: int
    ) -> Tuple[DataLoader, DataLoader]:
        split_pct = 0.9
        dataset = Dataset(path_dataset)
        n_training = int(split_pct * len(dataset))
        dataset_training = Subset(dataset, indices=range(0, n_training))
        dataset_validation = Subset(dataset,
                                    indices=range(n_training, len(dataset)))
        dataloader_training = DataLoader(dataset_training, batch_size,
                                         shuffle=True)
        dataloader_validation = DataLoader(dataset_validation, batch_size,
                                           shuffle=False)
        return dataloader_training, dataloader_validation

    def set_epoch_stop(self, epoch_stop: int):
        """
        Set new epoch value at which the training should stop.
        """

        # stopping epoch can't be set before current epoch
        if epoch_stop <= self._epoch_current:
            epoch_stop = self._epoch_current

        # set epoch stop
        self._epoch_stop = epoch_stop

        # allocate space for the performance parameters
        self._train_mse_per_epoch.allocate(epoch_stop)
        self._valid_mse_per_epoch.allocate(epoch_stop)
        self._valid_mse_per_phase_per_epoch.allocate(self.n_phase * epoch_stop)
        self._valid_tae_per_phase_per_epoch.allocate(self.n_phase * epoch_stop)

    def save(self, path, name=None, backups: List[Path] = None):
        """
        Save model to <path>, the filename can either be included in <path>
        or provided separately as <name>

        There is also the option for backups, reason being that terminating the
        script while the model is being saved results in a corrupted file.
        <backups> should be a list of pathlib.Path, with each Path being the
        path of the backup file. More concretely, the last backup (backup[-1])
        will be deleted, backups[idx] are renamed to backups[idx+1], and
        path (+name) is renamed to backup[0].
        """

        # if path is string, convert it to pathlib.Path
        if isinstance(path, str):
            path = Path(path)

        # if name is given, join folder and name, otherwise just use path
        if name is not None:
            path = path.joinpath(name)

        # manage backups
        if backups is not None:

            # remove oldest backup
            if backups[-1].exists():
                os.remove(backups[-1])

            # rename other backups (if they exist)
            for idx in range(len(backups) - 2, -1, -1):
                if backups[idx].exists():
                    backups[idx].rename(backups[idx + 1])

            # create backup from given path
            if path.exists():
                path.rename(backups[0])

        # save
        torch.save(self._to_dict(), path)

    def load(self,
             file: Union[str, Path],
             backup: Union[str, Path],
             dl_train: DataLoader,
             dl_val: DataLoader, log: Log):
        """
        Load Design class from <file>, also set the new number of <epochs>
        to be run and the <log> file to write to.
        """

        # if file is string -> convert to pathlib.Path
        if isinstance(file, str):
            file = Path(file)

        # check if file exists
        if not file.exists():
            raise Exception('ERROR: file: "%s" does not exist' % str(file))

        # load file
        try:
            self._from_dict(torch.load(file,
                                       map_location=torch.device('cpu')))
        except RuntimeError:
            # if failed, try loading the backup
            print('WARNING: loading file %s failed, trying to load backup' %
                  file)
            if isinstance(backup, str):
                backup = Path(backup)
            if not backup.exists():
                raise Exception('ERROR: could not load file %s and backup %s'
                                ' does not exist' % (file, backup))

            self._from_dict(torch.load(backup,
                                       map_location=torch.device('cpu')))
            print('INFO: successfully loaded backup')

        # set non serializable attributes
        self._log = log
        self._dl_train = dl_train
        self._dl_val = dl_val

        # add 1 to current epoch, otherwise the last iteration will be run
        # again
        self._epoch_current += 1

        return self

    def train(self, progress_settings: Progress.Settings):
        """
        Train and evaluate the model/design, the <progress> during training
        is displayed/saved according to its
        settings.
        """

        # initialize progress display/saver
        res_img_out = self._dl_train.dataset.shapes[Dataset.KEY.OUTPUT][1]
        progress = Progress(progress_settings, self, self._log,
                            res_img_out)

        # load previous model state if needed
        if progress.load_design:
            if progress.path_design.exists():
                self.load(progress.path_design,
                          progress.path_design_backup[0],
                          self._dl_train,
                          self._dl_val,
                          self._log)

        # reset starting epoch
        self._epoch_start = self._epoch_current

        # move model to gpu/cpu
        self._model = self._model.to(device=self.device)

        # start timer
        self._timer = time()

        # loop over epochs
        for self._epoch_current in range(self._epoch_start, self._epoch_stop):
            # train
            mse_train = self.train_1epoch(progress)

            # evaluate
            img_true, img_pred, mse_valid, mse_p, tae_p = self._evaluate()

            # log progress
            self._train_mse_per_epoch.append(self._epoch_current, mse_train)
            self._valid_mse_per_epoch.append(self._epoch_current, mse_valid)
            self._valid_mse_per_phase_per_epoch.append(
                self._epoch_current,
                mse_p['phase'],
                mse_p['mse']
            )
            self._valid_tae_per_phase_per_epoch.append(
                self._epoch_current,
                tae_p['phase'],
                tae_p['tae']
            )

            # show/save progress
            #   img_true and img_pred are the true and predicted images of
            #   the last batch, these are used for the
            #   preview
            self.saved_after_epoch = True
            progress(img_true, img_pred)

            # # update learning rate
            # if lr_sched is not None:
            #     lr_sched.step()

        # epoch current isn't updated after the last iteration, so this has
        # to be done manually
        self._epoch_current += 1

        # move model back to cpu
        self._model = self._model.cpu()

    def train_1epoch(self, progress: Progress):
        """
        Train model for 1 epoch
        """

        self.n_batches = len(self._dl_train)
        mse = 0.0

        # loop over batches
        for self.idx_batch, data in enumerate(self._dl_train):
            # make sure model is in train mode
            self._model.train()

            # abbreviate variables and move them to <device>
            kwargs = {'device': self.device, 'dtype': self.dtype}
            yt = data[Dataset.KEY.OUTPUT].to(**kwargs)
            x_img = data[Dataset.KEY.INPUT_IMG].to(**kwargs)
            x_meta = data[Dataset.KEY.INPUT_META].to(**kwargs)

            # calculate predicted outcome and loss
            yp = self._model(x_img, x_meta)
            loss = self._loss_function(yp, yt)

            # calculate gradient (a.k.a. backward propagation)
            loss.backward()

            # update model parameters using the gradient
            self._optimizer.step()

            mse += loss.cpu().detach().item() / self.n_batches

            # save model if interval has passed (and reset timer)
            if (time() - self._timer) / 60 > \
                    progress.settings.save_interval:
                self._timer = time()
                self.saved_after_epoch = False
                # update progress (& save)
                progress(yt, yp, mse_batch=mse)

        return mse

    def _evaluate(self):
        """
        evaluate model on validation dataset
        """

        # set model to evaluation mode
        self._model.eval()

        n_batches = len(self._dl_val)
        mse1 = 0.0
        tae = {'phase': [], 'tae': []}
        mse = {'phase': [], 'mse': []}

        # gradient does not have to be calculated during evaluation
        with torch.no_grad():
            # loop over the evaluation dataset
            for idx_batch, data in enumerate(self._dl_val):
                # abbreviate variables and move to <device>
                yt = data[Dataset.KEY.OUTPUT].to(device=self.device,
                                                 dtype=self.dtype)
                x1 = data[Dataset.KEY.INPUT_IMG].to(device=self.device,
                                                    dtype=self.dtype)
                x2 = data[Dataset.KEY.INPUT_META].to(device=self.device,
                                                     dtype=self.dtype)

                # calculate predicted outcome and loss
                yp = self._model(x1, x2)
                loss = self._loss_function(yp, yt)

                mse1 += loss.cpu().detach().item() / n_batches
                phase = (x2[:, 1].cpu().detach() * 180 / np.pi).tolist()
                tae['phase'].extend(phase)
                tae['tae'].extend(self._calc_tae(torch.abs(yt - yp)))
                mse['phase'].extend(phase)
                mse['mse'].extend(self._calc_mse(torch.abs(yt - yp)))

        return yt, yp, mse1, mse, tae

    @staticmethod
    def _calc_tae(img_err):
        return torch.max(torch.max(img_err, dim=3).values,
                         dim=2).values.tolist()

    @staticmethod
    def _calc_mse(img_err):
        return torch.mean(img_err ** 2, dim=[2, 3]).tolist()

    @staticmethod
    def _is_var_serializable(var) -> bool:
        """
        check if attribute name is serializable. Unfortunately, there is no
        build-in method for this, hence this
        function.
        """
        try:
            json.dumps(var)
            return True
        except (TypeError, OverflowError):
            return False

    def _to_dict(self):
        dictionary = {'serial': {}, 'par2d': {}, 'par3d': {}, 'state_dict': {},
                      'non_serializable': {}}
        for key, val in vars(self).items():
            if self._is_var_serializable(val):
                dictionary['serial'][key] = val
            elif isinstance(val, PerformanceParameter2D):
                dictionary['par2d'][key] = val.to_dict()
            elif isinstance(val, PerformanceParameter3D):
                dictionary['par3d'][key] = val.to_dict()
            elif hasattr(val, 'state_dict') and hasattr(val,
                                                        'load_state_dict'):
                dictionary['state_dict'][key] = val
            else:
                dictionary['non_serializable'][key] = None
        return dictionary

    def _from_dict(self, dictionary):
        for key, val in dictionary['serial'].items():
            setattr(self, key, val)
        if 'par2d' in dictionary:
            for key, val in dictionary['par2d'].items():
                setattr(self, key, PerformanceParameter2D.load(val))
        if 'par3d' in dictionary:
            for key, val in dictionary['par3d'].items():
                setattr(self, key, PerformanceParameter3D.load(val))
        for key, val in dictionary['state_dict'].items():
            setattr(self, key, val)

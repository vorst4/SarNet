import abc
import json
import os
from pathlib import Path
from util.timer import Timer
from typing import Union, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader, Subset

import models
import settings
from .data import Dataset
from .log import Log
# from .performance import MsePerSubEpoch, PerformanceParameter3D
from util.performance import Performance
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
        self._timer = None
        self.performance: Performance = None
        self.saved_after_epoch = None
        self.n_batches = None
        self.idx_batch = None
        torch.manual_seed(self._torch_seed)

    def create(self,
               dataloader_training: List[torch.utils.data.DataLoader],
               dataloader_validation: List[torch.utils.data.DataLoader],
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

        # performance parameters
        #   get total number of validation samples
        n_validation_samples = 0
        for dl in self._dl_val:
            n_validation_samples += len(dl.dataset)
        #   create performance object
        self.performance = Performance(n_validation_samples)

        # log
        # todo: expand upon information that is being logged
        log.logprint(self.info())
        log.logprint('...done')
        exit()

        return self

    def info(self):
        # number of training and validation samples per subset:
        nt, nv = [], []
        for dlt, dlv in zip(self._dl_train, self._dl_val):
            nt.append(len(dlt.dataset))
            nv.append(len(dlv.dataset))

        # number of parameters
        n_par = sum(p.numel() for p in self._model.parameters())

        # create info String
        s = ''
        s += '  model: %s\n' % str(type(self._model))
        s += '  device: %s\n' % str(self.device)
        s += '  dataset:\n'
        s += '    number of subsets: %i\n' % settings.n_subsets
        s += '    size total: %i \n' % len(self._dl_train[0].dataset.dataset)
        s += '    size train: %i (%.0f%%) [%s]\n' % \
             (sum(nt), settings.train_pct, ','.join(map(str, nt)))
        s += '    size valid: %i (%.0f%%) [%s]\n' % \
             (sum(nv), 100 - settings.train_pct, ','.join(map(str, nv)))
        s += '  number of model parameters: %s\n' % '{:,}'.format(n_par)
        s += '  memory usage\n' + self._log.memory_usage(' ' * 4)

        return s

    def gpu_memory(self):
        s = ''
        s += ''

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

    # @classmethod
    # def get_default_dataloaders(
    #         cls,
    #         path_dataset: Union[str, Path],
    #         batch_size: int
    # ) -> Tuple[DataLoader, DataLoader]:
    #     split_pct = 0.9
    #     dataset = Dataset(path_dataset)
    #     n_training = int(split_pct * len(dataset))
    #     dataset_training = Subset(dataset, indices=range(0, n_training))
    #     dataset_validation = Subset(dataset,
    #                                 indices=range(n_training, len(dataset)))
    #     dataloader_training = DataLoader(dataset_training, batch_size,
    #                                      shuffle=True)
    #     dataloader_validation = DataLoader(dataset_validation, batch_size,
    #                                        shuffle=False)
    #     return dataloader_training, dataloader_validation

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
        self.performance.mse_train.allocate(epoch_stop * settings.n_subsets)
        self.performance.mse_valid.allocate(epoch_stop * settings.n_subsets)
        self.performance.loss_valid.allocate(epoch_stop * settings.n_subsets)

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

        timer = Timer(self._log)
        self._timer = timer

        # initialize progress display/saver
        # res_img_out = self._dl_train[0].dataset.shapes[Dataset.KEY.OUTPUT][1]
        res_img_out = 64  # todo: make this dynamic
        timer.start()
        progress = Progress(progress_settings, self, self._log,
                            res_img_out)
        timer.stop('created progress obj')

        # load previous model state if needed
        timer.start()
        if progress.load_design:
            if progress.path_design.exists():
                self.load(progress.path_design,
                          progress.path_design_backup[0],
                          self._dl_train,
                          self._dl_val,
                          self._log)
            timer.stop('loaded design')

        # reset starting epoch
        self._epoch_start = self._epoch_current

        # move model to gpu/cpu
        timer.start()
        self._model = self._model.to(device=self.device)
        timer.stop('moved model to gpu')

        # start timer
        self._timer.start()

        # loop over epochs
        for epoch in range(int(self._epoch_start), self._epoch_stop):

            # loop over training and validation subsets
            for idx, (dl_t, dl_v) in enumerate(zip(self._dl_train,
                                                   self._dl_val)):
                # if the model was loaded, skip until current epoch has been
                # reached
                ec = epoch + idx / len(self._dl_train)
                if ec < self._epoch_current:
                    continue
                self._epoch_current = epoch + idx / len(self._dl_train)

                # memory before training
                self._log.logprint('  memory usage (cpu): ' +
                                   self._log.cpu_memory_usage())
                if torch.cuda.is_available():
                    self._log.logprint('  memory usage (gpu): \n' +
                                       torch.cuda.memory_summary())

                # train
                timer.start()
                self.train_subset(dl_t)
                timer.stop('finished training')

                # memory after train subset
                self._log.logprint('  memory usage (cpu): ' +
                                   self._log.cpu_memory_usage())
                if torch.cuda.is_available():
                    self._log.logprint('  memory usage (gpu): \n' +
                                       torch.cuda.memory_summary())

                # evaluate
                timer.start()
                img_true, img_pred = self.evaluate_subset(dl_v)
                timer.stop('finished validating')

                # memory after validation subset
                self._log.logprint('  memory usage (cpu): ' +
                                   self._log.cpu_memory_usage())
                if torch.cuda.is_available():
                    self._log.logprint('  memory usage (gpu): \n' +
                                       torch.cuda.memory_summary())

                # save progress
                progress(img_true, img_pred)

            # memory after 1 epoch
            self._log.logprint('  memory usage (cpu): ' +
                               self._log.cpu_memory_usage())
            if torch.cuda.is_available():
                self._log.logprint('  memory usage (gpu): \n' +
                                   torch.cuda.memory_summary())

            # # evaluate
            # img_true, img_pred, mse_valid, mse_p, tae_p = self._evaluate()

            # log progress
            # self._train_mse_per_epoch.append(self._epoch_current, mse_train)
            # self._valid_mse_per_epoch.append(self._epoch_current, mse_valid)
            # self._valid_mse_per_phase_per_epoch.append(
            #     self._epoch_current,
            #     mse_p['phase'],
            #     mse_p['mse']
            # )
            # self._valid_tae_per_phase_per_epoch.append(
            #     self._epoch_current,
            #     tae_p['phase'],
            #     tae_p['tae']
            # )

            # show/save progress
            #   img_true and img_pred are the true and predicted images of
            #   the last batch, these are used for the
            #   preview
            # self.saved_after_epoch = True
            # progress(img_true, img_pred)

            # # update learning rate
            # if lr_sched is not None:
            #     lr_sched.step()

        # epoch current isn't updated after the last iteration, so this has
        # to be done manually
        self._epoch_current += 1

        # move model back to cpu
        self._model = self._model.cpu()

    def train_subset(self, dataloader: DataLoader):
        """
        Train on the given training subset
        """

        self.n_batches = len(dataloader)
        mse = 0.0

        timer = Timer(self._log)

        timer_ = Timer(self._log)
        timer_.start()

        # loop over batches
        for self.idx_batch, data in enumerate(dataloader):
            timer.start()

            timer_.stop('\tloaded batch')

            # make sure model is in train mode
            timer_.start()
            self._model.train()
            timer_.stop('\tset model to train mode')

            # abbreviate variables and move them to <device>
            timer_.start()
            kwargs = {'device': self.device, 'dtype': self.dtype}
            yt = data[Dataset.KEY.OUTPUT].to(**kwargs)
            x_img = data[Dataset.KEY.INPUT_IMG].to(**kwargs)
            x_meta = data[Dataset.KEY.INPUT_META].to(**kwargs)
            timer_.stop('\tabbreviated variables and moved them to device')

            # calculate predicted outcome and loss
            timer_.start()
            yp = self._model(x_img, x_meta)
            timer_.stop('\tcalculated prediction')

            timer_.start()
            loss = self._loss_function(yp, yt)
            timer_.stop('\tcalculated loss')

            # calculate gradient (a.k.a. backward propagation)
            timer_.start()
            loss.backward()
            timer_.stop('\tcalculated backward propagation')

            # update model parameters using the gradient
            timer_.start()
            self._optimizer.step()
            timer_.stop('\ttook optimizer step')

            # calculate mse
            timer_.start()
            mse += loss.cpu().detach().item() / self.n_batches
            timer_.stop('\tcalculated mse')

            timer.stop('trained for 1 batch')

        # update performance parameter
        timer.start()
        self.performance.mse_train.append(epoch=self._epoch_current,
                                          mse_train=mse)
        timer.stop('updated performance parameter')

    def evaluate_subset(self, dataloader: DataLoader):
        """
        evaluate model on validation dataset
        """

        timer = Timer(self._log).start()
        timer_ = Timer(self._log)

        # set model to evaluation mode
        self._model.eval()
        timer.stop('model set to evaluation mode')

        n_batches = len(dataloader)
        mse = 0.0
        # tae = {'phase': [], 'tae': []}
        # mse = {'phase': [], 'mse': []}

        timer.start()
        timer_.start()
        # gradient does not have to be calculated during evaluation
        with torch.no_grad():
            # loop over the evaluation dataset
            for idx_batch, data in enumerate(dataloader):
                timer_.stop('loaded validation batch')

                timer_.start()
                # abbreviate variables and move to <device>
                kwargs = {'device': self.device, 'dtype': self.dtype}
                yt = data[Dataset.KEY.OUTPUT].to(**kwargs)
                x_img = data[Dataset.KEY.INPUT_IMG].to(**kwargs)
                x_meta = data[Dataset.KEY.INPUT_META].to(**kwargs)
                timer_.stop('abbreviated variables and moved to device')

                # calculate predicted outcome and loss
                timer_.start()
                yp = self._model(x_img, x_meta)
                timer_.stop('calculated prediction')

                timer_.start()
                loss = self._loss_function(yp, yt)
                timer_.stop('calculated loss')

                # update performance parameter
                timer_.start()
                self.performance.loss_valid.append(
                    epoch=self._epoch_current,
                    loss=loss,
                    sample_idx=data[Dataset.KEY.INDEX]
                )
                timer_.stop('added performance parameter (loss_valid)')

                # calculate mse
                timer_.start()
                mse += loss.cpu().detach().item() / n_batches
                timer_.stop('calculated mse')

                timer.stop('calculated 1 validation batch')

        # update performance parameter
        timer.start()
        self.performance.mse_valid.append(epoch=self._epoch_current,
                                          mse_train=mse)
        timer.stop('updated performance parameter (mse_valid)')

        return yt, yp

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
        dictionary = {'serial': {},
                      'performance': self.performance.to_dict(),
                      'state_dict': {},
                      'non_serializable': {}}
        for key, val in vars(self).items():
            if self._is_var_serializable(val):
                dictionary['serial'][key] = val
            elif hasattr(val, 'state_dict') and hasattr(val,
                                                        'load_state_dict'):
                dictionary['state_dict'][key] = val
            else:
                dictionary['non_serializable'][key] = None
        return dictionary

    def _from_dict(self, dictionary):
        for key, val in dictionary['serial'].items():
            setattr(self, key, val)
        self.performance = Performance.load(dictionary['performance'])
        for key, val in dictionary['state_dict'].items():
            setattr(self, key, val)

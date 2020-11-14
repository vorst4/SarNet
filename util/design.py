import abc
import json
import os
from pathlib import Path
from util.timer import Timer
from typing import Union, List, Tuple, Optional

import torch
from torch import exp
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader

import models
import settings
from .data import Dataset, _Subset
from .log import Log
from util.performance import Performance
from .progress import Progress


class MAPE5i:
    def __init__(self, img_res):
        self.mae_loss = nn.L1Loss(reduction='none')
        self.r5 = int(0.05 * img_res ** 2)
        self.scalar = 100 / self.r5

    def _top5(self, n_batch, batch_true):
        return torch.argsort(batch_true.view(n_batch, -1), dim=1)[:self.r5]

    def __call__(self, batch_pred, batch_true) -> torch.tensor:
        # batch size
        n = batch_true.shape[0]

        # flatten tensors, and make sure they are on cpu memory
        yt = batch_true.view(n, -1).cpu().detach()
        yp = batch_pred.view(n, -1).cpu().detach()

        # obtain ids/pixel-locations of top 5% largest yt values
        ids = torch.argsort(yt, dim=1, descending=True)[:, :self.r5]

        # calculate mape5i
        # todo: find a more efficient way than a for loop
        mape5i = torch.empty(n)
        for i, j in enumerate(ids):
            a = self.scalar * torch.sum(torch.abs(
                (yt[i, j] - yp[i, j]) / yt[i, j]
            ))
            # todo: find a solution to below
            mape5i[i] = a if not torch.isinf(a) else 100

        return mape5i


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
        self._dls_train: List[_Subset] = []
        self._dls_valid: List[_Subset] = []
        self._model: Optional[nn.Module()] = None
        self._loss_function = None
        self._optimizer = None
        self._lr_sched = None
        self._log: Optional[Log] = None
        self._epoch_start: int = -1
        self.epoch_current: float = -1.
        self._epoch_stop: int = -1
        self._timer: Optional[Timer] = None
        self.performance: Optional[Performance] = None
        self._mape5i: Optional[MAPE5i] = None
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
        self._dls_train = dataloader_training
        self._dls_valid = dataloader_validation
        self._model = model
        self._loss_function = loss_function
        self._optimizer = optimizer
        self._lr_sched = lr_scheduler
        self._log = log
        self._epoch_start = 0
        self.epoch_current = 0
        self._epoch_stop = epochs

        # performance parameters
        #   get total number of validation samples
        n_validation_samples = 0
        for dl in self._dls_valid:
            n_validation_samples += len(dl.dataset)
        #   create performance object
        self.performance = Performance(n_validation_samples)

        # log
        log.__call__(self.info())

        return self

    def info(self):
        # todo: show number of cpu cores, clock frequency, and cpu usage
        # number of training and validation samples per subset:
        nt, nv = [], []
        for dlt, dlv in zip(self._dls_train, self._dls_valid):
            nt.append(len(dlt.dataset))
            nv.append(len(dlv.dataset))

        # number of parameters
        n_par = sum(p.numel() for p in self._model.parameters() if
                    p.requires_grad)

        # abbreviate format_number function & train_pct
        f = self._log.format_num
        train_pct = settings.dataset.train_pct

        # create info String
        s = ''
        s += '  model: %s\n' % str(type(self._model))
        s += '  device: %s\n' % str(self.device)
        s += '  dataset:\n'
        s += '    number of subsets: %s\n' % f(settings.dataset.n_subsets)
        s += '    size total: %s \n' % \
             f(len(self._dls_train[0].dataset.dataset))
        s += '    size train: %s (%.0f%%), subset_min:%s subset_max:%s]\n' % \
             (f(sum(nt)), train_pct, f(min(nt)), f(max(nt)))
        s += '    size valid: %s (%.0f%%), subset_min:%s subset_max:%s]\n' % \
             (f(sum(nv)), 100 - train_pct, f(min(nt)),
              f(max(nt)))
        s += '  batch size: %i\n' % settings.batch_size
        s += '  number of model parameters: %s\n' % f(n_par)
        s += '  memory usage\n' + self._log.memory_usage(' ' * 4)
        return s

    def get_epoch_stop(self) -> int:
        """
        Get epoch at which the training stops
        """
        return self._epoch_stop

    def get_epoch_current(self) -> float:
        """
        Get epoch at which the training stops
        """
        return self.epoch_current

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

    # todo: obsolete ?
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
        if epoch_stop <= self.epoch_current:
            epoch_stop = self.epoch_current

        # set epoch stop
        self._epoch_stop = epoch_stop

        # allocate space for the performance parameters
        self.performance.loss_train.allocate(
            epoch_stop * settings.dataset.n_subsets
        )
        self.performance.loss_valid.allocate(
            epoch_stop * settings.dataset.n_subsets
        )
        self.performance.loss_valid.allocate(
            epoch_stop * settings.dataset.n_subsets
        )

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
             dl_train: List[DataLoader],
             dl_val: List[DataLoader],
             log: Log):
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
            self._from_dict(torch.load(
                file, map_location=torch.device('cpu')
            ))
        except (RuntimeError, EOFError):
            # if failed, try loading the backup
            self._log.__call__('WARNING: loading file %s failed, trying '
                               'to load backup' % file)
            if isinstance(backup, str):
                backup = Path(backup)
            if not backup.exists():
                raise Exception('ERROR: could not load file %s and backup %s'
                                ' does not exist' % (file, backup))

            self._from_dict(torch.load(backup,
                                       map_location=torch.device('cpu')))
            self._log.__call__('INFO: successfully loaded backup')

        # set non serializable attributes
        self._log = log
        self._dls_train = dl_train
        self._dls_valid = dl_val

        return self

    def train(self, progress_settings: Progress.Settings):
        """
        Train and evaluate the model/design, the <progress> during training
        is displayed/saved according to its
        settings.
        """

        timer = Timer(self._log)
        self._timer = timer

        # get the resolution of the output images (assuming they are square)
        sample = self._dls_train[0].dataset.__getitem__(0)
        img_res = sample[Dataset.KEY.OUTPUT].shape[1]
        del sample

        timer.start()
        progress = Progress(progress_settings, self, self._log, img_res)
        timer.stop('created progress obj')

        self._mape5i = MAPE5i(img_res)

        # load previous model state if needed
        timer.start()
        if progress.load_design:
            if progress.path_design.exists():
                self.load(progress.path_design,
                          progress.path_design_backup[0],
                          self._dls_train,
                          self._dls_valid,
                          self._log)
            timer.stop('loaded design')

        # reset starting epoch
        self._epoch_start = self.epoch_current

        # move model to gpu/cpu
        timer.start()
        self._model = self._model.to(device=self.device)
        timer.stop('moved model to gpu')

        # start timer
        self._timer.start()

        # print memory before starting to train
        self._log.__call__('memory usage before training: \n'
                           + self._log.memory_usage(' ' * 2))

        # loop over epochs
        for epoch in range(int(self._epoch_start), self._epoch_stop):

            # loop over training and validation subsets
            for idx, (dl_t, dl_v) in enumerate(zip(self._dls_train,
                                                   self._dls_valid)):
                # continue till current epoch is reached
                if (epoch + idx / len(self._dls_train)) < self.epoch_current:
                    continue

                # train
                timer.start()
                self.train_subset(dl_t, progress)
                timer.stop('finished training')

                # evaluate
                timer.start()
                self.evaluate_subset(dl_v, progress)
                timer.stop('finished validating')

                # save progress
                timer.start()
                self.epoch_current = epoch + (idx + 1) / len(self._dls_train)
                progress()
                timer.stop('saved progress')

            # shuffle dataset, the DataLoader will only shuffle the subsets,
            #   not the train/valid dataset as a whole
            self._dls_train[0].dataset.dataset.shuffle()

            # print memory usage after each full epoch
            self._log.__call__('memory usage:\n' +
                               self._log.memory_usage(' ' * 2))

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

        # update learning rate
        if self._lr_sched is not None:
            self._lr_sched.step()

        # epoch current isn't updated after the last iteration, so this has
        # to be done manually
        self.epoch_current += 1

        # move model back to cpu
        self._model = self._model.cpu()

    def train_subset(self, dataloader: DataLoader, progress: Progress):
        """
        Train on the given training subset
        """

        n_batches = len(dataloader)
        loss_batch = 0.0
        mse = 0.0
        mape5 = 0.0

        timer = Timer(self._log)

        timer_ = Timer(self._log)
        timer_.start()

        # loop over batches
        for self.idx_batch, data in enumerate(dataloader):
            # timer.start()

            # timer_.stop('\tloaded batch')

            # make sure model is in train mode
            # timer_.start()
            self._model.train()
            # timer_.stop('\tset model to train mode')

            # abbreviate variables and move them to <device>
            timer_.start()
            kwargs = {'device': self.device, 'dtype': self.dtype}
            yt = data[Dataset.KEY.OUTPUT].to(**kwargs)
            x_img = data[Dataset.KEY.INPUT_IMGS].to(**kwargs)
            x_meta = data[Dataset.KEY.INPUT_META].to(**kwargs)
            # timer_.stop('\tabbreviated variables and moved them to device')

            # calculate predicted outcome and loss
            # timer_.start()
            yp = self._model(x_img, x_meta)
            # timer_.stop('\tcalculated prediction')

            # timer_.start()
            yp, loss = self._loss(yp, yt)
            # timer_.stop('\tcalculated loss')

            # calculate gradient (a.k.a. backward propagation)
            # timer_.start()
            loss.backward()
            # timer_.stop('\tcalculated backward propagation')

            # update model parameters using the gradient
            # timer_.start()
            self._optimizer.step()
            # timer_.stop('\ttook optimizer step')

            # add imgs to preview buffer
            # timer_.start()
            progress.add_imgs_to_preview_buffer(yp, yt)
            # timer_.stop('\tadded imgs to preview buffer')

            # calculate mse
            # timer_.start()
            loss_batch += loss.cpu().detach().item() / n_batches
            mse += functional.mse_loss(yp, yt) / n_batches
            mape5i = self._mape5i(yp, yt)
            mape5 += torch.sum(mape5i) / (mape5i.shape[0] * n_batches)
            # timer_.stop('\tcalculated mse')

            # timer.stop('trained for 1 batch')

        # update performance parameter
        timer.start()
        self.performance.loss_train.append(epoch=self.epoch_current,
                                           loss_train=loss_batch)
        self.performance.mse_train.append(epoch=self.epoch_current,
                                          mse_train=mse)
        self.performance.mape5_train.append(epoch=self.epoch_current,
                                            mape5_train=mape5)
        timer.stop('updated performance parameter')

    def evaluate_subset(self, dataloader: DataLoader, progress: Progress):
        """
        evaluate model on validation dataset. the progress object is needed
        to write the evaluated images to its preview-buffer
        """

        timer = Timer(self._log).start()
        timer_ = Timer(self._log)

        # set model to evaluation mode
        self._model.eval()
        timer.stop('model set to evaluation mode')

        n_batch = len(dataloader)
        loss_batch = 0.0
        mse = 0.0
        mape5 = 0.0
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
                x_img = data[Dataset.KEY.INPUT_IMGS].to(**kwargs)
                x_meta = data[Dataset.KEY.INPUT_META].to(**kwargs)
                timer_.stop('abbreviated variables and moved to device')

                # calculate predicted outcome and loss
                timer_.start()
                yp = self._model(x_img, x_meta)
                timer_.stop('calculated prediction')

                timer_.start()
                yp, loss = self._loss(yp, yt)
                timer_.stop('calculated loss')

                # calculate/append performance parameter (mse per sample)
                timer_.start()
                mape5i = self._mape5i(yp, yt)
                self.performance.mape5i.append(
                    epoch=self.epoch_current,
                    mape5i=mape5i.tolist(),
                    sample_idx=data[Dataset.KEY.INDEX].tolist()
                )
                timer_.stop('added performance parameter (loss_valid)')

                # add predicted and true images to progress-preview
                progress.add_imgs_to_preview_buffer(yp, yt)

                # calculate mse
                timer_.start()
                loss_batch += loss.cpu().detach().item() / n_batch
                mse += functional.mse_loss(yp, yt) / n_batch
                mape5 += torch.sum(mape5i) / (mape5i.shape[0] * n_batch)
                timer_.stop('calculated mse')
                timer.stop('calculated 1 validation batch')

        # update performance parameter
        timer.start()
        self.performance.loss_valid.append(epoch=self.epoch_current,
                                           loss_valid=loss_batch)
        self.performance.mse_valid.append(epoch=self.epoch_current,
                                          mse_valid=mse)
        self.performance.mape5_valid.append(epoch=self.epoch_current,
                                            mape5_valid=mape5)
        timer.stop('updated performance parameter (mse_valid)')

    def _loss(self, yp, yt):
        # If not variational auto-encoder
        if 'V' not in str(type(self._model)):
            return yp, self._loss_function(yp, yt)

        # If variational auto-encoder
        else:
            # unpack dict
            yp, logvar, mu = yp['y'], yp['var'], yp['mu']
            # calculate loss
            kld = -0.5 * torch.sum(1 + logvar - mu ** 2 - exp(logvar))
            # bce = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
            return yp, self._loss_function(yp, yt) + kld

    @classmethod
    def parameters_available_designs(cls, indent=''):
        modelnames = cls.get_available_modelnames()
        s = ''
        for name in modelnames:
            model = cls.get_model_from_name(name)

            n_par = sum(p.numel() for p in model.parameters() if
                        p.requires_grad)

            s += '%s%s: %s\n' % (indent, name, Log.format_num(n_par))

        return s[:-1]

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

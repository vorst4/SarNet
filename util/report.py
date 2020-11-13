import matplotlib.pyplot as plt
from pathlib import Path
from util.design import Design
from util.data import Dataset
import settings
from util.log import Log
import torch
from zipfile import ZipFile
from util.data import IDX, KEY
from numpy import pi
import numpy as np

root_models = 'C:/Users/Dennis/Documents/desktop_resnet_output'
modelpaths = [str(p) for p in Path(root_models).glob('*/design.pt')]
dataset_path = settings.dataset.file
n_samples_dataset = settings.dataset.max_samples
interpolation_scale = 0.25


class Report:
    def __init__(self):
        # dummy dataset & loaders
        ds = Dataset(settings.dataset, )
        dlt, dlv = ds.dataloaders_train, ds.dataloaders_valid

        # dummy log
        settings.log.directory = ''
        settings.log.save_log = False
        log = Log(settings.log, job_id=0, server_id=0, n_cpus=0, n_gpus=0)

        # dummy loss function
        lf = torch.nn.MSELoss()

        performances = []
        modelnames = []
        for path in modelpaths:
            modelname = path.split('\\')[-2].split('_')[1]
            modelnames.append(modelname)

            model = Design.get_model_from_name(modelname)

            design = Design().create(
                dlt, dlv, model, lf, None, None, 0, log
            )

            design = design.load(path, '', dlt, dlv, log)

            performances.append(design.performance)

        # ATTRIBUTES
        self.performances = performances
        self.modelnames = modelnames
        self.dataset = None  # only load dataset if needed, to speed things up
        self.slice_phases = slice(1, 24, 2)
        self.slice_amplitudes = slice(0, 24, 2)

        # self.loss_train()
        # self.loss_valid()
        # self.mse_train()
        # self.mse_valid()
        self.mape5_train()
        self.mape5_valid()
        #
        # self.mape5i_idx()
        # self.mape5i_phase()

        plt.show()
        exit()

    @staticmethod
    def average(x, y, ns):
        """
        average data [x,y] over every ns sample-points
        """
        n = len(x)
        xa, ya = [], []
        for idx in range(len(x)):
            ids = slice(idx * ns, (idx + 1) * ns, 1)
            try:
                xa.append(np.mean(x[ids]))
                ya.append(np.mean(y[ids]))
            except IndexError:
                break
        return xa, ya

    def _plot(self, attribute, save, show, n_average=None, log=True):
        plt.figure()
        for p, name in zip(self.performances, self.modelnames):
            obj = getattr(p, attribute)
            # print(obj.idx)
            # print(obj.x[:obj.idx])
            print(obj.y)
            x, y = obj.x[:obj.idx], obj.y[:obj.idx]
            if n_average is not None:
                x, y = self.average(x, y, n_average)
            if log:
                plt.semilogy(x, y, label=name)
            else:
                plt.plot(x, y, label=name)
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.legend()
        if save:
            # todo: save plot
            pass
        if show:
            plt.show()

    def loss_train(self, save: bool = False, show: bool = False, ):
        self._plot('loss_train', save, show)
        plt.title('Loss Training')

    def loss_valid(self, save: bool = False, show: bool = False):
        self._plot('loss_valid', save, show)
        plt.title('Loss Validation')

    def mse_train(self, save: bool = False, show: bool = False):
        self._plot('mse_train', save, show)
        plt.title('MSE Training')

    def mse_valid(self, save: bool = False, show: bool = False):
        self._plot('mse_valid', save, show)
        plt.title('MSE Validation')

    def mape5_train(self, save: bool = False, show: bool = False):
        self._plot('mape5_train', save, show, log=False)
        plt.title('MAPE5 Training')

    def mape5_valid(self, save: bool = False, show: bool = False):
        self._plot('mape5_valid', save, show, log=False)
        plt.title('MAPE5 Validation')
        pass

    def mape5i_idx(self, save: bool = False, show: bool = False):
        plt.figure()
        for p, name in zip(self.performances, self.modelnames):
            obj = p.mape5i
            last_epoch = obj.x[-1]

            # get indexes as a slice that belong to last epoch
            slice_epoch = slice(obj.x.index(last_epoch), len(obj.x), 1)

            # get data
            mape5i = torch.tensor(obj.y[slice_epoch])
            ids = torch.tensor(obj.z[slice_epoch])

            # sort it according to ids
            arg_sort = torch.argsort(ids)
            ids_sorted = ids[arg_sort]
            mape5i_sorted = mape5i[arg_sort]

            plt.semilogy(ids_sorted, mape5i_sorted, label=name)
        plt.xlabel('ids')
        plt.ylabel('Loss')
        plt.legend()
        if save:
            # todo: save plot
            pass
        if show:
            plt.show()

    def mape5i_phase(self, save: bool = False, show: bool = False):
        self.load_dataset()

        plt.figure()
        for p, name in zip(self.performances, self.modelnames):
            obj = p.mape5i
            last_epoch = obj.x[-1]

            # get indexes as a slice that belong to last epoch
            slice_epoch = slice(obj.x.index(last_epoch), len(obj.x), 1)

            # get data
            mape5i = torch.tensor(obj.y[slice_epoch])
            ids = torch.tensor(obj.z[slice_epoch])

            # load phases
            phases = torch.empty((12, ids.shape[0]))
            for idx1, idx2 in enumerate(ids):
                p = self.dataset[KEY.INPUT_META][int(idx2), self.slice_phases]
                p = 360 / 255 * p.type(torch.float)
                phases[:, idx1] = p

            phases_mean = torch.mean(phases, dim=0)

            # sort it according to ids
            arg_sort = torch.argsort(ids)
            ids_sorted = ids[arg_sort]
            mape5i_sorted = mape5i[arg_sort]

            plt.semilogy(ids_sorted, mape5i_sorted, label=name)
        plt.xlabel('ids')
        plt.ylabel('Loss')
        plt.legend()
        if save:
            # todo: save plot
            pass
        if show:
            plt.show()

    def mape5i_amplitude(self, save: bool = False, show: bool = False):
        self.load_dataset()
        pass

    def load_dataset(self):

        # skip if already loaded
        if self.dataset is not None:
            return

        path = dataset_path
        # append memory for dataset
        dataset = {
            KEY.INDEX: torch.empty(n_samples_dataset, dtype=torch.int32),
            KEY.INPUT_META: torch.empty((n_samples_dataset,
                                         2 * settings.N_ANTENNAS),
                                        dtype=torch.uint8)
        }
        if '.zip' in path:
            with ZipFile(dataset_path, 'r') as zipfile:
                with zipfile.open('dataset.csv') as file:
                    # skip header
                    next(file)

                    # read every line
                    for idx, byte_array in enumerate(file.readlines()):

                        # break if max_lines is reached
                        if idx >= n_samples_dataset:
                            break

                        # convert bytes to string and split string at delimiter
                        array = byte_array.decode('utf-8') \
                            .split(settings.dataset.csv_delimiter)

                        # add index
                        dataset[KEY.INDEX][idx] = int(array[IDX.INDEX])

                        # add input meta
                        dataset[KEY.INPUT_META][idx, :] = torch.tensor(
                            # convert to uint8 range
                            [255 * float(a) for a in array[IDX.INPUT_META]]
                        )

            # number of samples that were read
            n_samples = idx

            # reduce dataset if too much space was allocated
            if n_samples < n_samples_dataset:
                dataset[KEY.INDEX] = dataset[KEY.INDEX][:n_samples]
                dataset[KEY.INPUT_META] = dataset[KEY.INPUT_META][:n_samples,
                                          :]
        else:
            raise NotImplemented()

        self.dataset = dataset

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from .design import Design
from .progress import Progress


class TuneAdam:
    """
    Tune the Adam optimizer for one or multiple models. Only the learning
    rate and beta1 can be tuned.
    """

    def __init__(self, directory, dl_train, dl_val, log):
        self._dir = directory
        self._dlt = dl_train
        self._dlv = dl_val
        self._log = log
        self._lf = torch.nn.MSELoss()

    def _train(self, modelname, lr, beta1, epochs):

        filename = modelname + ' lr=' + str(lr) + ' b1=' + str(beta1) + '.pt'

        # return if model already exists
        path = Path(self._dir).joinpath(filename)
        if path.exists():
            return
        self._log.logprint(str(path))

        # create and train design
        progress_settings = Progress.Settings(
            save_design=True,
            load_design=True,
            path=self._dir,
            filename_design=filename
        )
        model = Design.get_model_from_name(modelname)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=(beta1, 0.999)
        )
        design = Design().create(self._dlt, self._dlv, model, self._lf,
                                 optimizer, None, epochs, self._log)
        design.train(progress_settings)
        design.save(path)

    def tune_learning_rate(self, epochs, modelnames, learning_rates, beta1):
        """
        Train the models with modelnames multiple times, each time for a
        certain amount of epochs and different learning rate.
        """

        if isinstance(modelnames, list):
            for modelname in modelnames:
                b1 = beta1 if len(beta1) == 1 else beta1[modelname]
                for lr in learning_rates:
                    self._train(modelname, lr, b1, epochs)
        else:
            for lr in learning_rates:
                self._train(modelnames, lr, beta1, epochs)

    def tune_beta1(self, epochs, modelnames, beta1s, learning_rate):
        """
        Train the models with modelnames multiple times, each time for a
        certain amount of epochs and different beta1.
        """
        for modelname in modelnames:
            lr = learning_rate if len(learning_rate) == 1 else learning_rate[
                modelname]
            for beta1 in beta1s:
                self._train(modelname, lr, beta1, epochs)

    def _load_data(self):

        # path to each model inside the given directory
        paths = list(Path(self._dir).glob('*.pt'))

        # modelnames
        data = {'models': {}, 'y_lim': [None, None]}
        for path in paths:
            split = path.stem.split(' ', 1)
            modelname = split[0]
            parameters = split[1]  # learning rate and beta1

            # load training and validation data from model
            design = Design().load(path, self._dlt, self._dlv, self._log)
            dt = design.get_data_train()
            dv = design.get_data_val()

            y_min = min(min(dt[1]), min(dv[1]))
            y_max = max(max(dt[1]), max(dv[1]))
            if data['y_lim'][0] is None or data['y_lim'][0] > y_min:
                data['y_lim'][0] = y_min
            if data['y_lim'][1] is None or data['y_lim'][1] < y_max:
                data['y_lim'][1] = y_max

            # add modelname to data if not present yet
            if modelname not in data['models']:
                data['models'][modelname] = {}

            # add loaded data to dictionary
            data['models'][modelname][parameters] = (dt, dv)

        return data

    def plot(self):
        """
        Load the 'explored models', models trained for <epochs> with
        different learning rates. Plot the training and  validation loss for
        each different type of model.
        """

        data = self._load_data()
        data['y_lim'][1] = 10

        subplot_name = (' train', ' val')
        for modelname, model_data in data['models'].items():
            # fig, ax = plt.subplots(num=modelname, figsize=(18, 9),
            # nrows=1, ncols=2)
            c = 1.5
            fig, ax = plt.subplots(num=modelname, figsize=(c * 8, c * 3),
                                   nrows=1, ncols=2)
            fig.canvas.set_window_title(modelname)
            for idx in (0, 1):
                for parameters, xy in model_data.items():
                    lr = float(parameters.split(' ')[0].split('=')[1])
                    if lr > 1e-3:
                        continue
                    # ax[idx].semilogy(xy[idx][0], xy[idx][1], '.',
                    # markersize=1, label=parameters)
                    ax[idx].semilogy(xy[idx][0], xy[idx][1], label=parameters)

                leg = ax[idx].legend()
                for idx2 in range(len(leg.legendHandles)):
                    leg.legendHandles[idx2]._legmarker.set_markersize(20)

                # leg.legendHandles[1]._legmarker.set_markersize(6)
                ax[idx].set_title(modelname + subplot_name[idx])
                ax[idx].set_ylabel('loss')
                ax[idx].set_xlabel('epoch')
                ax[idx].set_ylim(data['y_lim'])

        # show plot
        plt.show()

    def save_plot(self,
                  dst='C:/Users/Dennis/Google Drive/thesis/progress '
                      'update/images 6/tuning_results.png'):

        c = 1.5
        dpi = 100
        w = c * 1920 / dpi
        h = c * 550 / dpi

        data = self._load_data()

        fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(w, h),
                                num='tuning_results', dpi=dpi)
        fig.canvas.set_window_title('tuning_results')
        plt.tight_layout(rect=[0.009, 0.007, 1, 0.999])

        data = self._load_data()
        data['y_lim'][1] = 10

        subplot_name = (' train', ' val')
        ctr = -1
        for modelname, model_data in data['models'].items():
            ctr += 1
            for idx in (0, 1):
                for parameters, xy in model_data.items():
                    lr = float(parameters.split(' ')[0].split('=')[1])
                    if lr > 1e-3:
                        continue
                    # ax[idx].semilogy(xy[idx][0], xy[idx][1], '.',
                    # markersize=1, label=parameters)
                    axs[idx][ctr].semilogy(xy[idx][0], xy[idx][1],
                                           label=parameters.split(' ')[0])

                leg = axs[idx][ctr].legend()
                for idx2 in range(len(leg.legendHandles)):
                    leg.legendHandles[idx2]._legmarker.set_markersize(20)

                axs[idx][ctr].set_title(modelname + subplot_name[idx])
                axs[idx][ctr].set_ylim(data['y_lim'])

        for idx in (0, 1):
            axs[idx][0].set_ylabel('loss')
        for idx in range(6):
            axs[1][idx].set_xlabel('epoch')

        # save and show
        plt.savefig(dst)
        from PIL import Image
        Image.open(dst).show()

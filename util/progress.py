from pathlib import Path
from typing import List
from typing import Union, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .log import Log

if TYPE_CHECKING:
    from .design import Design


class Progress:
    """
    This class displays, loads and/or saves the progress made during
    training a PyTorch model. The progress is
    displayed by printing the loss each epoch, as well as plotting the loss
    (lossplot) and plotting a
    preview, which is a side-by-side comparison of several true and
    predicted images.

    When calling the method 'train' of the class 'Design',
    the Progress.Settings must be passed.
    """

    IDX_LOSSPLOT = 0
    IDX_PREVIEW = 1

    @staticmethod
    class Settings:
        def __init__(self,
                     print_: bool = True,
                     lossplot: bool = False,
                     preview: bool = False,
                     load_design: bool = False,
                     save_design: bool = False,
                     save_lossplot: bool = False,
                     save_preview: bool = False,
                     save_interval: float = 10,
                     path: Union[str, Path] = None,
                     filename_design: str = 'design.pt',
                     n_backups: int = 3,
                     filename_best_design: str = 'best_design.pt',
                     filename_lossplot: str = 'lossplot.png',
                     subdir_preview: str = 'preview',
                     filename_prefix_preview: str = 'preview e',
                     filename_postfix_preview: str = '.png',
                     preview_vstack: int = 4,
                     preview_hstack: int = 5,
                     preview_padding: int = 1,
                     figure_size: (float, float) = (8.0, 6.0)
                     ):
            """
            Settings of the progress class, which displays, loads, and/or
            saves the progress made during training.
            :param print: (bool) to print progress during training or not
            :param lossplot: (booL) to plot the training and validation loss
                during training or not
            :param preview: (bool) to display preview or not, the preview
                contains several true and predicted validation images.
            :param save_interval (int) interval in minutes after which the
                model is saved
            :param load_design: (bool) to load or ignore the previous
                design: at "path + filename_design"
            :param save_design: (bool) to save design at "path +
                filename_design" or not
            :param save_lossplot: (bool) to save the loss plot at "path +
                filename_lossplot" or not.
            :param save_preview: (bool) to save the preview or not,
                the preview contains several true and predicted validation
                images. A seperate preview is saved during each epoch
            :param path: the base path to which everything is saved
            :param filename_design: filename of the design, if saved
            :param filename_lossplot: filename of the lossplot, if saved
            :param subdir_preview: the subdirectory in which the preview is
                saved, unlike the design and lossplot, a separate preview is
                saved each epoch
            :param filename_prefix_preview:
            :param filename_postfix_preview:
            :param preview_vstack: the number of paired (trained &
                predicted) images stacked vertically in the preview
            :param preview_hstack: the number of paired (trained &
                predicted) images stacked horizontally in the preview
            :param preview_padding: the amount of padding in pixels around
                each image in the preview
            :param figure_size: the size of the lossplot and preview that
                are generated
            """
            self.print = print_
            self.lossplot = lossplot
            self.preview = preview
            self.load_design = load_design
            self.save_design = save_design
            self.save_lossplot = save_lossplot
            self.save_preview = save_preview
            self.save_interval = save_interval
            self.path = path
            self.filename_design = filename_design
            self.filename_best_design = filename_best_design
            self.filename_lossplot = filename_lossplot
            self.n_backups = n_backups
            self.subdir_preview = subdir_preview
            self.filename_prefix_preview = filename_prefix_preview
            self.filename_postfix_preview = filename_postfix_preview
            self.preview_vstack = preview_vstack
            self.preview_hstack = preview_hstack
            self.preview_padding = preview_padding
            self.figure_size = figure_size

        def __str__(self):
            s = '-' * 69 + '\n'
            s += repr(self) + '\n'
            for att in dir(self):
                if att[0] != '_':
                    s += '  ' + att + ' = ' + repr(getattr(self, att)) + '\n'
            s = '-' * 69 + '\n'
            return s

    def __init__(self,
                 settings: Settings,
                 design: 'Design',
                 log: Log,
                 img_res: int):

        # --- ATTRIBUTES --- #
        # misc
        self.settings = settings
        self.log = log
        self.design = design
        self.lossplot_min, self.lossplot_max = 0.1, 0.1
        self.load_design = self.settings.load_design
        # paths
        paths = self._get_paths(settings)
        self.path_design: Path = paths['design']
        self.path_design_backup: List[Path] = paths['design_bak']
        self.path_best_design: Path = paths['best']
        self.path_best_design_backup: List[Path] = paths['best_bak']
        self.path_lossplot: Path = paths['lossplot']
        self.dir_preview: Path = paths['preview']

        # create dir_preview if needed
        if self.settings.save_preview and not self.dir_preview.exists():
            self.dir_preview.mkdir()

        # set interactive mode of matplotlib on, if needed
        if self.settings.lossplot or self.settings.preview:
            plt.ion()

        # --- LOSSPLOT --- #

        # create lossplot figure window
        if self.settings.lossplot or self.settings.save_lossplot:
            plt.figure(num=self.IDX_LOSSPLOT,
                       figsize=self.settings.figure_size)
            self.line_train, = plt.semilogy([], [], label='loss_train')
            self.line_val, = plt.semilogy([], [], label='loss_val')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            if self.settings.lossplot:
                plt.show()

        # --- PREVIEW --- #

        # abbreviate vars
        vstack = self.settings.preview_vstack
        hstack = self.settings.preview_hstack

        # define resolution of image and of padded image
        self.res = img_res
        self.res2 = self.res + 2 * self.settings.preview_padding
        self.n_imgs = hstack * vstack

        # pre-allocate space to speed up the script
        self.imgs = np.zeros((self.n_imgs, 2, self.res2, self.res2))
        self.img_ = np.zeros((2 * hstack, vstack * self.res2, self.res2))
        self.img_ = np.zeros((vstack * self.res2, 2 * hstack * self.res2))

        # create plot window
        if self.settings.preview:
            plt.figure(num=self.IDX_PREVIEW, figsize=self.settings.figure_size)
            plt.axis('off')
            plt.show()

    def __call__(self, yt_val, yp_val, mse_batch=None):

        # print progress, if desired
        if self.settings.print:
            self._print(mse_batch)

        # save design, if path is given
        if self.settings.save_design:
            self.design.save(self.path_design,
                             backups=self.path_design_backup)

            # if the last validation loss is the lowest, also save it as
            # 'best' design (if epoch is finished)
            if self.design.saved_after_epoch:
                _, lv = self.design.get_valid_mse_per_epoch()
                if lv[-1] == min(lv):
                    self.design.save(self.path_best_design,
                                     backups=self.path_best_design_backup)

        # lossplot
        if self.settings.lossplot or self.settings.save_lossplot and \
                self.design.saved_after_epoch:
            self._lossplot()

        # preview
        if self.settings.preview or self.settings.save_preview:
            self._preview(yt_val, yp_val)

    @staticmethod
    def _get_paths(settings: Settings) -> {}:

        # return nones if path isn't set
        if settings.path is None:
            return None, None, None

        # if path is string, convert it to pathlib.Path
        if isinstance(settings.path, str):
            settings.path = Path(settings.path)

        # verify that path exists
        if not settings.path.exists():
            raise ValueError(
                'Given progress.settings.path does not exist ' + str(
                    settings.path))

        # return paths
        paths_design_backups = []
        paths_best_design_backups = []
        for idx in range(1, settings.n_backups + 1):
            paths_design_backups.append(
                settings.path.joinpath(
                    settings.filename_design.replace('.', '_%i.' % idx)
                )
            )
            paths_best_design_backups.append(
                settings.path.joinpath(
                    settings.filename_best_design.replace('.', '_%i.' % idx)
                )
            )
        return {
            'design': settings.path.joinpath(settings.filename_design),
            'design_bak': paths_design_backups,
            'best': settings.path.joinpath(settings.filename_best_design),
            'best_bak': paths_best_design_backups,
            'lossplot': settings.path.joinpath(settings.filename_lossplot),
            'preview': settings.path.joinpath(settings.subdir_preview),
        }

    def _print(self, mse_batch):
        """
        Print progress
        """
        ec = self.design.get_epoch_current()
        es = self.design.get_epoch_stop()
        bi = self.design.idx_batch
        bt = self.design.n_batches
        if mse_batch is None:
            _, lt = self.design.get_train_mse_per_epoch()
            lt = lt[-1]
            _, lv = self.design.get_valid_mse_per_epoch()
            lv = lv[-1]
            self.log.logprint(
                'Epoch:%3.0f/%i, Batch:%i/%i, loss_train: %3.12f, loss_val: '
                '%3.12f' % (ec, es, bi, bt, lt, lv)
            )
        else:
            self.log.logprint(
                'Epoch:%3.0f/%i, Batch:% 3i/%i, mse_batch: %3.12f'
                % (ec-1, es, bi, bt, mse_batch)
            )

    def _lossplot(self):
        """
        update display and/or save lossplot
        """

        # update lossplot
        et, lt = self.design.get_train_mse_per_epoch()
        ev, lv = self.design.get_valid_mse_per_epoch()
        self.line_train.set_xdata(et)
        self.line_train.set_ydata(lt)
        self.line_val.set_xdata(ev)
        self.line_val.set_ydata(lv)

        # adjust axes
        fig = plt.figure(self.IDX_LOSSPLOT)
        y_last = (lt[-1], lv[-1])
        if self.lossplot_min > min(y_last):
            self.lossplot_min = min(y_last)
        if self.lossplot_max < max(y_last):
            self.lossplot_max = max(y_last)
        plt.xlim([0, et[-1] + 1])
        plt.ylim([0.9 * self.lossplot_min, 1.1 * self.lossplot_max])

        # update lossplot display
        if self.settings.lossplot:
            fig.canvas.draw()
            fig.canvas.flush_events()

        # save lossplot
        if self.settings.save_lossplot:
            fig.savefig(self.path_lossplot)

    def _preview(self, imgs_true, imgs_pred):
        """
        update display and/or save preview, preview being a side by side
        comparison of (multiple) true output images
        and predictions.
        """

        # make sure imgs has the right shape
        self.imgs = self.imgs.reshape((-1, 2, self.res2, self.res2))

        # extract n_imgs from yt_val and yp_val, and pad them, the padded
        # result is stored in <imgs>
        p = self.settings.preview_padding
        self.imgs[:, 0, p:-p, p:-p] = imgs_true.cpu().detach().numpy()[
                                      :self.n_imgs, 0, :, :]
        self.imgs[:, 1, p:-p, p:-p] = imgs_pred.cpu().detach().numpy()[
                                      :self.n_imgs, 0, :, :]

        # currently <imgs> is a matrix with images. The matrix has two
        # columns with the true images on the first
        # column and the predicted imgs on the second. Reshape this matrix
        # to a grid pattern of side-by-side comparison
        hstack, vstack = self.settings.preview_hstack, self.settings.preview_vstack
        self.imgs = self.imgs.reshape(vstack, 2 * hstack, self.res2, self.res2)

        # concatenate to create one big image
        self.img_ = np.concatenate(self.imgs, axis=1)
        self.img = np.concatenate(self.img_, axis=1)

        # clip to [0, 1]
        self.img = _clip(self.img, 0, 1)

        # show img
        if self.settings.preview:
            plt.figure(self.IDX_PREVIEW)
            plt.imshow(self.img, cmap='hot')

        # save img
        if self.settings.save_preview:
            path = self.dir_preview.joinpath(
                self.settings.filename_prefix_preview +
                str(self.design.get_epoch_current() + 1) +
                ' b%i-%i' % (self.design.idx_batch, self.design.n_batches) +
                self.settings.filename_postfix_preview)
            Image.fromarray((255.0 * self.img).astype(np.uint8)).save(path)


def _clip(matrix, min_, max_):
    matrix[matrix < min_] = min_
    matrix[matrix > max_] = max_
    return matrix

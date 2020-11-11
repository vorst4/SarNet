from pathlib import Path
from typing import List
from typing import Union, TYPE_CHECKING
import torch
import matplotlib.pyplot as plt
import numpy as np
from util.base_obj import BaseObj
from PIL import Image
import settings

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
    class Settings(BaseObj):
        def __init__(self,
                     print_: bool = True,
                     lossplot: bool = False,
                     preview: bool = False,
                     load_design: bool = False,
                     save_design: bool = False,
                     save_lossplot: bool = False,
                     save_preview: bool = False,
                     path: Union[str, Path] = None,
                     filename_design: str = 'design.pt',
                     n_backups: int = 3,
                     filename_best_design: str = 'best_design.pt',
                     filename_lossplot: str = 'lossplot.pdf',
                     subdir_preview: str = 'preview',
                     filename_prefix_preview: str = 'preview e',
                     filename_postfix_preview: str = '.png',
                     preview_vstack: int = 4,
                     preview_hstack: int = 5,
                     preview_padding: int = 2,
                     preview_padding_set: int = 16,
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
            :param preview_padding_set: amount of padding in pixels between
                training and validation set.
            :param figure_size: the size of the lossplot and preview that
                are generated
            """
            super().__init__(indent=' ' * 2)
            self.print = print_
            self.lossplot = lossplot
            self.preview = preview
            self.load_design = load_design
            self.save_design = save_design
            self.save_lossplot = save_lossplot
            self.save_preview = save_preview
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
            self.preview_padding_set = preview_padding_set
            self.figure_size = figure_size

        @staticmethod
        def add_subdir(
                path: str,
                modelname: str,
                partition_id: int,
                job_id: int
        ) -> str:
            if not settings.progress.load_design:
                dirname = '%s_%s_d%s_s%i_j%i' % (settings.ds, modelname,
                                                 Log.date_time(),
                                                 partition_id, job_id)
                path = str(Path(path).joinpath(dirname))
                if not Path(settings.progress.path).exists():
                    Path(settings.progress.path).mkdir()
            else:
                path = (sorted(list(
                    Path(path).glob('*%s*' % modelname)
                ))[-1])

            return path

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

        # number of horizontal/vertical images per train/valid set
        nh = 2 * self.settings.preview_hstack  # true + predicted
        nv = self.settings.preview_vstack

        # padding between images, and padding between images of data/valid set
        pad = self.settings.preview_padding
        pad_set = self.settings.preview_padding_set

        # size of 1 image in the preview buffer (resolution + padding)
        size = img_res + 2 * pad

        # define attributes
        self.res = img_res
        self.n_imgs = nh * nv

        # allocate space for image buffer
        self.imgs_buffer = np.zeros((
            2 * nv * size + pad_set,  # px vertical
            nh * size  # px horizontal
        ))
        self.img_idx_train = 0
        self.img_idx_valid = 0

        # add white line to separate train/valid set
        self.imgs_buffer[nv * size + pad_set // 2, :] = 1.

        # create plot window
        if self.settings.preview:
            plt.figure(num=self.IDX_PREVIEW, figsize=self.settings.figure_size)
            plt.axis('off')
            plt.show()

    def add_imgs_to_preview_buffer(self, imgs_pred, imgs_true):

        # add it to the training-imgs or validation-imgs buffer depending on
        #   wherever grad is enabled
        # print(self.n_imgs)
        # print(self.img_idx_train)
        # print(self.img_idx_train)
        if torch.is_grad_enabled():
            self.img_idx_train = self._add_imgs_to_buffer(
                buffer=self.imgs_buffer,
                idx_buffer=self.img_idx_train,
                imgs_pred=imgs_pred,
                imgs_true=imgs_true,
            )
        else:
            self.img_idx_valid = self._add_imgs_to_buffer(
                buffer=self.imgs_buffer,
                idx_buffer=self.img_idx_valid,
                imgs_pred=imgs_pred,
                imgs_true=imgs_true,
            )

    def _add_imgs_to_buffer(self,
                            buffer: np.ndarray,
                            idx_buffer: int,
                            imgs_pred: np.ndarray,
                            imgs_true: np.ndarray) -> int:

        # return if buffer is full
        if idx_buffer >= self.n_imgs:
            return idx_buffer

        # loop through images (<batch_size> images should be given)
        for idx in range(imgs_pred.shape[0]):

            # add true img to buffer
            self._add_img_to_buffer(buffer,
                                    idx_buffer,
                                    imgs_true[idx, 0, :, :])

            # update counter
            idx_buffer += 1

            # add predicted img to buffer
            self._add_img_to_buffer(buffer,
                                    idx_buffer,
                                    imgs_pred[idx, 0, :, :])

            # update idx buffer
            idx_buffer += 1

            # return if buffer is full
            if idx_buffer >= self.n_imgs:
                return idx_buffer

        # return current buffer idx
        return idx_buffer

    def _add_img_to_buffer(self,
                           buffer: np.ndarray,
                           idx: int,
                           img: torch.tensor) -> None:

        # number of horizontal/vertical images (per data/valid set)
        nh = 2 * self.settings.preview_hstack  # true + predicted
        nv = self.settings.preview_vstack

        # padding around each img
        pad = self.settings.preview_padding

        # the size that 1 image takes up in the buffer (resolution + padding)
        size = self.res + 2 * pad

        # training images have a vertical offset since they are displayed
        #   below the validation images
        if torch.is_grad_enabled():
            v_offset = nv * size + self.settings.preview_padding_set
        else:
            v_offset = 0

        # index of image in the buffer
        idx_h = idx % nh
        idx_v = idx // nh

        # location (in pixels) of image in buffer
        px_h = slice(
            idx_h * size + pad,  # horizontal pixel start
            (idx_h + 1) * size - pad  # horizontal pixel end
        )
        px_v = slice(
            idx_v * size + pad + v_offset,  # vertical pixel start
            (idx_v + 1) * size - pad + v_offset  # vertical pixel end
        )

        # add image to buffer
        buffer[px_v, px_h] = img.cpu().detach().numpy()

    def __call__(self):

        # print progress, if desired
        if self.settings.print:
            self._print()

        # save design, if path is given
        if self.settings.save_design:
            self.design.save(self.path_design,
                             backups=self.path_design_backup)

            # if the last validation loss is the lowest, also save it as
            # 'best' design (if epoch is finished)
            _, lv = self.design.performance.mse_valid()
            if lv[-1] == min(lv):
                self.design.save(self.path_best_design,
                                 backups=self.path_best_design_backup)

        # lossplot
        if self.settings.lossplot or self.settings.save_lossplot:
            self._lossplot()

        # preview
        if self.settings.preview or self.settings.save_preview:
            self._preview()

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

    def _print(self):
        """
        Print progress
        """
        ec = self.design.epoch_current
        es = self.design.get_epoch_stop()
        _, lt = self.design.performance.mse_train(idx=-1)
        _, lv = self.design.performance.mse_valid(idx=-1)
        self.log.__call__(
            'Epoch:%8.4f/%i, loss_train: %3.8f, loss_val: %3.8f' %
            (ec, es, lt, lv)
        )

    def _lossplot(self):
        """
        update display and/or save lossplot
        """

        # update lossplot
        et, lt = self.design.performance.mse_train()
        ev, lv = self.design.performance.mse_valid()
        self.line_train.set_xdata(et)
        self.line_train.set_ydata(lt)
        self.line_val.set_xdata(ev)
        self.line_val.set_ydata(lv)

        # adjust axes
        fig = plt.figure(self.IDX_LOSSPLOT)
        plt.xlim([0, et[-1] + 0.1])
        plt.ylim([0.9 * min([min(lt), min(lv)]),
                  1.1 * max([max(lt), max(lv)])])

        # update lossplot display
        if self.settings.lossplot:
            fig.canvas.draw()
            fig.canvas.flush_events()

        # save lossplot
        if self.settings.save_lossplot:
            try:
                fig.savefig(self.path_lossplot)
            except PermissionError:
                self.log.__call__(
                    'WARNING: permission error occurred during saving the '
                    'lossplot'
                )

    def _preview(self):
        """
        update display and/or save preview, preview being a side by side
        comparison of (multiple) true and predicted output images.

        The preview is split into a top and bottom, the top are the images
        from the validation dataset and the bottom from the training dataset.
        """

        # reset buffer counters
        self.img_idx_train, self.img_idx_valid = 0, 0

        # # currently <imgs> is a matrix with images. The matrix has two
        # #   columns with the true images on the first
        # #   column and the predicted imgs on the second. Reshape this matrix
        # #   to a grid pattern of side-by-side comparison
        # hstack = self.settings.preview_hstack
        # vstack = self.settings.preview_vstack
        # self.imgs = self.imgs.reshape(vstack, 2 * hstack, self.res2,
        # self.res2)
        #
        # # concatenate to create one big image
        # self.img_ = np.concatenate(self.imgs, axis=1)
        # self.img = np.concatenate(self.img_, axis=1)
        #
        # # reshape imgs back to original shape
        # self.imgs = self.imgs.reshape((-1, 2, self.res2, self.res2))

        # clip to [0, 1]
        _clip(self.imgs_buffer, 0, 1)
        # self.img = _clip(self.img, 0, 1)

        # show img
        if self.settings.preview:
            plt.figure(self.IDX_PREVIEW)
            plt.imshow(self.imgs_buffer, cmap='hot')
            # plt.imshow(self.img, cmap='hot')

        # save img
        if self.settings.save_preview:
            path = self.dir_preview.joinpath(
                self.settings.filename_prefix_preview +
                '%.4f' % self.design.get_epoch_current() +
                self.settings.filename_postfix_preview)
            Image.fromarray(
                (255.0 * self.imgs_buffer).astype(np.uint8)
            ).save(path)
            # Image.fromarray((255.0 * self.img).astype(np.uint8)).save(path)


def _clip(matrix, min_, max_):
    matrix[matrix < min_] = min_
    matrix[matrix > max_] = max_

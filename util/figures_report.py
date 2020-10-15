from pathlib import Path
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from .design import Design
from .log import Log

IEEE_FONTSIZE = 8
IEEE_FONTSIZE_SMALL = 6
IEEE_COLUMN_WIDTH = 2.02  # inch


def _change_default_figure_settings():
    plt.rcParams["axes.titlesize"] = str(IEEE_FONTSIZE)
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['axes.titlepad'] = 0
    plt.rcParams['axes.labelpad'] = 0

    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["figure.figsize"] = IEEE_COLUMN_WIDTH, IEEE_COLUMN_WIDTH
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.constrained_layout.h_pad"] = 0
    plt.rcParams["figure.constrained_layout.hspace"] = 0
    plt.rcParams["figure.constrained_layout.w_pad"] = 0
    plt.rcParams["figure.constrained_layout.wspace"] = 0
    plt.rcParams["figure.subplot.bottom"] = 0
    plt.rcParams["figure.subplot.hspace"] = 0
    plt.rcParams["figure.subplot.left"] = 0
    plt.rcParams["figure.subplot.right"] = 1
    plt.rcParams["figure.subplot.top"] = 1
    plt.rcParams["figure.subplot.wspace"] = 0

    plt.rcParams["font.family"] = 'Times New Roman'
    plt.rcParams["font.size"] = str(IEEE_FONTSIZE)

    plt.rcParams['legend.borderaxespad'] = 0.2
    plt.rcParams['legend.borderpad'] = 0.1
    plt.rcParams['legend.columnspacing'] = 0
    plt.rcParams['legend.fancybox'] = False
    plt.rcParams['legend.framealpha'] = 1
    plt.rcParams["legend.fontsize"] = str(IEEE_FONTSIZE_SMALL)
    plt.rcParams["legend.labelspacing"] = 0
    plt.rcParams['legend.markerscale'] = 5

    plt.rcParams["lines.linewidth"] = 1

    plt.rcParams['savefig.pad_inches'] = 0
    plt.rcParams['savefig.format'] = 'pdf'
    plt.rcParams['savefig.bbox'] = 'tight'

    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['xtick.minor.width'] = 0.5
    plt.rcParams['xtick.major.pad'] = 1

    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['ytick.major.pad'] = 1
    plt.rcParams['ytick.minor.width'] = 0.5
    plt.rcParams['ytick.minor.pad'] = 1


def _get_square_aspect(axis: plt.Axes, log10=True):
    x_min, x_max = axis.get_xlim()
    y_min, y_max = axis.get_ylim()
    return (x_max - x_min) / (np.log10(y_max) - np.log10(y_min))


def _remap(array, indexes1: List[int], indexes2: List[int]):
    for ii in range(len(indexes1)):
        array[indexes1[ii]], array[indexes2[ii]] = array[indexes2[ii]], array[
            indexes1[ii]]
    return array


def figures_report(paths_design: List[Path],
                   dir_dst: Union[Path, str],
                   fn_train_mse_epoch: Union[
                       Path, str] = 'train_mse_epoch.pdf',
                   fn_valid_mse_epoch: Union[
                       Path, str] = 'valid_mse_epoch.pdf',
                   fn_mse_phase: Union[Path, str] = 'mse_phase.pdf',
                   fn_tae_phase: Union[Path, str] = 'tae_phase.pdf',
                   show: bool = True
                   ):
    # convert dir_dst to pathlib.Path if it is a string
    if isinstance(dir_dst, str):
        dir_dst = Path(dir_dst)

    # get modelnames and load designs
    log = Log('output', save_log=False)
    dlt, dlv = Design.get_default_dataloaders('data/dataset.csv', 1)
    designs = [Design().load(path, dlt, dlv, log) for path in paths_design]
    modelnames = [str(path.parents[0]).split('\\')[-1] for path in
                  paths_design]

    # change modelnames
    modelnames = [modelname.replace('AE', '').replace('Info', 'Conv3') for
                  modelname in modelnames]

    # change order of designs/modelnames
    # ids_from = [0, -1, -1, -2, -1]
    # ids_to = [-1, 1, 2, 3, 4]
    ids_from = [2, 3]
    ids_to = [0, 1]
    modelnames = _remap(modelnames, ids_from, ids_to)
    designs = _remap(designs, ids_from, ids_to)

    # calculate: mean squared error & top absolute error
    tae, mse, imgs_ae, imgs_p = [], [], [], []
    for design in designs:
        tae_, mse_, imgs_ae_, imgs_p_, imgs_t = _evaluate(design)
        tae.append(tae_)
        mse.append(mse_)
        imgs_ae.append(imgs_ae_)
        imgs_p.append(imgs_p_)

    # create and save figures
    _change_default_figure_settings()
    _mse_epoch(designs, modelnames, (dir_dst.joinpath(fn_train_mse_epoch),
                                     dir_dst.joinpath(fn_valid_mse_epoch)))
    _mse_phase(mse, modelnames, dir_dst.joinpath(fn_mse_phase))
    _tae_phase(tae, modelnames, dir_dst.joinpath(fn_tae_phase))

    # # create and save gifs
    # for (modelname, imgs_ae_, imgs_p_) in zip(modelnames, imgs_ae, imgs_p):
    #     _gif_from_tensor(imgs_p_, dir_dst.joinpath('tmp/'+modelname+'pred.gif'))
    #     max_tae = max([t.max() for t in imgs_ae])
    #     _gif_from_tensor(imgs_ae_/max_tae, dir_dst.joinpath(modelname+'error.gif'))

    if show:
        plt.show()


def _gif_from_tensor(tensor: torch.Tensor, dst: Path):
    from array2gif import write_gif
    np_gif = np.tile(
        np.array(tensor.clamp(0, 1) * 255, dtype=int).swapaxes(1, -1),
        (1, 1, 1, 3))
    gif = [np_gif[idx, :, :, :] for idx in range(np_gif.shape[0])]
    write_gif(gif, dst, fps=50)


def _mse_epoch(designs, modelnames, paths_mse_epoch):
    n_epochs = 1000
    n_rows = 2
    x = np.arange(1, n_epochs + 1)

    # obtain mse of train and val per epoch
    y = [[], []]
    for design in designs:
        yyt = design._loss_train
        yyv = design._loss_val
        yt = np.zeros(n_epochs, dtype=float)
        yv = np.zeros(n_epochs, dtype=float)
        ntb = int(len(yyt) / n_epochs)  # number of train batches
        nvb = int(len(yyv) / n_epochs)  # number of valid batches
        for idx2 in range(n_epochs):
            yt[idx2] = np.mean(yyt[slice(idx2 * ntb, (idx2 + 1) * ntb - 1, 1)])
            yv[idx2] = np.mean(yyv[slice(idx2 * nvb, (idx2 + 1) * nvb - 1, 1)])
        y[0].append(yt)
        y[1].append(yv)

    titles = ['Training', 'Validation']
    for idx in (0, 1):
        plt.figure(num=paths_mse_epoch[idx].stem)
        for idx2, modelname in enumerate(modelnames):
            plt.semilogy(x, y[idx][idx2], '.', markersize=1, label=modelname)
        plt.title(titles[idx])
        plt.xlabel('epochs')
        plt.ylabel('mean squared error')
        plt.ylim([1e-4, 1e-1])
        plt.gca().set_aspect(_get_square_aspect(plt.gca()))
        plt.legend()
        plt.savefig(paths_mse_epoch[idx])


def _mse_phase(mse, modelnames, path_mse_phase):
    plt.figure(num=path_mse_phase.stem)
    for mse_, modelname in zip(mse, modelnames):
        plt.semilogy(mse_['phase'], mse_['mse'], label=modelname)
    plt.ylim([1e-4, 1e-1])
    plt.ylabel('mean squared error')
    plt.xlabel('phase [deg]')
    plt.legend()
    plt.gca().set_aspect(_get_square_aspect(plt.gca()))
    plt.savefig(path_mse_phase)


def _tae_phase(tae, modelnames, path_tae_phase):
    plt.figure(num=path_tae_phase.stem)
    lim = [0, 0]
    for tae_, modelname in zip(tae, modelnames):
        plt.semilogy(tae_['phase'], tae_['tae'], label=modelname)
        lim = np.min(tae_['phase']), np.max(tae_['phase'])
    plt.semilogy(lim, [1e-2, 1e-2], ':', label='TAE=1%')
    plt.ylim([9e-3, 1e0])
    plt.ylabel('top absolute error')
    plt.xlabel('phase [deg]')
    plt.legend(loc='lower right', bbox_to_anchor=(0.97, 0.03))
    plt.gca().set_aspect(_get_square_aspect(plt.gca()))
    plt.savefig(path_tae_phase)


def _evaluate(design):
    # set model to evaluation mode
    design._model.eval()

    tae = {'phase': [], 'tae': []}
    mse = {'phase': [], 'mse': []}
    imgs_ae: torch.Tensor = torch.Tensor()
    imgs_p = torch.Tensor()
    imgs_t = torch.Tensor()

    # gradient does not have to be calculated during evaluation
    with torch.no_grad():
        # loop over the evaluation dataset
        for idx_batch, data in enumerate(design._dl_val):
            # abbreviate variables and move to <device>
            yt = data['output_img'].to(device=design.device,
                                       dtype=design.dtype)
            x1 = data['input_img'].to(device=design.device, dtype=design.dtype)
            x2 = data['input_meta'].to(device=design.device,
                                       dtype=design.dtype)

            yp = design._model(x1, x2).cpu().detach()

            yd = torch.abs(yt - yp)

            phase = (x2[:, 1].cpu().detach() * 180 / np.pi).tolist()
            tae['phase'].extend(phase)
            tae['tae'].extend(_calc_tae(yd))
            mse['phase'].extend(phase)
            mse['mse'].extend(_calc_mse(yd))

            imgs_ae = yd if len(imgs_ae) == 0 else torch.cat((imgs_ae, yd))
            imgs_p = yp if len(imgs_p) == 0 else torch.cat((imgs_p, yp))
            imgs_t = yt if len(imgs_t) == 0 else torch.cat((imgs_t, yt))

    return tae, mse, imgs_ae, imgs_p, imgs_t


def _calc_tae(img_err):
    return torch.max(torch.max(img_err, dim=3).values, dim=2).values.tolist()


def _calc_mse(img_err):
    return torch.mean(img_err ** 2, dim=[2, 3]).tolist()

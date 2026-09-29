"""Paneles reales por caso; la selección de corte/ejemplos queda registrada."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def panels(image, gt, pred, out, case_id):
    image, gt, pred = np.asarray(image), np.asarray(gt), np.asarray(pred)
    if (image.ndim != 4 or image.shape[0] != 4 or gt.ndim != 3 or not gt.size or
            gt.shape != pred.shape or image.shape[1:] != gt.shape):
        raise ValueError('Four aligned MRI channels and two aligned 3D masks required')
    if not all(np.isfinite(x).all() for x in (image, gt, pred)):
        raise ValueError('Figure inputs must contain finite values')
    gt, pred = gt.astype(bool), pred.astype(bool)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Prespecified slice with maximum GT area; ties choose first. Not a model-dependent slice.
    z = int(np.argmax(gt.sum(axis=(0, 1))))
    g, p = gt[:, :, z].T, pred[:, :, z].T
    def gray(ax, data):
        nz = data[np.isfinite(data) & (data != 0)]
        lo, hi = np.percentile(nz, [1, 99]) if nz.size else (0, 1)
        ax.imshow(data.T, cmap='gray', origin='lower', vmin=lo, vmax=hi)
        ax.axis('off')
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    captions = ['T1: structural anatomy', 'T1c: contrast enhancement',
                'T2: water-sensitive signal', 'FLAIR: fluid-suppressed signal']
    for i, ax in enumerate(axes):
        gray(ax, image[i, :, :, z])
        if g.any() and not g.all():
            ax.contour(g, levels=[.5], colors=['yellow'], linewidths=.8)
        ax.set_title(captions[i], fontsize=9)
    fig.suptitle(f'{case_id} | axial RAS z={z}; yellow: whole-tumor reference')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'{case_id}_modalities.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 5, figsize=(17, 4))
    for ax in axes:
        gray(ax, image[3, :, :, z])
    axes[1].imshow(np.ma.masked_where(~g, g), cmap=ListedColormap(['#ffe34d']), alpha=.6, origin='lower')
    axes[2].imshow(np.ma.masked_where(~p, p), cmap=ListedColormap(['#22b8ff']), alpha=.6, origin='lower')
    for mask, color in ((g, '#ffe34d'), (p, '#22b8ff')):
        if mask.any() and not mask.all():
            axes[3].contour(mask, levels=[.5], colors=[color], linewidths=1.)
    errors = np.zeros(g.shape, dtype=int)
    errors[g & p], errors[~g & p], errors[g & ~p] = 1, 2, 3
    axes[4].imshow(np.ma.masked_where(errors == 0, errors), origin='lower',
                   cmap=ListedColormap(['#43c463', '#f04438', '#3094ff']), vmin=1, vmax=3, alpha=.8)
    for ax, title in zip(axes, ['MRI (FLAIR)', 'Ground truth', 'Prediction', 'GT yellow / Pred cyan',
                               'TP green / FP red / FN blue']):
        ax.set_title(title, fontsize=10)
    fig.suptitle(f'{case_id} | z={z}, slice selected by maximum reference area')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'{case_id}_comparison.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    return z

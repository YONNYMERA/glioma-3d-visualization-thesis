"""Métricas por caso: no oculta fallos de máscara vacía ni denominadores indefinidos."""
import numpy as np
from scipy import ndimage


def ratio(n, d):
    return float(n / d) if d else float('nan')


def binary_metrics(pred, gt, spacing=(1., 1., 1.), tolerance_mm=1.):
    pred, gt = np.asarray(pred), np.asarray(gt)
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError('Masks must contain finite values')
    pred, gt = pred.astype(bool), gt.astype(bool)
    spacing = np.asarray(spacing, float)
    if (pred.shape != gt.shape or pred.ndim != 3 or not pred.size or
            spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0)):
        raise ValueError('Aligned 3D masks and positive 3D spacing required')
    if not np.isfinite(tolerance_mm) or tolerance_mm < 0:
        raise ValueError('Surface tolerance must be finite and nonnegative')
    tp, fp = np.count_nonzero(pred & gt), np.count_nonzero(pred & ~gt)
    tn, fn = np.count_nonzero(~pred & ~gt), np.count_nonzero(~pred & gt)
    p, g = tp + fp, tp + fn
    ml = float(np.prod(spacing) / 1000.)
    out = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
           'Dice': ratio(2 * tp, p + g) if p + g else 1.,
           'IoU': ratio(tp, tp + fp + fn) if p + g else 1.,
           'Sensitivity': ratio(tp, g), 'Precision': ratio(tp, p),
           'Specificity': ratio(tn, tn + fp), 'NPV': ratio(tn, tn + fn),
           'Vol_GT_mL': g * ml, 'Vol_Pred_mL': p * ml,
           'Vol_Error_mL': (p - g) * ml, 'AVD_mL': abs(p - g) * ml,
           'RVE_percent': ratio(100 * (p - g), g),
           'Abs_RVE_percent': ratio(100 * abs(p - g), g),
           'empty_gt': int(g == 0), 'empty_pred': int(p == 0),
           'surface_tolerance_mm': float(tolerance_mm)}
    if p and g:
        structure = ndimage.generate_binary_structure(3, 1)
        ps = pred ^ ndimage.binary_erosion(pred, structure=structure, border_value=0)
        gs = gt ^ ndimage.binary_erosion(gt, structure=structure, border_value=0)
        dpg = ndimage.distance_transform_edt(~gs, sampling=spacing)[ps]
        dgp = ndimage.distance_transform_edt(~ps, sampling=spacing)[gs]
        # HD95: max of directed 95th percentiles of boundary-voxel-center distances.
        # ASSD: sum of both directed distances / total boundary voxel count (not area weighted).
        # SurfaceDice below is separately area weighted using MONAI subvoxel surfaces.
        out['HD95_mm'] = float(max(np.percentile(dpg, 95), np.percentile(dgp, 95)))
        out['ASSD_mm'] = float((dpg.sum() + dgp.sum()) / (len(dpg) + len(dgp)))
        import torch
        from monai.metrics import compute_surface_dice
        out['SurfaceDice'] = float(compute_surface_dice(
            torch.from_numpy(pred[None, None].astype(np.float32)),
            torch.from_numpy(gt[None, None].astype(np.float32)),
            class_thresholds=[float(tolerance_mm)], include_background=True,
            spacing=tuple(spacing), use_subvoxels=True).item())
    elif p == 0 and g == 0:
        out.update(HD95_mm=0., ASSD_mm=0., SurfaceDice=1.)
    else:
        out.update(HD95_mm=float('inf'), ASSD_mm=float('inf'), SurfaceDice=0.)
    return out

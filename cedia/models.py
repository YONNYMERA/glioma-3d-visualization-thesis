"""Configuración MONAI extraída del cuaderno; SegResNet es experimento nuevo."""
import torch
from monai.networks.nets import UNet, SegResNet
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    Lambdad, NormalizeIntensityd, CropForegroundd, SpatialPadd, DivisiblePadd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d, RandGaussianNoised, ToTensord)
from monai.inferers import sliding_window_inference


def binarize(x):
    return (x > 0).float()


def model(name):
    if name == 'unet':
        return UNet(spatial_dims=3, in_channels=4, out_channels=1,
                    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
                    num_res_units=2, norm='batch')
    if name == 'segresnet':
        return SegResNet(spatial_dims=3, init_filters=16, in_channels=4, out_channels=1,
                         blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
                         norm=('GROUP', {'num_groups': 8}), dropout_prob=None)
    raise ValueError(name)


def transforms(training=False, roi=(128, 128, 128), with_prediction=False):
    keys = ['image', 'label'] + (['prediction'] if with_prediction else [])
    modes = ('bilinear', 'nearest') + (('nearest',) if with_prediction else ())
    ops = [LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys),
           Spacingd(keys=keys, pixdim=(1., 1., 1.), mode=modes),
           Orientationd(keys=keys, axcodes='RAS'),
           Lambdad(keys=['label'] + (['prediction'] if with_prediction else []), func=binarize),
           NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
           CropForegroundd(keys=keys, source_key='image', allow_smaller=False)]
    if training:
        ops.extend([SpatialPadd(keys=keys, spatial_size=roi),
                    RandCropByPosNegLabeld(keys=keys, label_key='label', spatial_size=roi,
                                          pos=1, neg=1, num_samples=2)])
        ops.extend(RandFlipd(keys=keys, prob=.5, spatial_axis=axis) for axis in range(3))
        ops.extend([RandRotate90d(keys=keys, prob=.5, max_k=3),
                    RandGaussianNoised(keys=['image'], prob=.15, std=.1)])
    else:
        ops.append(DivisiblePadd(keys=keys, k=16))
    ops.append(ToTensord(keys=keys))
    return Compose(ops)


def infer(net, inputs, roi, overlap=.5):
    return torch.sigmoid(sliding_window_inference(inputs, roi, 1, net, overlap=overlap, mode='constant')) > .5


def load_weights(net, path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if 'model' in state:
        state = state['model']
    net.load_state_dict(state, strict=True)
    return net.eval()

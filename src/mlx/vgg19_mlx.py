"""MLX/NHWC port of the Vgg19 perceptual feature extractor from src/modules/model.py.

torchvision vgg19.features up to relu5_1 (index 29), sliced into 5 activation taps
(after feature idx 1, 6, 11, 20, 29) = [relu1_1, relu2_1, relu3_1, relu4_1, relu5_1].
ImagePyramide downsamples at given scales (reuses blocks_mlx.AntiAliasInterpolation2d).
perceptual_pyramid_loss computes the L1 multi-scale perceptual loss (model.py lines 171-180).

Conv layers are named to mirror torchvision feature indices so a vgg19.features
state_dict transfers key-for-key (conv weight (out,in,kH,kW)->(out,kH,kW,in)).
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from blocks_mlx import AntiAliasInterpolation2d

# VGG19 'E' feature layout up to idx 29. (idx, kind, in, out) for convs; pools/relus implicit.
_CONV_IDX = {
    0: (3, 64),
    2: (64, 64),
    5: (64, 128),
    7: (128, 128),
    10: (128, 256),
    12: (256, 256),
    14: (256, 256),
    16: (256, 256),
    19: (256, 512),
    21: (512, 512),
    23: (512, 512),
    25: (512, 512),
    28: (512, 512),
}
_POOL_IDX = {4, 9, 18, 27}
_RELU_IDX = {1, 3, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26, 29}
_TAPS = [1, 6, 11, 20, 29]  # collect activation after these feature indices
_N = 30  # process idx 0..29


class Vgg19(nn.Module):
    def __init__(self):
        super().__init__()
        # ImageNet normalisation (NHWC: last-axis broadcast)
        self.mean = mx.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        self.std = mx.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        # convs keyed "conv{idx}" so load maps features.{idx}.weight
        self.convs = {}
        for idx, (ci, co) in _CONV_IDX.items():
            self.convs[f"conv{idx}"] = nn.Conv2d(ci, co, 3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def __call__(self, x):
        # x NHWC in [0,1]
        x = (x - self.mean) / self.std
        taps = []
        for i in range(_N):
            if i in _CONV_IDX:
                x = self.convs[f"conv{i}"](x)
            elif i in _POOL_IDX:
                x = self.pool(x)
            else:  # relu
                x = nn.relu(x)
            if i in _TAPS:
                taps.append(x)
        return taps  # [relu1_1..relu5_1]


class ImagePyramide(nn.Module):
    def __init__(self, scales, num_channels):
        super().__init__()
        self.scales = list(scales)
        self.downs = {}
        for s in scales:
            self.downs[str(s).replace(".", "-")] = AntiAliasInterpolation2d(
                num_channels, s
            )

    def __call__(self, x):
        out = {}
        for s in self.scales:
            key = str(s).replace(".", "-")
            out["prediction_" + str(s)] = self.downs[key](x)
        return out


def perceptual_pyramid_loss(vgg, pyramid, generated, real, scales, weights):
    """L1 multi-scale perceptual loss (model.py 171-180). generated/real are NHWC images."""
    pg = pyramid(generated)
    pr = pyramid(real)
    total = mx.array(0.0)
    for s in scales:
        xg = vgg(pg["prediction_" + str(s)])
        yg = vgg(pr["prediction_" + str(s)])
        for i, w in enumerate(weights):
            total = total + w * mx.mean(mx.abs(xg[i] - mx.stop_gradient(yg[i])))
    return total


def load_vgg_from_torch(mlx_vgg, torch_features_sd):
    """torch_features_sd = vgg19.features.state_dict() (keys '0.weight','2.weight',...)."""
    flat = []
    for k, v in torch_features_sd.items():
        idx = int(k.split(".")[0])
        if idx not in _CONV_IDX:
            continue
        which = k.split(".")[1]  # weight|bias
        arr = mx.array(v.detach().cpu().numpy())
        if arr.ndim == 4:
            arr = mx.transpose(arr, (0, 2, 3, 1))
        flat.append((f"convs.conv{idx}.{which}", arr))
    mlx_vgg.unfreeze(recurse=True)
    mlx_vgg.update(tree_unflatten(flat))
    mlx_vgg.eval()
    return mlx_vgg

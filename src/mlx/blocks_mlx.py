"""MLX/NHWC port of the conv blocks in src/modules/util.py (TPS).

Blocks: ResBlock2d, UpBlock2d, DownBlock2d, SameBlock2d, Encoder, Decoder,
Hourglass, AntiAliasInterpolation2d. Named to mirror torchvision-style keys so a
torch state_dict transfers key-for-key (conv weight (out,in,kH,kW)->(out,kH,kW,in);
InstanceNorm weight/bias are 1-D, no transpose).

load_state_from_torch(mlx_module, torch_state_dict) transfers a flat torch dict.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten


def _instnorm(c):
    return nn.InstanceNorm(c, affine=True)


class ResBlock2d(nn.Module):
    def __init__(self, in_features, kernel_size, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size, padding=padding)
        self.norm1 = _instnorm(in_features)
        self.norm2 = _instnorm(in_features)

    def __call__(self, x):
        out = nn.relu(self.norm1(x))
        out = self.conv1(out)
        out = nn.relu(self.norm2(out))
        out = self.conv2(out)
        return out + x


class UpBlock2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_features, out_features, kernel_size, padding=padding, groups=groups
        )
        self.norm = _instnorm(out_features)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def __call__(self, x):
        out = self.up(x)
        out = self.conv(out)
        out = self.norm(out)
        return nn.relu(out)


class DownBlock2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_features, out_features, kernel_size, padding=padding, groups=groups
        )
        self.norm = _instnorm(out_features)
        self.pool = nn.AvgPool2d(kernel_size=2)

    def __call__(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = nn.relu(out)
        return self.pool(out)


class SameBlock2d(nn.Module):
    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_features, out_features, kernel_size, padding=padding, groups=groups
        )
        self.norm = _instnorm(out_features)

    def __call__(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return nn.relu(out)


class Encoder(nn.Module):
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super().__init__()
        down = []
        for i in range(num_blocks):
            inp = in_features if i == 0 else min(max_features, block_expansion * (2**i))
            outp = min(max_features, block_expansion * (2 ** (i + 1)))
            down.append(DownBlock2d(inp, outp, kernel_size=3, padding=1))
        self.down_blocks = down

    def __call__(self, x):
        outs = [x]
        for d in self.down_blocks:
            outs.append(d(outs[-1]))
        return outs


class Decoder(nn.Module):
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super().__init__()
        up = []
        self.out_channels = []
        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(
                max_features, block_expansion * (2 ** (i + 1))
            )
            self.out_channels.append(in_filters)
            out_filters = min(max_features, block_expansion * (2**i))
            up.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))
        self.up_blocks = up
        self.out_channels.append(block_expansion + in_features)

    def __call__(self, x, mode=0):
        out = x.pop()
        outs = []
        for u in self.up_blocks:
            out = u(out)
            skip = x.pop()
            out = mx.concatenate([out, skip], axis=-1)  # NHWC channel concat
            outs.append(out)
        return out if mode == 0 else outs


class Hourglass(nn.Module):
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super().__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_channels = self.decoder.out_channels

    def __call__(self, x, mode=0):
        return self.decoder(self.encoder(x), mode)


class AntiAliasInterpolation2d(nn.Module):
    """Band-limited downsampling: fixed depthwise gaussian conv + nearest interpolate."""

    def __init__(self, channels, scale):
        super().__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka
        self.scale = scale
        self.channels = channels
        # build gaussian kernel (kh, kw)
        ax = mx.arange(kernel_size).astype(mx.float32)
        mean = (kernel_size - 1) / 2
        g1 = mx.exp(-((ax - mean) ** 2) / (2 * sigma**2))
        kernel = g1[:, None] * g1[None, :]
        kernel = kernel / mx.sum(kernel)
        # depthwise weight NHWC: (out=channels, kh, kw, in_per_group=1)
        self.weight = mx.broadcast_to(
            kernel[None, :, :, None], (channels, kernel_size, kernel_size, 1)
        )

    def __call__(self, x):
        if self.scale == 1.0:
            return x
        # x NHWC; pad H,W then depthwise conv (groups=channels), then nearest-resize
        xp = mx.pad(x, [(0, 0), (self.ka, self.kb), (self.ka, self.kb), (0, 0)])
        out = mx.conv2d(xp, self.weight, stride=1, padding=0, groups=self.channels)
        # F.interpolate(scale_factor=scale) nearest, scale<1 -> downsample
        n, h, w, c = out.shape
        nh, nw = int(h * self.scale), int(w * self.scale)
        # nearest index map
        ih = (mx.arange(nh).astype(mx.float32) / self.scale).astype(mx.int32)
        iw = (mx.arange(nw).astype(mx.float32) / self.scale).astype(mx.int32)
        out = out[:, ih, :, :]
        out = out[:, :, iw, :]
        return out


def load_state_from_torch(mlx_module, torch_state_dict):
    flat = []
    for k, v in torch_state_dict.items():
        if k.endswith("num_batches_tracked"):
            continue
        arr = mx.array(v.detach().cpu().numpy())
        if arr.ndim == 4:
            arr = mx.transpose(arr, (0, 2, 3, 1))
        flat.append((k, arr))
    mlx_module.unfreeze(recurse=True)
    mlx_module.update(tree_unflatten(flat))
    mlx_module.eval()
    return mlx_module

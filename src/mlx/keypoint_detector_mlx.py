"""MLX port of src/modules/keypoint_detector.py::KPDetector (the NK path).

KPDetector = torchvision resnet18 (fg_encoder) with fc -> num_tps*5*2, then
sigmoid * 2 - 1, reshaped to (bs, num_tps*5, 2).

This file ports resnet18 faithfully (BasicBlock x2 per stage) in MLX/NHWC, named to
mirror torchvision so a torch state_dict transfers key-for-key (conv weights
transposed (out,in,kH,kW)->(out,kH,kW,in); num_batches_tracked dropped).

load_kpdetector_from_torch(mlx_model, torch_state_dict) does the transfer; used by
both the parity test and (later) the TPS vox.pth.tar warm-start.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(planes)
        if downsample:
            # torch: Sequential(Conv2d 1x1 stride, BatchNorm) -> keys downsample.0 / .1
            self.downsample = [
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm(planes),
            ]
        else:
            self.downsample = None

    def __call__(self, x):
        identity = x
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample[1](self.downsample[0](x))
        return nn.relu(out + identity)


class ResNet18(nn.Module):
    """torchvision resnet18 (NHWC), fc -> num_classes."""

    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._stage(64, 64, 2, stride=1)
        self.layer2 = self._stage(64, 128, 2, stride=2)
        self.layer3 = self._stage(128, 256, 2, stride=2)
        self.layer4 = self._stage(256, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def _stage(self, in_planes, planes, blocks, stride):
        layers = [
            BasicBlock(
                in_planes,
                planes,
                stride=stride,
                downsample=(stride != 1 or in_planes != planes),
            )
        ]
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes, stride=1, downsample=False))
        return layers

    def __call__(self, x):
        # x: NHWC
        x = nn.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        for blk in self.layer1:
            x = blk(x)
        for blk in self.layer2:
            x = blk(x)
        for blk in self.layer3:
            x = blk(x)
        for blk in self.layer4:
            x = blk(x)
        x = mx.mean(x, axis=(1, 2))  # adaptive avg pool -> (B, 512)
        return self.fc(x)


class KPDetector(nn.Module):
    """Δ-free NK detector. fg_encoder=resnet18, out reshaped to (bs, num_tps*5, 2)."""

    def __init__(self, num_tps):
        super().__init__()
        self.num_tps = num_tps
        self.fg_encoder = ResNet18(num_tps * 5 * 2)

    def __call__(self, image):
        # image: NHWC
        fg_kp = self.fg_encoder(image)
        bs = fg_kp.shape[0]
        fg_kp = mx.sigmoid(fg_kp)
        fg_kp = fg_kp * 2 - 1
        return {"fg_kp": fg_kp.reshape(bs, self.num_tps * 5, -1)}


def load_kpdetector_from_torch(mlx_model, torch_state_dict, prefix="fg_encoder."):
    """Transfer a torch KPDetector state_dict into the MLX KPDetector.

    torch keys are like 'fg_encoder.conv1.weight', 'fg_encoder.layer1.0.bn1.running_mean',
    'fg_encoder.fc.weight'. We strip nothing (mlx model also nests under fg_encoder),
    transpose conv weights, drop num_batches_tracked.
    """
    flat = []
    for k, v in torch_state_dict.items():
        if k.endswith("num_batches_tracked"):
            continue
        arr = mx.array(v.detach().cpu().numpy())
        if arr.ndim == 4:  # conv weight (out,in,kH,kW) -> (out,kH,kW,in)
            arr = mx.transpose(arr, (0, 2, 3, 1))
        flat.append((k, arr))
    mlx_model.unfreeze(recurse=True)
    mlx_model.update(tree_unflatten(flat))
    mlx_model.eval()
    return mlx_model

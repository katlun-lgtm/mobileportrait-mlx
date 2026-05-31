"""MLX/NHWC port of src/modules/bg_motion_predictor.py::BGMotionPredictor.

resnet18 with a 6-channel conv1 (concat of source+driving on the channel axis) and
fc -> 6, reshaped into the top two rows of a 3x3 affine (bottom row [0,0,1]). Used as the
optional background-motion estimator in the dense-motion path (bg_param) when bg=True.

The resnet body is the same ResNet18 ported for the keypoint detector, reused with
in_channels=6. Named to mirror torch (self.bg_encoder.conv1/.../.fc) so a torch
state_dict transfers key-for-key; conv weights are transposed (out,in,kH,kW)->(out,kH,kW,in).
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from keypoint_detector_mlx import ResNet18


class BGMotionPredictor(nn.Module):
    """source,driving (NHWC) -> (bs,3,3) affine. Init = identity (fc zero, bias [1,0,0,0,1,0])."""

    def __init__(self):
        super().__init__()
        self.bg_encoder = ResNet18(6, in_channels=6)
        # torch zero-inits fc.weight and sets bias to the identity affine [1,0,0,0,1,0]
        self.bg_encoder.fc.weight = mx.zeros_like(self.bg_encoder.fc.weight)
        self.bg_encoder.fc.bias = mx.array([1.0, 0, 0, 0, 1, 0], dtype=mx.float32)

    def __call__(self, source_image, driving_image):
        # source,driving: NHWC -> concat on channel axis (=last in NHWC)
        bs = source_image.shape[0]
        x = mx.concatenate([source_image, driving_image], axis=-1)  # (bs,h,w,6)
        prediction = self.bg_encoder(x)  # (bs,6)
        top = prediction.reshape(bs, 2, 3)
        bottom = mx.broadcast_to(mx.array([[0.0, 0.0, 1.0]]), (bs, 1, 3))
        return mx.concatenate([top, bottom], axis=1)  # (bs,3,3)


def load_bg_from_torch(mlx_model, torch_state_dict):
    """Transfer a torch BGMotionPredictor state_dict (keys 'bg_encoder.*') into the MLX
    model. Conv weights transposed; num_batches_tracked dropped."""
    flat = []
    for k, v in torch_state_dict.items():
        if k.endswith("num_batches_tracked"):
            continue
        arr = mx.array(v.detach().cpu().numpy())
        if arr.ndim == 4:  # conv weight (out,in,kH,kW) -> (out,kH,kW,in)
            arr = mx.transpose(arr, (0, 2, 3, 1))
        flat.append((k, arr))
    mlx_keys = set(k for k, _ in tree_flatten(mlx_model.parameters()))
    torch_keys = set(k for k, _ in flat)
    mlx_model.unfreeze(recurse=True)
    mlx_model.update(tree_unflatten(flat))
    mlx_model.eval()
    return sorted(mlx_keys - torch_keys), sorted(torch_keys - mlx_keys)

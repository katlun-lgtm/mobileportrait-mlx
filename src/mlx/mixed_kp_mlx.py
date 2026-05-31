"""MLX port of MobilePortrait Δ1 — MixedKPDetector.

Mirrors src/modules/keypoint_detector.py::MixedKPDetector plus its two helpers
(src/modules/mixed_kp.py::MixedKP, src/modules/fk_detector.py::FKDetector):

    MixedKPDetector = NK (learned KPDetector, already ported in keypoint_detector_mlx)
                      + frozen 106-pt FK detector
                      + MixedKP fusion MLP

The FK detector is FROZEN (no grad, never trained). Two backends:
  - "stub": deterministic ring of points in [-1,1], identical to the torch stub, so the
    fusion MLP + downstream DMN can be built and parity-tested with no extra deps.
  - "insightface": real buffalo_l 2d106 landmarks (Mac/training-box only; lazy import).

The fused keypoints are exposed as "fg_kp" so DenseMotionNetwork / TPS consume them
unchanged; "nk_kp" and "fk_kp" are also returned (the Δ3 kp_distance loss needs fk_kp).

load_mixedkp_from_torch() transfers a torch MixedKPDetector state_dict key-for-key
(conv weights transposed (out,in,kH,kW)->(out,kH,kW,in); torch Sequential mlp indices
0/2/4 -> mlx list indices 0/1/2; num_batches_tracked dropped).
"""

import re

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from keypoint_detector_mlx import KPDetector


class MixedKP(nn.Module):
    """Fuse FK (106) + NK (num_nk) keypoints -> num_mixed keypoints in [-1,1].

    concat(fk 106*2, nk num_nk*2) -> Linear/ReLU stack -> num_mixed*2 -> tanh -> (.,.,2).
    Torch stores the MLP as nn.Sequential(Linear, ReLU, Linear, ReLU, Linear); here the
    three Linears live in self.mlp (a list) and ReLU is applied functionally between them.
    """

    def __init__(self, num_fk=106, num_nk=50, num_mixed=50, hidden=(256, 256)):
        super().__init__()
        din = num_fk * 2 + num_nk * 2
        layers, d = [], din
        for h in hidden:
            layers.append(nn.Linear(d, h))
            d = h
        layers.append(nn.Linear(d, num_mixed * 2))
        self.mlp = layers  # 3 Linears; ReLU applied between (see __call__)
        self.num_mixed = num_mixed

    def __call__(self, fk, nk):
        """fk (bs,106,2), nk (bs,num_nk,2) -> (bs,num_mixed,2) in [-1,1]."""
        bs = fk.shape[0]
        x = mx.concatenate([fk.reshape(bs, -1), nk.reshape(bs, -1)], axis=1)
        n = len(self.mlp)
        for i, lin in enumerate(self.mlp):
            x = lin(x)
            if i < n - 1:
                x = nn.relu(x)
        return mx.tanh(x).reshape(bs, self.num_mixed, 2)


class FKDetector(nn.Module):
    """Frozen 106-pt facial-keypoint detector (no trainable params).

    image is NHWC in [0,1] (MLX convention). stub backend ignores pixels and returns a
    deterministic placeholder; insightface backend runs real 2d106 landmark detection.
    """

    def __init__(self, backend="stub", num_fk=106):
        super().__init__()
        self.num_fk = num_fk
        self.backend = backend
        self._app = None
        if backend == "insightface":
            self._init_insightface()

    def _init_insightface(self):
        from insightface.app import FaceAnalysis  # lazy: only on the training box

        self._app = FaceAnalysis(
            name="buffalo_l", allowed_modules=["detection", "landmark_2d_106"]
        )
        # ctx_id=-1 -> CPU providers (Mac has no CUDA onnxruntime); FK is cheap one-shot.
        self._app.prepare(ctx_id=-1, det_size=(256, 256))

    def __call__(self, image):
        """image (bs,H,W,3) in [0,1] -> (bs,num_fk,2) normalized to [-1,1]."""
        bs = image.shape[0]
        if self.backend == "stub":
            # torch: g=linspace(-0.8,0.8,num_fk); stack([g, g.flip(0)]).
            # g.flip(0) of an ascending linspace == descending linspace (exact).
            g = mx.linspace(-0.8, 0.8, self.num_fk)
            g_flip = mx.linspace(0.8, -0.8, self.num_fk)
            pts = mx.stack([g, g_flip], axis=-1)  # (num_fk,2) placeholder, NOT real
            return mx.broadcast_to(pts[None], (bs, self.num_fk, 2))
        # insightface path (image NHWC RGB in [0,1])
        import cv2
        import numpy as np

        arr = (np.clip(np.array(image), 0, 1) * 255).astype("uint8")  # bs,H,W,3 RGB
        H, W = arr.shape[1:3]
        out = np.zeros((bs, self.num_fk, 2), dtype="float32")
        for i in range(bs):
            faces = self._app.get(cv2.cvtColor(arr[i], cv2.COLOR_RGB2BGR))
            if faces:
                f = max(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                )
                lmk = np.asarray(f.landmark_2d_106, dtype="float32")  # (106,2) pixels
                out[i] = lmk / np.array([W, H], dtype="float32") * 2 - 1
        return mx.array(out)


class MixedKPDetector(nn.Module):
    """Δ1 drop-in replacement for KPDetector: NK + frozen FK + MixedKP MLP."""

    def __init__(self, num_tps, fk_backend="stub", num_fk=106):
        super().__init__()
        self.num_tps = num_tps
        self.nk = KPDetector(num_tps)
        self.fk = FKDetector(backend=fk_backend, num_fk=num_fk)
        self.mixed = MixedKP(num_fk=num_fk, num_nk=num_tps * 5, num_mixed=num_tps * 5)

    def __call__(self, image):
        # image: NHWC
        nk_kp = self.nk(image)["fg_kp"]  # (bs, num_tps*5, 2)  learned
        fk_kp = self.fk(image)  # (bs, num_fk, 2)     frozen
        mixed_kp = self.mixed(fk_kp, nk_kp)  # (bs, num_tps*5, 2)
        return {"fg_kp": mixed_kp, "nk_kp": nk_kp, "fk_kp": fk_kp}


def load_mixedkp_from_torch(mlx_model, torch_state_dict):
    """Transfer a torch MixedKPDetector state_dict into the MLX MixedKPDetector.

    torch keys: 'nk.fg_encoder.*' (resnet18), 'mixed.mlp.{0,2,4}.{weight,bias}'.
    The FK detector has no params. Conv weights are transposed; Linear weights (out,in)
    transfer directly; the Sequential mlp indices 0/2/4 collapse to list indices 0/1/2.
    """
    flat = []
    for k, v in torch_state_dict.items():
        if k.endswith("num_batches_tracked"):
            continue
        m = re.match(r"(mixed\.mlp\.)(\d+)(\..*)$", k)
        if m:
            k = f"{m.group(1)}{int(m.group(2)) // 2}{m.group(3)}"
        arr = mx.array(v.detach().cpu().numpy())
        if arr.ndim == 4:  # conv weight (out,in,kH,kW) -> (out,kH,kW,in)
            arr = mx.transpose(arr, (0, 2, 3, 1))
        flat.append((k, arr))
    mlx_model.unfreeze(recurse=True)
    mlx_model.update(tree_unflatten(flat))
    mlx_model.eval()
    return mlx_model

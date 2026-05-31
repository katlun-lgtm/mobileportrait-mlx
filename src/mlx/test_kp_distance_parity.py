"""Parity: MobilePortrait Δ3 kp_distance loss — torch vs MLX.

Builds a torch and an MLX MixedKPDetector with identical weights, runs the same image
through both, then compares the raw kp_distance value:
    torch: |mk[:, :n] - fk[:, :n]|.mean()   (mk=fg_kp, fk=fk_kp, n=min counts)
    mlx:   losses_mlx.kp_distance_loss(kp_driving)
Uses the deterministic FK "stub" backend so no insightface is needed.

Run on the Mac:
    cd ~/mobileportrait-mlx && ~/lp-mlx/.venv/bin/python src/mlx/test_kp_distance_parity.py
"""

import importlib
import os
import sys

import numpy as np
import torch

import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
# ref uses relative imports (from .fk_detector) -> import as the `modules` package
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)

ref = importlib.import_module("modules.keypoint_detector")
TorchMKD = ref.MixedKPDetector

from mixed_kp_mlx import MixedKPDetector as MlxMKD  # noqa: E402
from mixed_kp_mlx import load_mixedkp_from_torch  # noqa: E402
from losses_mlx import kp_distance_loss  # noqa: E402

NUM_TPS = 10
BS = 2
H = W = 256
TOL = 5e-3

torch.manual_seed(0)
np.random.seed(0)

tmodel = TorchMKD(num_tps=NUM_TPS, fk_backend="stub").eval()
mmodel = MlxMKD(num_tps=NUM_TPS, fk_backend="stub")
load_mixedkp_from_torch(mmodel, tmodel.state_dict())

img = np.random.rand(BS, 3, H, W).astype("float32")  # NCHW [0,1]
with torch.no_grad():
    tkp = tmodel(torch.from_numpy(img))
    fk, mk = tkp["fk_kp"], tkp["fg_kp"]
    n = min(fk.shape[1], mk.shape[1])
    t_val = float(torch.abs(mk[:, :n] - fk[:, :n]).mean())

img_nhwc = np.ascontiguousarray(np.transpose(img, (0, 2, 3, 1)))
mkp = mmodel(mx.array(img_nhwc))
m_val = float(kp_distance_loss(mkp))

diff = abs(t_val - m_val)
rel = diff / (abs(t_val) + 1e-12)
ok = diff < TOL
print(
    f"kp_distance  torch={t_val:.6f}  mlx={m_val:.6f}  absdiff={diff:.3e}  rel={rel:.3e}"
)
print(f"n_points compared = {n}")
print(f"FAILS={0 if ok else 1}")
print("RESULT_ALL_PASS" if ok else "RESULT_FAIL")

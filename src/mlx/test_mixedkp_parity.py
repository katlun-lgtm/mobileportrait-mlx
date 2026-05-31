"""Parity test: torch MixedKPDetector (Δ1) vs MLX MixedKPDetector.

Loads identical weights (torch state_dict -> MLX), runs the same random image through
both, and compares fg_kp (fused), nk_kp (learned) and fk_kp (frozen) element-wise.
fk_kp uses the deterministic "stub" backend so the test needs no insightface.

Run on the Mac (MLX + torch + torchvision):
    cd ~/mobileportrait-mlx && ~/lp-mlx/.venv/bin/python src/mlx/test_mixedkp_parity.py
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
    tout = tmodel(torch.from_numpy(img))

img_nhwc = np.ascontiguousarray(np.transpose(img, (0, 2, 3, 1)))
mout = mmodel(mx.array(img_nhwc))
mx.eval(mout["fg_kp"], mout["nk_kp"], mout["fk_kp"])

fails = 0
for key in ["fg_kp", "nk_kp", "fk_kp"]:
    t = tout[key].numpy()
    m = np.array(mout[key])
    if t.shape != m.shape:
        print(f"FAIL {key:8s} shape torch={t.shape} mlx={m.shape}")
        fails += 1
        continue
    md = float(np.abs(t - m).max())
    mn = float(np.abs(t - m).mean())
    ok = md < TOL
    print(
        f"{'PASS' if ok else 'FAIL'} {key:8s} maxdiff={md:.3e} mean={mn:.3e} tol={TOL:.0e}"
    )
    if not ok:
        fails += 1

print(f"FAILS={fails}")
print("RESULT_ALL_PASS" if fails == 0 else "RESULT_FAIL")

"""Parity: src/mlx/keypoint_detector_mlx.py vs PyTorch src/modules/keypoint_detector.py::KPDetector.

Loads identical (random-init) torch weights into the MLX model and compares fg_kp.
Run on the Mac:
  cd ~/mobileportrait-mlx && ~/lp-mlx/.venv/bin/python src/mlx/test_keypoint_detector_mlx_parity.py
"""

import importlib.util
import os
import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
REF = os.path.join(REPO, "src", "modules", "keypoint_detector.py")

# the reference imports `from .fk_detector import FKDetector` etc; load it as part of
# the `modules` package by putting src on path and importing modules.keypoint_detector.
import sys

sys.path.insert(0, os.path.join(REPO, "src"))
import torch

ref = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("ref_kpd", REF)
)
# ref uses relative imports (from .fk_detector) -> import via package instead
import importlib

ref = importlib.import_module("modules.keypoint_detector")

import keypoint_detector_mlx as M  # same dir on path when run as script

NUM_TPS = 10
BS = 2
np.random.seed(0)
torch.manual_seed(0)

# torch reference
t_model = ref.KPDetector(NUM_TPS)
t_model.eval()
sd = t_model.state_dict()

# mlx port + weight transfer
m_model = M.KPDetector(NUM_TPS)
M.load_kpdetector_from_torch(m_model, sd)

# same input: torch NCHW, mlx NHWC
img = np.random.randn(BS, 3, 256, 256).astype(np.float32)
with torch.no_grad():
    t_out = t_model(torch.tensor(img))["fg_kp"].cpu().numpy()
m_out = np.array(m_model(mx.array(img.transpose(0, 2, 3, 1)))["fg_kp"])

print("shapes", t_out.shape, m_out.shape)
fails = 0
if t_out.shape != m_out.shape:
    print("FAIL shape mismatch")
    fails += 1
else:
    d = float(np.max(np.abs(t_out - m_out)))
    tol = 2e-3
    ok = d <= tol
    print(f"{'PASS' if ok else 'FAIL'}  fg_kp  maxdiff={d:.3e}  tol={tol:.0e}")
    if not ok:
        fails += 1
    # also sanity: outputs in [-1,1]
    print(f"range mlx [{m_out.min():.3f}, {m_out.max():.3f}]")

print(f"\nFAILS={fails}")
print("ALL_PASS" if fails == 0 else "ANY_FAIL")

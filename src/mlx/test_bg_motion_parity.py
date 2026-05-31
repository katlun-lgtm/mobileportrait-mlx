"""Parity: src/mlx/bg_motion_predictor_mlx.py vs torch BGMotionPredictor.

Loads identical torch weights into the MLX model and compares the (bs,3,3) affine on a
random source+driving pair. Run on the Mac:
    cd ~/mobileportrait-mlx && ~/lp-mlx/.venv/bin/python src/mlx/test_bg_motion_parity.py
"""

import importlib
import os
import sys

import numpy as np
import torch

import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)

ref = importlib.import_module("modules.bg_motion_predictor")
from bg_motion_predictor_mlx import BGMotionPredictor as MlxBG  # noqa: E402
from bg_motion_predictor_mlx import load_bg_from_torch  # noqa: E402

BS = 2
H = W = 64
TOL = 5e-3

torch.manual_seed(0)
np.random.seed(0)

tmodel = ref.BGMotionPredictor().eval()
# randomise fc so parity exercises a real prediction (torch inits it to identity)
torch.nn.init.normal_(tmodel.bg_encoder.fc.weight, std=0.01)
mmodel = MlxBG()
miss, unexp = load_bg_from_torch(mmodel, tmodel.state_dict())
print(f"KEYDRIFT missing={len(miss)} unexpected={len(unexp)}")
if miss:
    print("  missing:", miss[:6])
if unexp:
    print("  unexpected:", unexp[:6])

src = np.random.rand(BS, 3, H, W).astype("float32")
drv = np.random.rand(BS, 3, H, W).astype("float32")
with torch.no_grad():
    tout = tmodel(torch.from_numpy(src), torch.from_numpy(drv)).numpy()

src_nhwc = np.ascontiguousarray(np.transpose(src, (0, 2, 3, 1)))
drv_nhwc = np.ascontiguousarray(np.transpose(drv, (0, 2, 3, 1)))
mout = mmodel(mx.array(src_nhwc), mx.array(drv_nhwc))
mx.eval(mout)
m = np.array(mout)

fails = 0
if m.shape != tout.shape:
    print(f"FAIL shape torch={tout.shape} mlx={m.shape}")
    fails += 1
else:
    md = float(np.abs(tout - m).max())
    mn = float(np.abs(tout - m).mean())
    ok = md < TOL
    print(
        f"{'PASS' if ok else 'FAIL'} affine maxdiff={md:.3e} mean={mn:.3e} tol={TOL:.0e}"
    )
    if not ok:
        fails += 1
    print("bottom row mlx[0]:", m[0, 2].tolist(), "(expect [0,0,1])")

print(f"FAILS={fails}")
print("RESULT_ALL_PASS" if fails == 0 else "RESULT_FAIL")

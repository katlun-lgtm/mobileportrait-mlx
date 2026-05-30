"""Parity: src/mlx/inpainting_network_mlx.py vs PyTorch src/modules/inpainting_network.py.

Builds torch DenseMotionNetwork to produce a real dense_motion dict, then runs both the
torch and MLX InpaintingNetwork (identical transferred weights, eval) on the same source +
dense_motion and compares prediction / deformed.
Run on the Mac:
  cd ~/mobileportrait-mlx && ~/lp-mlx/.venv/bin/python src/mlx/test_inpainting_network_mlx_parity.py
"""

import os
import sys
import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)
import torch
import importlib

ref_inp = importlib.import_module("modules.inpainting_network")
ref_dm = importlib.import_module("modules.dense_motion")
import inpainting_network_mlx as M

np.random.seed(0)
torch.manual_seed(0)
DM_CFG = dict(
    block_expansion=64,
    num_blocks=5,
    max_features=1024,
    num_tps=10,
    num_channels=3,
    scale_factor=0.25,
    bg=False,
    multi_mask=True,
    kp_variance=0.01,
)
INP_CFG = dict(
    num_channels=3,
    block_expansion=64,
    max_features=512,
    num_down_blocks=3,
    multi_mask=True,
)
BS = 2
fails = 0


def fail(m):
    global fails
    fails += 1
    print("FAIL " + m)


# torch dense_motion -> real dict
t_dm = ref_dm.DenseMotionNetwork(**DM_CFG).eval()
src = np.random.randn(BS, 3, 256, 256).astype(np.float32)
kp_d = (np.random.rand(BS, 50, 2) * 2 - 1).astype(np.float32)
kp_s = (np.random.rand(BS, 50, 2) * 2 - 1).astype(np.float32)
with torch.no_grad():
    dm_t = t_dm(
        torch.tensor(src), {"fg_kp": torch.tensor(kp_d)}, {"fg_kp": torch.tensor(kp_s)}
    )

# torch + mlx inpainting, identical weights
t_inp = ref_inp.InpaintingNetwork(**INP_CFG).eval()
m_inp = M.InpaintingNetwork(**INP_CFG)
missing, unexpected = M.load_inpaint_from_torch(m_inp, t_inp.state_dict())
print(f"KEYDRIFT(info) missing={len(missing)} unexpected={len(unexpected)}")

with torch.no_grad():
    out_t = t_inp(torch.tensor(src), dm_t)

# convert dm dict to MLX NHWC
dm_m = {
    "deformation": mx.array(dm_t["deformation"].numpy()),  # (B,h,w,2) already
    "contribution_maps": mx.array(
        dm_t["contribution_maps"].numpy().transpose(0, 2, 3, 1)
    ),
    "deformed_source": mx.array(
        dm_t["deformed_source"].numpy().transpose(0, 1, 3, 4, 2)
    ),
    "occlusion_map": [
        mx.array(o.numpy().transpose(0, 2, 3, 1)) for o in dm_t["occlusion_map"]
    ],
}
out_m = m_inp(mx.array(src.transpose(0, 2, 3, 1)), dm_m)


def cmp(name, m_arr, t_np_nchw, tol):
    global fails
    a = np.array(m_arr)
    b = t_np_nchw.numpy().transpose(0, 2, 3, 1)  # NCHW->NHWC
    if a.shape != b.shape:
        fail(f"{name} shape mlx={a.shape} torch={b.shape}")
        return
    d = float(np.max(np.abs(a - b)))
    mean = float(np.mean(np.abs(a - b)))
    ok = d <= tol
    print(
        f"{'PASS' if ok else 'FAIL'}  {name:14s} maxdiff={d:.3e} mean={mean:.3e} tol={tol:.0e}"
    )
    if not ok:
        fails += 1


cmp("prediction", out_m["prediction"], out_t["prediction"], 5e-3)
cmp("deformed", out_m["deformed"], out_t["deformed"], 5e-3)

print(f"\nFAILS={fails}")
print("RESULT_ALL_PASS" if fails == 0 else "RESULT_ANY_FAIL")

"""Parity: src/mlx/dense_motion_mlx.py vs PyTorch src/modules/dense_motion.py.

Transfers identical torch weights into the MLX module, runs both (eval) on the same
input, compares deformed_source / contribution_maps / deformation / occlusion pyramid.

NOTE on key-drift: comparing mlx parameters() keys vs torch state_dict() keys yields a
spurious result for list-submodules (MLX list-flatten vs torch ModuleList enumeration);
it is reported INFORMATIONALLY only. The real correctness gate is the numerical tensor
parity below — exact transfer is proven by the per-tensor maxdiffs (an unloaded weight
would show O(1) diff, not ~1e-6).

Run on the Mac:
  cd ~/mobileportrait-mlx && ~/lp-mlx/.venv/bin/python src/mlx/test_dense_motion_mlx_parity.py
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

ref = importlib.import_module("modules.dense_motion")
import dense_motion_mlx as M

np.random.seed(0)
torch.manual_seed(0)
CFG = dict(
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
BS = 2
fails = 0


def fail(msg):
    global fails
    fails += 1
    print("FAIL " + msg)


t_dm = ref.DenseMotionNetwork(**CFG).eval()
m_dm = M.DenseMotionNetwork(**CFG)
missing, unexpected = M.load_dm_from_torch(m_dm, t_dm.state_dict())
# informational only — see module docstring; does NOT gate.
print(
    f"KEYDRIFT(info) missing(mlx-only)={len(missing)} unexpected(torch-only)={len(unexpected)}"
)

src = np.random.randn(BS, 3, 256, 256).astype(np.float32)
kp_d = (np.random.rand(BS, 50, 2) * 2 - 1).astype(np.float32)
kp_s = (np.random.rand(BS, 50, 2) * 2 - 1).astype(np.float32)

with torch.no_grad():
    t_out = t_dm(
        torch.tensor(src), {"fg_kp": torch.tensor(kp_d)}, {"fg_kp": torch.tensor(kp_s)}
    )
m_out = m_dm(
    mx.array(src.transpose(0, 2, 3, 1)),
    {"fg_kp": mx.array(kp_d)},
    {"fg_kp": mx.array(kp_s)},
)


def cmp(name, m_arr, t_np_nhwc, tol):
    global fails
    a = np.array(m_arr)
    b = t_np_nhwc
    if a.shape != b.shape:
        fail(f"{name} shape mlx={a.shape} torch={b.shape}")
        return
    d = float(np.max(np.abs(a - b)))
    ok = d <= tol
    print(f"{'PASS' if ok else 'FAIL'}  {name:26s} maxdiff={d:.3e} tol={tol:.0e}")
    if not ok:
        fails += 1


cmp("deformation", m_out["deformation"], t_out["deformation"].numpy(), 5e-3)
cmp(
    "contribution_maps",
    m_out["contribution_maps"],
    t_out["contribution_maps"].numpy().transpose(0, 2, 3, 1),
    2e-3,
)
cmp(
    "deformed_source",
    m_out["deformed_source"],
    t_out["deformed_source"].numpy().transpose(0, 1, 3, 4, 2),
    5e-3,
)
to = t_out["occlusion_map"]
mo = m_out["occlusion_map"]
if len(to) != len(mo):
    fail(f"occlusion len mlx={len(mo)} torch={len(to)}")
else:
    print(f"occlusion pyramid len={len(to)}")
    for i, (tt, mm) in enumerate(zip(to, mo)):
        cmp(f"occlusion[{i}]", mm, tt.numpy().transpose(0, 2, 3, 1), 5e-3)

print(f"\nFAILS={fails}")
print("RESULT_ALL_PASS" if fails == 0 else "RESULT_ANY_FAIL")

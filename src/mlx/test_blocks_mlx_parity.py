"""Parity: src/mlx/blocks_mlx.py vs reference src/modules/util.py blocks.
Run on the Mac: cd ~/mobileportrait-mlx && ~/lp-mlx/.venv/bin/python src/mlx/test_blocks_mlx_parity.py
"""

import os
import sys
import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
import torch
import importlib

ref = importlib.import_module("modules.util")  # reference blocks
import blocks_mlx as B

np.random.seed(0)
torch.manual_seed(0)
fails = 0


def nchw_to_nhwc(a):
    return np.ascontiguousarray(a.transpose(0, 2, 3, 1))


def cmp(name, m_out_nhwc, t_out_nchw, tol):
    global fails
    a = np.array(m_out_nhwc)
    b = t_out_nchw.detach().cpu().numpy().transpose(0, 2, 3, 1)
    if a.shape != b.shape:
        print(f"FAIL  {name:28s} shape mlx={a.shape} torch={b.shape}")
        fails += 1
        return
    d = float(np.max(np.abs(a - b)))
    ok = d <= tol
    print(f"{'PASS' if ok else 'FAIL'}  {name:28s} maxdiff={d:.3e} tol={tol:.0e}")
    if not ok:
        fails += 1


def xfer(mlx_mod, tmod):
    B.load_state_from_torch(mlx_mod, tmod.state_dict())


x = np.random.randn(2, 32, 64, 64).astype(np.float32)
xt = torch.tensor(x)
xm = mx.array(nchw_to_nhwc(x))

# ResBlock2d
t = ref.ResBlock2d(32, kernel_size=3, padding=1).eval()
m = B.ResBlock2d(32, 3, 1)
xfer(m, t)
cmp("ResBlock2d", m(xm), t(xt), 2e-4)

# SameBlock2d
t = ref.SameBlock2d(32, 48, kernel_size=3, padding=1).eval()
m = B.SameBlock2d(32, 48, kernel_size=3, padding=1)
xfer(m, t)
cmp("SameBlock2d", m(xm), t(xt), 2e-4)

# DownBlock2d
t = ref.DownBlock2d(32, 48, kernel_size=3, padding=1).eval()
m = B.DownBlock2d(32, 48, kernel_size=3, padding=1)
xfer(m, t)
cmp("DownBlock2d", m(xm), t(xt), 2e-4)

# UpBlock2d
t = ref.UpBlock2d(32, 48, kernel_size=3, padding=1).eval()
m = B.UpBlock2d(32, 48, kernel_size=3, padding=1)
xfer(m, t)
cmp("UpBlock2d", m(xm), t(xt), 2e-4)

# Hourglass (block_expansion=16, in=32, num_blocks=3, max=256)
t = ref.Hourglass(16, 32, num_blocks=3, max_features=256).eval()
m = B.Hourglass(16, 32, num_blocks=3, max_features=256)
xfer(m, t)
cmp("Hourglass mode0", m(xm, mode=0), t(xt, mode=0), 1e-3)
print("  hourglass out_channels mlx=", m.out_channels, "torch=", t.out_channels)

# AntiAliasInterpolation2d (no weights to transfer; fixed gaussian)
t = ref.AntiAliasInterpolation2d(32, 0.25).eval()
m = B.AntiAliasInterpolation2d(32, 0.25)
cmp("AntiAlias 0.25", m(xm), t(xt), 2e-4)

print(f"\nFAILS={fails}")
print("ALL_PASS" if fails == 0 else "ANY_FAIL")

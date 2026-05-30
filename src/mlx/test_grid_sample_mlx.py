"""Validate the differentiable MLX grid_sample (fwd + d_x + d_grid) against PyTorch.

Run on the Mac: ~/lp-mlx/.venv/bin/python src/mlx/test_grid_sample_mlx.py
Requires torch (2.12) + mlx (0.31.2) in the venv. float32 only (atomic float add).

Checks, for align_corners in {False,True} x padding_mode in {zeros,border}:
  - forward matches torch.nn.functional.grid_sample (max abs diff)
  - d_x  matches torch autograd grad wrt x
  - d_grid matches torch autograd grad wrt grid
under a random cotangent (so the VJP is exercised, not just sum()).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mlx.core as mx
import torch
import torch.nn.functional as F
from grid_sample_mlx import make_grid_sample_2d, VALIDATED_MLX_VERSION

print(
    "mlx",
    mx.__version__,
    "(validated against",
    VALIDATED_MLX_VERSION + ")",
    "torch",
    torch.__version__,
)

N, H, W, C, Ho, Wo = 2, 12, 10, 5, 9, 7
rng = np.random.RandomState(0)
x_np = rng.randn(N, H, W, C).astype("float32")
grid_np = (
    rng.rand(N, Ho, Wo, 2).astype("float32") * 2.4 - 1.2
)  # include some out-of-bounds
cot_np = rng.randn(N, Ho, Wo, C).astype("float32")


def torch_run(ac, pm):
    xt = torch.tensor(x_np.transpose(0, 3, 1, 2), requires_grad=True)  # NCHW
    gt = torch.tensor(grid_np, requires_grad=True)
    out = F.grid_sample(
        xt, gt, mode="bilinear", padding_mode=pm, align_corners=ac
    )  # (N,C,Ho,Wo)
    cot = torch.tensor(cot_np.transpose(0, 3, 1, 2))
    out.backward(cot)
    return (
        out.detach().numpy().transpose(0, 2, 3, 1),  # -> (N,Ho,Wo,C)
        xt.grad.numpy().transpose(0, 2, 3, 1),  # -> (N,H,W,C)
        gt.grad.numpy(),
    )


def mlx_run(ac, pm):
    gs = make_grid_sample_2d(align_corners=ac, padding_mode=pm)
    x = mx.array(x_np)
    grid = mx.array(grid_np)
    cot = mx.array(cot_np)
    out = gs(x, grid)
    mx.eval(out)
    # VJP via vjp transform with our cotangent
    _, vjps = mx.vjp(lambda xx, gg: gs(xx, gg), (x, grid), (cot,))
    dx, dgrid = vjps
    mx.eval(dx, dgrid)
    return np.array(out), np.array(dx), np.array(dgrid)


worst = 0.0
fails = []
for ac in (False, True):
    for pm in ("zeros", "border"):
        to, tdx, tdg = torch_run(ac, pm)
        mo, mdx, mdg = mlx_run(ac, pm)
        df = float(np.abs(to - mo).max())
        ddx = float(np.abs(tdx - mdx).max())
        ddg = float(np.abs(tdg - mdg).max())
        worst = max(worst, df, ddx, ddg)
        tag = f"ac={ac} pm={pm}"
        print(f"{tag:18s} fwd={df:.2e} d_x={ddx:.2e} d_grid={ddg:.2e}")
        for nm, v in (("fwd", df), ("d_x", ddx), ("d_grid", ddg)):
            if v > 1e-3:
                fails.append(f"{tag} {nm}={v:.2e}")

print("WORST", f"{worst:.2e}")
if fails:
    print("FAIL:", "; ".join(fails))
    sys.exit(1)
print("GRID_SAMPLE_MLX OK — fwd+d_x+d_grid match PyTorch within 1e-3")

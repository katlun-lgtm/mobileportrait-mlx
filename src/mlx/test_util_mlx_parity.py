"""Parity: src/mlx/util_mlx.py vs the PyTorch reference src/modules/util.py.

Run on the Mac:
  cd ~/mobileportrait-mlx && ~/lp-mlx/.venv/bin/python src/mlx/test_util_mlx_parity.py
Prints PASS/FAIL per check + a final ALL_PASS / ANY_FAIL line. READ the output.
"""

import importlib.util
import os
import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))  # .../mobileportrait-mlx
REF = os.path.join(REPO, "src", "modules", "util.py")

spec = importlib.util.spec_from_file_location("ref_util", REF)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)

import torch

import util_mlx as M  # same dir (src/mlx) on path when run as a script

rng = np.random.default_rng(0)
fails = 0


def cmp(name, mlx_arr, torch_arr, tol):
    global fails
    a = np.array(mlx_arr)
    b = (
        torch_arr.detach().cpu().numpy()
        if hasattr(torch_arr, "detach")
        else np.asarray(torch_arr)
    )
    if a.shape != b.shape:
        print(f"FAIL  {name:34s} shape mlx={a.shape} torch={b.shape}")
        fails += 1
        return
    d = float(np.max(np.abs(a - b))) if a.size else 0.0
    ok = d <= tol
    print(f"{'PASS' if ok else 'FAIL'}  {name:34s} maxdiff={d:.3e} tol={tol:.0e}")
    if not ok:
        fails += 1


# 1) make_coordinate_grid
for hw in [(8, 8), (5, 7), (64, 64)]:
    g_mlx = M.make_coordinate_grid(hw)
    g_t = ref.make_coordinate_grid(hw, torch.float32)
    cmp(f"coord_grid {hw}", g_mlx, g_t, 1e-6)

# 2) to/from homogeneous
c = rng.standard_normal((4, 6, 2)).astype(np.float32)
cmp(
    "to_homogeneous",
    M.to_homogeneous(mx.array(c)),
    ref.to_homogeneous(torch.tensor(c)),
    1e-6,
)
ch = rng.standard_normal((4, 6, 3)).astype(np.float32)
cmp(
    "from_homogeneous",
    M.from_homogeneous(mx.array(ch)),
    ref.from_homogeneous(torch.tensor(ch)),
    1e-6,
)

# 3) kp2gaussian — kp shape (bs, k, 2)
kp = (rng.standard_normal((2, 10, 2)) * 0.5).astype(np.float32)
cmp(
    "kp2gaussian",
    M.kp2gaussian(mx.array(kp), (64, 64), 0.01),
    ref.kp2gaussian(torch.tensor(kp), (64, 64), 0.01),
    1e-4,
)

# 4) TPS mode 'kp' — kp_1, kp_2 shape (bs, gs, n, 2)
bs, gs, n = 2, 11, 5  # gs = num_tps+1, n = points per tps
kp1 = (rng.standard_normal((bs, gs, n, 2)) * 0.4).astype(np.float32)
kp2 = (rng.standard_normal((bs, gs, n, 2)) * 0.4).astype(np.float32)
tps_m = M.TPS("kp", bs, kp_1=mx.array(kp1), kp_2=mx.array(kp2))
tps_t = ref.TPS("kp", bs, kp_1=torch.tensor(kp1), kp_2=torch.tensor(kp2))
cmp("TPS.kp theta", tps_m.theta, tps_t.theta, 1e-3)
cmp("TPS.kp control_params", tps_m.control_params, tps_t.control_params, 1e-3)
# transform a 64x64 frame
frame_shape = (bs, 3, 64, 64)
fr_t = torch.zeros(frame_shape)
fr_m = mx.zeros(frame_shape)
cmp(
    "TPS.kp transform_frame",
    tps_m.transform_frame(fr_m),
    tps_t.transform_frame(fr_t),
    2e-3,
)

# 5) TPS mode 'random' — copy torch's sampled params into MLX, compare warp math only
torch.manual_seed(0)
tps_tr = ref.TPS("random", bs, sigma_affine=0.05, sigma_tps=0.005, points_tps=5)
tps_mr = M.TPS("random", bs, sigma_affine=0.05, sigma_tps=0.005, points_tps=5)
# overwrite MLX-sampled params with torch's exact ones
tps_mr.theta = mx.array(tps_tr.theta.detach().cpu().numpy())
tps_mr.control_points = mx.array(tps_tr.control_points.detach().cpu().numpy())
tps_mr.control_params = mx.array(tps_tr.control_params.detach().cpu().numpy())
coords = (rng.standard_normal((bs, 100, 2)) * 0.5).astype(np.float32)
cmp(
    "TPS.random warp_coordinates",
    tps_mr.warp_coordinates(mx.array(coords)),
    tps_tr.warp_coordinates(torch.tensor(coords)),
    1e-4,
)

print(f"\nFAILS={fails}")
print("ALL_PASS" if fails == 0 else "ANY_FAIL")

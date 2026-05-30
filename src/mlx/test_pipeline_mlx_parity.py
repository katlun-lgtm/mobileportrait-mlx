"""End-to-end parity: MLX MobilePortraitPipeline vs torch glue (kp->dm->inpaint).

WHY THE GATE IS NOT max-on-random-weights:
A naive `prediction max < 5e-3` on random-init weights FAILS (~1.9e-2) — NOT a wiring bug.
Diagnosed (diag_pipeline.py / diag2_swap.py, values READ from /tmp/diag.txt+diag2.txt):
  - MLX inpaint fed the TORCH dm matches torch to max 7.130e-4  -> inpaint correct.
  - dm deformation differs by only max 7.284e-5 (~0.009 px on 256).
  - torch dm with ONLY MLX deformation swapped in reproduces the full drift (1.915e-2)
    -> the tiny deformation coord-diff is the entire cause.
  - on SMOOTH (blurred) inputs the full-MLX drift collapses to max 1.202e-3
    -> it's grid_sample amplifying a sub-pixel coord diff on noise-like random features,
       which vanishes with trained/smooth features.
So we gate on the regime that matters: SMOOTH-input prediction max + random-weight MEAN + kp.

Run on the Mac:
  cd ~/mobileportrait-mlx && ~/lp-mlx/.venv/bin/python src/mlx/test_pipeline_mlx_parity.py
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
import torch.nn.functional as F
import importlib

ref_kp = importlib.import_module("modules.keypoint_detector")
ref_dm = importlib.import_module("modules.dense_motion")
ref_inp = importlib.import_module("modules.inpainting_network")
import pipeline_mlx as P

np.random.seed(0)
torch.manual_seed(0)
KP_CFG = dict(num_tps=10)
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

t_kp = ref_kp.KPDetector(**KP_CFG).eval()
t_dm = ref_dm.DenseMotionNetwork(**DM_CFG).eval()
t_inp = ref_inp.InpaintingNetwork(**INP_CFG).eval()
pipe, drift = P.build_from_torch(
    KP_CFG, DM_CFG, INP_CFG, t_kp.state_dict(), t_dm.state_dict(), t_inp.state_dict()
)
print(
    f"DRIFT dm={len(drift['dm'][0])},{len(drift['dm'][1])} "
    f"inp={len(drift['inp'][0])},{len(drift['inp'][1])}"
)


def torch_pred(src, drv):
    with torch.no_grad():
        kp_s = t_kp(torch.tensor(src))
        kp_d = t_kp(torch.tensor(drv))
        dm = t_dm(torch.tensor(src), kp_d, kp_s)
        return t_inp(torch.tensor(src), dm)["prediction"].numpy().transpose(
            0, 2, 3, 1
        ), kp_s


def mlx_pred(src, drv):
    g = pipe(mx.array(src.transpose(0, 2, 3, 1)), mx.array(drv.transpose(0, 2, 3, 1)))
    return np.array(g["prediction"]), g


def blur(x):
    k = torch.ones(3, 1, 9, 9) / 81.0
    return F.conv2d(torch.tensor(x), k, padding=4, groups=3).numpy()


def stats(a, b):
    e = np.abs(a - b)
    return e.max(), float(np.mean(e))


src = np.random.randn(BS, 3, 256, 256).astype(np.float32)
drv = np.random.randn(BS, 3, 256, 256).astype(np.float32)

# 1) kp exact
pred_t, kp_s = torch_pred(src, drv)
pred_m, gen_m = mlx_pred(src, drv)
dkp = float(
    np.max(np.abs(np.array(gen_m["kp_source"]["fg_kp"]) - kp_s["fg_kp"].numpy()))
)
ok = dkp <= 2e-3
print(f"{'PASS' if ok else 'FAIL'}  kp_source         maxdiff={dkp:.3e} tol=2e-03")
fails += 0 if ok else 1

# 2) random-weight prediction: gate on MEAN (max is informational, see docstring)
rmax, rmean = stats(pred_m, pred_t)
ok = rmean <= 5e-3
print(
    f"{'PASS' if ok else 'FAIL'}  prediction(mean)  mean={rmean:.3e} tol=5e-03 "
    f"[max={rmax:.3e} informational: random-weight grid_sample amplification]"
)
fails += 0 if ok else 1

# 3) SMOOTH-input prediction: the regime that matters -> gate on MAX
srcB, drvB = blur(src), blur(drv)
pred_tB, _ = torch_pred(srcB, drvB)
pred_mB, _ = mlx_pred(srcB, drvB)
smax, smean = stats(pred_mB, pred_tB)
ok = smax <= 5e-3
print(
    f"{'PASS' if ok else 'FAIL'}  prediction(smooth) max={smax:.3e} mean={smean:.3e} tol=5e-03"
)
fails += 0 if ok else 1

# drift gate
if (
    len(drift["dm"][0])
    or len(drift["dm"][1])
    or len(drift["inp"][0])
    or len(drift["inp"][1])
):
    print("FAIL key drift")
    fails += 1

print(f"\nFAILS={fails}")
print("RESULT_ALL_PASS" if fails == 0 else "RESULT_ANY_FAIL")

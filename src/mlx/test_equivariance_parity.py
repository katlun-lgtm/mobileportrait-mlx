"""Parity: losses_mlx.equivariance_loss vs the torch model.py equivariance_value term.

Identical kp-detector weights in both stacks; identical random-TPS params injected into
both (torch samples noise+control_params internally, we read them out and feed them to the
MLX TPS via _noise/_control_params). Compare the scalar loss.

Run on the Mac:
  cd ~/mobileportrait-mlx && ~/lp-mlx/.venv/bin/python src/mlx/test_equivariance_parity.py
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
ref_util = importlib.import_module("modules.util")
import keypoint_detector_mlx as MKP
import losses_mlx as ML

np.random.seed(0)
torch.manual_seed(0)
TP = dict(sigma_affine=0.05, sigma_tps=0.005, points_tps=5)
BS = 2
fails = 0

# torch kp detector + identical MLX kp detector
t_kp = ref_kp.KPDetector(num_tps=10).eval()
m_kp = MKP.KPDetector(num_tps=10)
MKP.load_kpdetector_from_torch(m_kp, t_kp.state_dict())

drv = np.random.rand(BS, 3, 256, 256).astype(np.float32)  # NCHW for torch
drv_t = torch.tensor(drv)
drv_m = mx.array(drv.transpose(0, 2, 3, 1))  # NHWC for mlx

# torch kp_driving
with torch.no_grad():
    kp_d_t = t_kp(drv_t)

# Build torch random TPS, capture its sampled params
torch.manual_seed(123)
tr = ref_util.TPS(mode="random", bs=BS, **TP)
# torch TPS random stores: self.theta (bs,2,3) = noise+eye, self.control_params (bs,1,P^2)
noise_t = (
    (tr.theta - torch.eye(2, 3).view(1, 2, 3)).detach().cpu().numpy()
)  # recover noise
cparams_t = tr.control_params.detach().cpu().numpy()

# torch equivariance value (model.py 189-205)
with torch.no_grad():
    grid = tr.transform_frame(drv_t)
    tf = F.grid_sample(drv_t, grid, padding_mode="reflection", align_corners=True)
    tkp = t_kp(tf)
    warped = tr.warp_coordinates(tkp["fg_kp"])
    t_loss = float(torch.abs(kp_d_t["fg_kp"] - warped).mean())

# mlx: inject the SAME noise + control_params so TPS math matches
kp_d_m = m_kp(drv_m)
m_loss = float(
    ML.equivariance_loss(
        m_kp,
        drv_m,
        kp_d_m,
        TP,
        noise=mx.array(noise_t),
        control_params=mx.array(cparams_t),
    )
)

rel = abs(m_loss - t_loss) / (abs(t_loss) + 1e-9)
# NOTE: border (mlx) vs reflection (torch) padding differ in a 1px strip -> the re-detected
# kp on the warped frame differ slightly -> the loss won't be bit-exact. Gate generously and
# ALSO report a kp-warp-only check that IS exact (same transform, no image).
print(f"equivariance_loss mlx={m_loss:.6f} torch={t_loss:.6f} rel={rel:.3e}")

# exact sub-check: warp_coordinates of the SAME kp must match torch (no grid_sample involved)
kp_test = (np.random.rand(BS, 50, 2) * 2 - 1).astype(np.float32)
with torch.no_grad():
    wt = tr.warp_coordinates(torch.tensor(kp_test)).cpu().numpy()
trm = ML.TPS(
    mode="random",
    bs=BS,
    _noise=mx.array(noise_t),
    _control_params=mx.array(cparams_t),
    **TP,
)
wm = np.array(trm.warp_coordinates(mx.array(kp_test)))
wd = float(np.max(np.abs(wt - wm)))
print(f"warp_coordinates_exact maxdiff={wd:.3e}")
if wd > 1e-4:
    print("FAIL warp_coordinates")
    fails += 1
else:
    print("PASS warp_coordinates")

# loss: accept if within 25% (border-vs-reflection only perturbs the boundary)
if rel <= 0.25:
    print("PASS equivariance_loss (within border-vs-reflection tolerance)")
else:
    print("FAIL equivariance_loss")
    fails += 1

print(f"\nFAILS={fails}")
print("RESULT_ALL_PASS" if fails == 0 else "RESULT_ANY_FAIL")

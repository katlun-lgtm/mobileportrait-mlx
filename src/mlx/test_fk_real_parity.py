"""Parity: MixedKPDetector with the REAL insightface FK backend — torch vs MLX.

The stub-FK parity test proves the MLP/fusion math; this one proves the real
insightface 2d106 path produces matching fk_kp (and therefore fg_kp) across the
torch and MLX FKDetector implementations, run on a REAL face image (not noise).

Both FKDetectors call the same insightface buffalo_l app; they differ only in how
they marshal the image to uint8 HWC RGB (torch: tensor.clamp.byte.permute; mlx:
np.clip on the NHWC array). Tolerance is loose-ish because the two are independent
FaceAnalysis instances and any preproc rounding shows up in landmark pixels.

Run on the Mac:
    cd ~/mobileportrait-mlx && ~/lp-mlx/.venv/bin/python src/mlx/test_fk_real_parity.py
"""

import glob
import importlib
import os
import sys

import numpy as np
import torch
from PIL import Image

import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)

ref = importlib.import_module("modules.keypoint_detector")
TorchMKD = ref.MixedKPDetector

from mixed_kp_mlx import MixedKPDetector as MlxMKD  # noqa: E402
from mixed_kp_mlx import load_mixedkp_from_torch  # noqa: E402
from losses_mlx import kp_distance_loss  # noqa: E402

NUM_TPS = 10
TOL = 2e-2

torch.manual_seed(0)
np.random.seed(0)

# two real frames -> a (2,3,H,W) batch in [0,1]
DATA = os.path.expanduser("~/mobileportrait-mlx/data/celebvhq_frames/test")
pngs = sorted(glob.glob(os.path.join(DATA, "*", "*.png")))[:2]
print(f"using {len(pngs)} real frames: {[os.path.basename(p) for p in pngs]}")
imgs = []
for p in pngs:
    a = np.asarray(Image.open(p).convert("RGB").resize((256, 256)), np.float32) / 255.0
    imgs.append(a)
img_nhwc = np.ascontiguousarray(np.stack(imgs))  # (2,256,256,3)
img_nchw = np.ascontiguousarray(np.transpose(img_nhwc, (0, 3, 1, 2)))

tmodel = TorchMKD(num_tps=NUM_TPS, fk_backend="insightface").eval()
# MixedKP zero-inits its residual head; randomise it so fg_kp exercises a real delta
torch.nn.init.normal_(tmodel.mixed.mlp[-1].weight, std=0.1)
torch.nn.init.normal_(tmodel.mixed.mlp[-1].bias, std=0.1)
mmodel = MlxMKD(num_tps=NUM_TPS, fk_backend="insightface")
load_mixedkp_from_torch(mmodel, tmodel.state_dict())

with torch.no_grad():
    tout = tmodel(torch.from_numpy(img_nchw))
mout = mmodel(mx.array(img_nhwc))
mx.eval(mout["fg_kp"], mout["nk_kp"], mout["fk_kp"])

# how many of the 106 FK points actually got detected (nonzero) in each impl
t_fk = tout["fk_kp"].numpy()
m_fk = np.array(mout["fk_kp"])
print(
    f"fk nonzero rows: torch={int((np.abs(t_fk).sum(-1) > 0).sum())} "
    f"mlx={int((np.abs(m_fk).sum(-1) > 0).sum())} of {t_fk.shape[0] * t_fk.shape[1]}"
)

fails = 0
for key in ["fk_kp", "fg_kp", "nk_kp"]:
    t = tout[key].numpy()
    m = np.array(mout[key])
    md = float(np.abs(t - m).max())
    mn = float(np.abs(t - m).mean())
    ok = md < TOL
    print(
        f"{'PASS' if ok else 'FAIL'} {key:8s} maxdiff={md:.3e} mean={mn:.3e} tol={TOL:.0e}"
    )
    if not ok:
        fails += 1

# kp_distance with real FK (should be a real, image-dependent value, not the stub ring)
tv = float(torch.abs(tout["fg_kp"][:, :50] - tout["fk_kp"][:, :50]).mean())
mv = float(kp_distance_loss(mout))
print(f"kp_distance(real FK)  torch={tv:.6f}  mlx={mv:.6f}  absdiff={abs(tv - mv):.3e}")
if abs(tv - mv) >= TOL:
    fails += 1

print(f"FAILS={fails}")
print("RESULT_ALL_PASS" if fails == 0 else "RESULT_FAIL")

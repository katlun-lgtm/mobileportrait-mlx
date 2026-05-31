"""Parity: src/mlx/vgg19_mlx.py vs PyTorch src/modules/model.py Vgg19 + perceptual loss.

Uses torchvision vgg19(weights=None) random features (no download); identical weights in
both. Compares the 5 feature taps + ImagePyramide downsamples + the multi-scale L1
perceptual loss scalar.

NOTE: the perceptual-loss section uses 256px images (vox-256 training resolution). VGG19 has
4 maxpools up to relu5_1, and the pyramid's smallest scale (0.125) must survive them
(256*0.125=32 -> 16 -> 8 -> 4 -> 2). A 64px base would give 8px at 0.125 -> crashes in torch
VGG (8->4->2->1->0). That's a test-resolution constraint, not a code issue.

Run on the Mac:
  cd ~/mobileportrait-mlx && ~/lp-mlx/.venv/bin/python src/mlx/test_vgg19_mlx_parity.py
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
from torchvision import models
import importlib

ref_model = importlib.import_module("modules.model")
import vgg19_mlx as V

np.random.seed(0)
torch.manual_seed(0)
SCALES = [1, 0.5, 0.25, 0.125]
WEIGHTS = [10, 10, 10, 10, 10]
BS = 2
fails = 0


def fail(m):
    global fails
    fails += 1
    print("FAIL " + m)


# torch reference Vgg19 with RANDOM features (avoid pretrained download)
t_vgg = ref_model.Vgg19.__new__(ref_model.Vgg19)
torch.nn.Module.__init__(t_vgg)
feats = models.vgg19(weights=None).features
t_vgg.slice1 = torch.nn.Sequential()
t_vgg.slice2 = torch.nn.Sequential()
t_vgg.slice3 = torch.nn.Sequential()
t_vgg.slice4 = torch.nn.Sequential()
t_vgg.slice5 = torch.nn.Sequential()
for x in range(2):
    t_vgg.slice1.add_module(str(x), feats[x])
for x in range(2, 7):
    t_vgg.slice2.add_module(str(x), feats[x])
for x in range(7, 12):
    t_vgg.slice3.add_module(str(x), feats[x])
for x in range(12, 21):
    t_vgg.slice4.add_module(str(x), feats[x])
for x in range(21, 30):
    t_vgg.slice5.add_module(str(x), feats[x])
t_vgg.mean = torch.nn.Parameter(
    torch.tensor(np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)).float(),
    requires_grad=False,
)
t_vgg.std = torch.nn.Parameter(
    torch.tensor(np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)).float(),
    requires_grad=False,
)
t_vgg.eval()

m_vgg = V.Vgg19()
V.load_vgg_from_torch(m_vgg, feats.state_dict())

# --- feature taps (64px is fine: 64->32->16->8->4 through 4 pools) ---
img = np.random.rand(BS, 3, 64, 64).astype(np.float32)
with torch.no_grad():
    t_taps = t_vgg(torch.tensor(img))
m_taps = m_vgg(mx.array(img.transpose(0, 2, 3, 1)))
for i in range(5):
    a = np.array(m_taps[i])
    b = t_taps[i].numpy().transpose(0, 2, 3, 1)
    if a.shape != b.shape:
        fail(f"tap{i} shape mlx={a.shape} torch={b.shape}")
        continue
    d = float(np.max(np.abs(a - b)))
    print(f"{'PASS' if d <= 2e-3 else 'FAIL'}  vgg_tap{i}   maxdiff={d:.3e} tol=2e-03")
    if d > 2e-3:
        fails += 1

# --- ImagePyramide (64px) ---
t_pyr = ref_model.ImagePyramide(SCALES, 3).eval()
m_pyr = V.ImagePyramide(SCALES, 3)
with torch.no_grad():
    tp = t_pyr(torch.tensor(img))
mp = m_pyr(mx.array(img.transpose(0, 2, 3, 1)))
for s in SCALES:
    a = np.array(mp["prediction_" + str(s)])
    b = tp["prediction_" + str(s)].numpy().transpose(0, 2, 3, 1)
    if a.shape != b.shape:
        fail(f"pyr_{s} shape mlx={a.shape} torch={b.shape}")
        continue
    d = float(np.max(np.abs(a - b)))
    print(f"{'PASS' if d <= 2e-3 else 'FAIL'}  pyr_{s}      maxdiff={d:.3e} tol=2e-03")
    if d > 2e-3:
        fails += 1

# --- perceptual loss scalar (256px: 0.125 scale -> 32px survives 4 VGG pools) ---
gen = np.random.rand(BS, 3, 256, 256).astype(np.float32)
real = np.random.rand(BS, 3, 256, 256).astype(np.float32)
with torch.no_grad():
    pg = t_pyr(torch.tensor(gen))
    pr = t_pyr(torch.tensor(real))
    tloss = 0.0
    for s in SCALES:
        xv = t_vgg(pg["prediction_" + str(s)])
        yv = t_vgg(pr["prediction_" + str(s)])
        for i, w in enumerate(WEIGHTS):
            tloss += w * torch.abs(xv[i] - yv[i].detach()).mean()
    tloss = float(tloss)
mloss = float(
    V.perceptual_pyramid_loss(
        m_vgg,
        m_pyr,
        mx.array(gen.transpose(0, 2, 3, 1)),
        mx.array(real.transpose(0, 2, 3, 1)),
        SCALES,
        WEIGHTS,
    )
)
rel = abs(mloss - tloss) / (abs(tloss) + 1e-9)
print(
    f"{'PASS' if rel <= 2e-3 else 'FAIL'}  perceptual_loss mlx={mloss:.5f} torch={tloss:.5f} rel={rel:.3e} tol=2e-03"
)
if rel > 2e-3:
    fails += 1

print(f"\nFAILS={fails}")
print("RESULT_ALL_PASS" if fails == 0 else "RESULT_ANY_FAIL")

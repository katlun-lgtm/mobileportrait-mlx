"""Stage A integration test — GeneratorFullModel with all 4 deltas, forward+backward on CPU.

kp_extractor = MixedKPDetector (Δ1). dense_motion exposes Δ2 residual + Δ3 mask heads.
inpainting = Δ4 synthesis. Δ3 losses fire because we pass driving lmk/fg-mask targets and
the MixedKPDetector exposes fk_kp. FK uses the CPU stub; real backend is the training box.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import torch

from modules.keypoint_detector import MixedKPDetector
from modules.dense_motion import DenseMotionNetwork
from modules.inpainting_network import InpaintingNetwork
from modules.model import GeneratorFullModel, landmark_mask_from_points

DM = dict(
    block_expansion=64,
    max_features=1024,
    num_blocks=5,
    num_tps=10,
    num_channels=3,
    scale_factor=0.25,
    bg=False,
    multi_mask=True,
)
GEN = dict(
    num_channels=3,
    block_expansion=64,
    max_features=512,
    num_down_blocks=3,
    multi_mask=True,
)
TRAIN = dict(
    scales=[1, 0.5, 0.25, 0.125],
    transform_params=dict(sigma_affine=0.05, sigma_tps=0.005, points_tps=5),
    loss_weights=dict(
        perceptual=[10, 10, 10, 10, 10],
        equivariance_value=10,
        warp_loss=10,
        bg=0,
        kp_distance=10,
        landmark_mask=10,
        fg_mask=10,
    ),
    dropout_epoch=35,
    dropout_maxp=0.3,
    dropout_inc_epoch=10,
    dropout_startp=0.1,
    bg_start=10,
)

out = []


def log(s):
    out.append(s)


torch.manual_seed(0)
kp = MixedKPDetector(num_tps=10, fk_backend="stub")
dm = DenseMotionNetwork(**DM)
gen = InpaintingNetwork(**GEN)
model = GeneratorFullModel(kp, None, dm, gen, TRAIN).train()

B, S = 2, 128
drv = torch.rand(B, 3, S, S)
batch = dict(
    source=torch.rand(B, 3, S, S),
    driving=drv,
    lmk_mask=landmark_mask_from_points(torch.rand(B, 106, 2) * 2 - 1, S),
    fg_mask=(torch.rand(B, 1, S, S) > 0.5).float(),
)

losses, generated = model(batch, epoch=40)
total = sum(losses.values())
log("loss terms: " + str({k: round(float(v), 4) for k, v in losses.items()}))
log(f"total: {float(total):.3f}")

assert torch.isfinite(total)
for k in ("perceptual", "equivariance_value", "warp_loss", "kp", "landmark", "mask"):
    assert k in losses, f"missing loss term: {k}"
assert generated["prediction"].shape == (B, 3, S, S), generated["prediction"].shape

total.backward()
mixed_grads = [p for p in model.kp_extractor.mixed.parameters() if p.grad is not None]
assert mixed_grads, "MixedKP MLP received no gradient"
fk_grads = any(p.grad is not None for p in model.kp_extractor.fk.parameters())
assert not fk_grads, "FK detector must stay frozen"
log(
    f"backward OK | MixedKP grads {len(mixed_grads)} tensors | FK frozen {not fk_grads}"
)
log("GeneratorFullModel (all 4 deltas) — ALL PASS")

open("/tmp/full_model_result.txt", "w").write("\n".join(out) + "\n")
for line in out:
    print(line)

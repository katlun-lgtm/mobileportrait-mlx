"""Δ2/Δ3/Δ4 shape + grad test — in-place TPS forks (dense_motion, inpainting_network)."""

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "src",
    ),
)

import torch

from modules.dense_motion import DenseMotionNetwork
from modules.inpainting_network import InpaintingNetwork

DM = dict(
    block_expansion=64,
    max_features=1024,
    num_blocks=5,
    num_tps=10,
    num_channels=3,
    scale_factor=0.25,
    bg=True,
    multi_mask=True,
)
GEN = dict(
    num_channels=3,
    block_expansion=64,
    max_features=512,
    num_down_blocks=3,
    multi_mask=True,
)

lines = []


def log(s):
    lines.append(s)
    print(s)


torch.manual_seed(0)
dm = DenseMotionNetwork(**DM)
gen = InpaintingNetwork(**GEN)

src = torch.rand(2, 3, 256, 256)
ks = {"fg_kp": torch.rand(2, 50, 2) * 2 - 1}
kd = {"fg_kp": torch.rand(2, 50, 2) * 2 - 1}

# Δ2/Δ3 — train mode exposes mask heads
dm.train()
d = dm(src, kd, ks)
assert d["deformation"].shape == (2, 64, 64, 2), d["deformation"].shape
assert d["fg_mask_pred"].shape == (2, 1, 64, 64), d["fg_mask_pred"].shape
assert d["lmk_mask_pred"].shape == (2, 1, 64, 64)
log(
    f"Δ2/Δ3 train: deformation {tuple(d['deformation'].shape)} "
    f"fg {tuple(d['fg_mask_pred'].shape)} lmk {tuple(d['lmk_mask_pred'].shape)}"
)

# residual flow has effect
dm.eval()
with torch.no_grad():
    d1 = dm(src, kd, ks)["deformation"]
    torch.nn.init.zeros_(dm.residual_flow.weight)
    torch.nn.init.zeros_(dm.residual_flow.bias)
    d0 = dm(src, kd, ks)["deformation"]
    assert "fg_mask_pred" not in dm(src, kd, ks), "eval must not expose mask heads"
eff = float((d1 - d0).abs().mean())
assert eff > 0
log(f"Δ2 residual effect mean|d|={eff:.5f}; eval hides mask heads OK")

# Δ4 — synthesis with pseudo-BG + multiview
dm.train()
dense = dm(src, kd, ks)
low = gen.low_ch
with torch.no_grad():
    bg = torch.rand(2, 3, 256, 256)
    mask = torch.rand(2, 1, 256, 256)
    one = gen.encode_lowest(src, bg, mask)
    mv = torch.stack([one + 0.01 * t for t in range(gen.num_multiview)], dim=1)
assert one.shape == (2, low, 32, 32), one.shape
out = gen(src, dense, multiview_feats=mv, pseudo_bg=bg, fg_mask=mask)
assert out["prediction"].shape == (2, 3, 256, 256), out["prediction"].shape
out0 = gen(src, dense)  # optionals omitted -> zeros
assert out0["prediction"].shape == (2, 3, 256, 256)
log(f"Δ4 prediction {tuple(out['prediction'].shape)} low_ch={low}")

# warp-loss path (get_encode) must work with +4-channel first conv
enc = gen.get_encode(src, dense["occlusion_map"])
log(f"Δ4 get_encode OK ({len(enc)} maps)")

# backward through everything
loss = out["prediction"].mean() + sum(o.mean() for o in dense["occlusion_map"])
loss.backward()
g_res = dm.residual_flow.weight.grad is not None
g_mv = gen.mv_merge.weight.grad is not None
assert g_res and g_mv, (g_res, g_mv)
log(f"backward OK: residual_flow grad={g_res} mv_merge grad={g_mv}")
log("Δ2/Δ3/Δ4 IN-PLACE — ALL PASS")

with open("/tmp/delta234_result.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

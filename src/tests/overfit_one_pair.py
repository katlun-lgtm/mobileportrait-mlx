"""Overfit ONE (source, driving) pair — the real wiring test.

Smoke tests prove tensors flow; this proves the model LEARNS. We fix a single source/driving
pair (real photo-like content so the losses are meaningful, not noise→noise) and train the full
MobilePortrait stack for N steps on CPU. If the wiring is correct the total loss — and in
particular the perceptual + warp terms — should drop substantially. A flat curve means a
gradient is silently not connected somewhere.

Self-reenactment setup: source and driving are the SAME image with a small synthetic affine
warp applied to the driving, so a correct model can drive source→driving and the reconstruction
loss has a real, reachable minimum. (noise→noise has no learnable structure and would plateau
even with correct wiring — a false negative.)
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np
import torch
import torch.nn.functional as F

from modules.keypoint_detector import MixedKPDetector
from modules.dense_motion import DenseMotionNetwork
from modules.inpainting_network import InpaintingNetwork
from modules.fk_detector import FKDetector
from modules.model import GeneratorFullModel, landmark_mask_from_points
from modules.mp_dataset import stub_fg_mask, stub_pseudo_bg

S = 128
STEPS = int(os.environ.get("OVERFIT_STEPS", "200"))

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
    num_down_blocks=3,
    max_features=512,
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
    dropout_epoch=10_000,
    dropout_maxp=0.3,
    dropout_inc_epoch=10,
    dropout_startp=0.1,
    bg_start=10_000,
)


def _structured_image(seed):
    """A smooth, structured RGB image (gradients + blobs) — photo-like enough that perceptual
    loss has real structure to fit, unlike uniform noise."""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:S, 0:S].astype("float32") / S
    img = np.stack(
        [
            0.5 + 0.4 * np.sin(6 * xx + 1),
            0.5 + 0.4 * np.cos(5 * yy + 2),
            0.5 + 0.3 * np.sin(4 * (xx + yy)),
        ]
    )
    for _ in range(5):  # a few soft blobs
        cy, cx, r = rng.rand(3)
        g = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (0.02 + 0.05 * r)))
        img += 0.3 * g[None] * rng.rand(3, 1, 1)
    return np.clip(img, 0, 1).astype("float32")


def _affine_warp(img_chw, tx=0.06, ty=-0.04, angle=0.12):
    """Apply a small affine warp (the 'motion' the model must learn to reproduce)."""
    t = torch.from_numpy(img_chw)[None]
    c, s = np.cos(angle), np.sin(angle)
    theta = torch.tensor([[[c, -s, tx], [s, c, ty]]], dtype=torch.float32)
    grid = F.affine_grid(theta, t.shape, align_corners=False)
    return F.grid_sample(t, grid, align_corners=False, padding_mode="border")[0].numpy()


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    source = _structured_image(1)
    driving = _affine_warp(source)
    fk = FKDetector(backend="stub")

    def fk_of(chw):
        with torch.no_grad():
            return fk(torch.from_numpy(chw[None]))[0]

    batch = {
        "source": torch.from_numpy(source)[None],
        "driving": torch.from_numpy(driving)[None],
        "fg_mask": torch.from_numpy(stub_fg_mask(driving))[None],
        "lmk_mask": landmark_mask_from_points(fk_of(driving)[None], S),
        "source_fg_mask": torch.from_numpy(stub_fg_mask(source))[None],
        "pseudo_bg": torch.from_numpy(stub_pseudo_bg(source, stub_fg_mask(source)))[
            None
        ],
    }

    kp = MixedKPDetector(
        fk_backend="stub",
        **{
            k: v
            for k, v in DM.items()
            if k in ("num_tps", "num_channels", "bg", "multi_mask")
        },
    )
    dense = DenseMotionNetwork(**DM)
    inp = InpaintingNetwork(**GEN)
    model = GeneratorFullModel(kp, None, dense, inp, TRAIN).train()

    params = list(kp.parameters()) + list(dense.parameters()) + list(inp.parameters())
    opt = torch.optim.Adam(params, lr=2e-4, betas=(0.5, 0.999))

    history = []
    first = {}
    for step in range(STEPS):
        losses, _ = model(batch, epoch=0)
        total = sum(v.mean() for v in losses.values())
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 10, norm_type=float("inf"))
        opt.step()
        history.append(float(total))
        if step == 0:
            first = {k: float(v.mean()) for k, v in losses.items()}
        if step % 20 == 0 or step == STEPS - 1:
            print(
                f"step {step:3d}  total {float(total):8.3f}  "
                + " ".join(f"{k}={float(v.mean()):.2f}" for k, v in losses.items())
            )

    last = {k: float(v.mean()) for k, v in losses.items()}
    t0, tN = history[0], min(history[-5:]) if len(history) >= 5 else history[-1]
    drop = (t0 - tN) / t0 * 100
    print(f"\ntotal loss: {t0:.3f} -> {tN:.3f}  ({drop:.1f}% drop over {STEPS} steps)")
    print("per-term first -> last:")
    for k in first:
        print(f"  {k:18s} {first[k]:8.3f} -> {last[k]:8.3f}")

    # Acceptance: overall loss drops clearly, and the two reconstruction-driving terms
    # (perceptual = image fidelity, warp = motion fidelity) both fall — that's what proves the
    # source->driving path actually learns, not just the cheap mask/kp terms.
    ok_total = drop > 25
    ok_percep = last["perceptual"] < first["perceptual"] * 0.85
    ok_warp = last["warp_loss"] < first["warp_loss"] * 0.85
    verdict = ok_total and ok_percep and ok_warp
    print(
        f"\nchecks: total>25%drop={ok_total} percep↓15%={ok_percep} warp↓15%={ok_warp}"
    )
    print("OVERFIT ONE PAIR —", "PASS" if verdict else "FAIL (wiring suspect)")
    with open("/tmp/overfit_result.txt", "w") as f:
        f.write(
            f"{'PASS' if verdict else 'FAIL'} drop={drop:.1f}% "
            f"percep {first['perceptual']:.1f}->{last['perceptual']:.1f} "
            f"warp {first['warp_loss']:.1f}->{last['warp_loss']:.1f}\n"
        )
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()

"""Benchmark ONE MobilePortrait training step (fwd+bwd+opt) — MPS vs CPU.

Purpose: convert the "Mac ~100x slower" guess into a measured ms/step so we can decide
train-on-Mac vs rent-a-GPU. Builds the full GeneratorFullModel (all 4 deltas + 6 losses),
times warmup-excluded steps at batch 4 / 256px (the real training shape), and reports
per-step wall time + a projected ETA for a warm-started reduced run (~20k steps).

Run on the Mac:
    ~/lp-mlx/.venv/bin/python ~/mobileportrait-mlx/src/tests/bench_train_step.py --device mps
    ~/lp-mlx/.venv/bin/python ~/mobileportrait-mlx/src/tests/bench_train_step.py --device cpu
"""

import argparse
import os
import sys
import time

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


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument(
        "--target-steps",
        type=int,
        default=20000,
        help="steps for the ETA projection (warm-started reduced run)",
    )
    args = ap.parse_args()
    dev = args.device
    B, S = args.batch, args.size

    torch.manual_seed(0)
    kp = MixedKPDetector(
        fk_backend="stub",
        **{k: DM[k] for k in ("num_tps", "num_channels", "bg", "multi_mask")},
    )
    dense = DenseMotionNetwork(**DM)
    inp = InpaintingNetwork(**GEN)
    model = GeneratorFullModel(kp, None, dense, inp, TRAIN).train().to(dev)

    params = list(kp.parameters()) + list(dense.parameters()) + list(inp.parameters())
    opt = torch.optim.Adam(params, lr=2e-4)

    def make_batch():
        src = torch.rand(B, 3, S, S)
        drv = torch.rand(B, 3, S, S)
        fk = torch.rand(B, 106, 2) * 2 - 1
        return {
            "source": src.to(dev),
            "driving": drv.to(dev),
            "fg_mask": (torch.rand(B, 1, S, S) > 0.5).float().to(dev),
            "lmk_mask": landmark_mask_from_points(fk, S).to(dev),
            "source_fg_mask": (torch.rand(B, 1, S, S) > 0.5).float().to(dev),
            "pseudo_bg": torch.rand(B, 3, S, S).to(dev),
        }

    def step():
        x = make_batch()
        losses, _ = model(x, epoch=0)
        loss = sum(v.mean() for v in losses.values())
        opt.zero_grad()
        loss.backward()
        opt.step()
        return float(loss)

    print(
        f"device={dev} batch={B} size={S} | warmup {args.warmup}, timing {args.iters}"
    )
    for _ in range(args.warmup):
        step()
    sync(dev)

    t0 = time.time()
    last = None
    for _ in range(args.iters):
        last = step()
        sync(dev)
    dt = (time.time() - t0) / args.iters

    eta_h = dt * args.target_steps / 3600
    print(f"per-step: {dt * 1000:.0f} ms  (last loss {last:.1f})")
    print(f"ETA for {args.target_steps} steps: {eta_h:.1f} h ({eta_h / 24:.1f} days)")
    with open(f"/tmp/bench_{dev}.txt", "w") as f:
        f.write(f"{dev} {dt * 1000:.0f}ms/step ETA {eta_h:.1f}h\n")


if __name__ == "__main__":
    main()

"""Convert a MobilePortrait PyTorch checkpoint (.pth.tar) to MLX safetensors.

Usage (on Mac in ~/lp-mlx/.venv):
    python scripts/convert_pth_to_safetensors.py \
        --input  checkpoints/mp3k-best.pth.tar \
        --output checkpoints/mp3k-best.safetensors

The input must have keys: kp_detector, dense_motion_network, inpainting_network.
Produces a safetensors file loadable by model.load_weights() in mp_reenact.py.
"""

import argparse
import os
import sys

HOME = os.path.expanduser("~/mobileportrait-mlx")
sys.path.insert(0, os.path.join(HOME, "src/mlx"))

import torch
import mlx.core as mx
from mixed_kp_mlx import MixedKPDetector, load_mixedkp_from_torch
from dense_motion_mlx import DenseMotionNetwork, load_dm_from_torch
from inpainting_network_mlx import InpaintingNetwork, load_inpaint_from_torch
from pipeline_mlx import MobilePortraitPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to .pth.tar checkpoint")
    ap.add_argument("--output", required=True, help="path to write .safetensors")
    args = ap.parse_args()

    print(f"Loading PyTorch checkpoint: {args.input}", flush=True)
    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    print(f"  keys: {list(ckpt.keys())}  step={ckpt.get('step', '?')}", flush=True)

    # Build MLX model (inference config: bg=False, same arch otherwise)
    kp = MixedKPDetector(num_tps=10, fk_backend="stub")
    dm = DenseMotionNetwork(
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
    inp = InpaintingNetwork(
        num_channels=3,
        block_expansion=64,
        max_features=512,
        num_down_blocks=3,
        multi_mask=True,
    )
    model = MobilePortraitPipeline(kp, dm, inp)
    print("MLX model built", flush=True)

    # Transfer weights using existing helpers
    kp_drift = load_mixedkp_from_torch(kp, ckpt["kp_detector"])
    dm_drift = load_dm_from_torch(dm, ckpt["dense_motion_network"])
    inp_drift = load_inpaint_from_torch(inp, ckpt["inpainting_network"])
    print(f"  kp  drift={kp_drift}", flush=True)
    print(f"  dm  drift={dm_drift}", flush=True)
    print(f"  inp drift={inp_drift}", flush=True)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    model.save_weights(args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Saved: {args.output}  ({size_mb:.1f} MB)", flush=True)
    print("CONVERT_OK", flush=True)


if __name__ == "__main__":
    main()

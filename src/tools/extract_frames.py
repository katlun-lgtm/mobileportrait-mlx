"""Precompute: decode each CelebV-HQ clip (.mp4) to a folder of resized frame images.

TPS FramesDataset has a fast path: when root_dir/<split>/<clip>/ is a DIRECTORY of frame images,
__getitem__ reads just 2 frames instead of decoding the whole 90-frame 1032^2 mp4 every call
(measured ~8.7 s/clip decode). This converts that ~8.7 s into a ~ms image read.

Usage (on the Mac, in ~/mobileportrait-mlx):
  ~/lp-mlx/.venv/bin/python src/tools/extract_frames.py \
      --src data/celebvhq --dst data/celebvhq_frames --size 256

Output layout: data/celebvhq_frames/{train,test}/<clipname>/0000.png ...
Point the training config's dataset_params.root_dir at the --dst directory.
"""

import argparse
import os
import sys

import numpy as np
from skimage.transform import resize
from skimage.io import imsave

# reference-tps on path for its memtest-free read via imageio
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS))
sys.path.insert(0, os.path.join(_ROOT, "reference-tps"))
import imageio.v2 as iio  # noqa: E402


def extract_clip(mp4_path, out_dir, size):
    if os.path.isdir(out_dir) and any(f.endswith(".png") for f in os.listdir(out_dir)):
        return 0  # already done
    os.makedirs(out_dir, exist_ok=True)
    frames = iio.mimread(mp4_path, memtest=False)
    n = 0
    for i, fr in enumerate(frames):
        if fr.ndim == 2:
            fr = np.stack([fr] * 3, axis=-1)
        if fr.shape[-1] == 4:
            fr = fr[..., :3]
        small = (resize(fr, (size, size), anti_aliasing=True) * 255).astype("uint8")
        imsave(os.path.join(out_dir, f"{i:04d}.png"), small, check_contrast=False)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/celebvhq")
    ap.add_argument("--dst", default="data/celebvhq_frames")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    total_clips = 0
    total_frames = 0
    for split in ("train", "test"):
        sdir = os.path.join(args.src, split)
        if not os.path.isdir(sdir):
            continue
        clips = sorted(f for f in os.listdir(sdir) if f.endswith(".mp4"))
        for j, clip in enumerate(clips):
            name = clip[:-4]
            out_dir = os.path.join(args.dst, split, name)
            try:
                n = extract_clip(os.path.join(sdir, clip), out_dir, args.size)
            except Exception as e:
                print(f"FAIL {split}/{clip}: {e}", flush=True)
                continue
            total_clips += 1
            total_frames += n
            if (j + 1) % 20 == 0:
                print(f"{split}: {j + 1}/{len(clips)} clips done", flush=True)
    print(f"DONE clips={total_clips} frames={total_frames}", flush=True)


if __name__ == "__main__":
    main()

"""Extract frames from CelebV-HQ videos.tar streamed from storagebox.

Reads the tar via SSH, decodes each matching .mp4 with ffmpeg, and saves
256x256-padded square PNGs to data/celebvhq_frames/{split}/{clip_name}/.

Usage (on vast.ai after uploading SSH key + clip lists):
    # train clips
    python scripts/extract_frames.py \
        --storagebox u590283@u590283.your-storagebox.de \
        --storagebox-port 23 \
        --tar-path datasets/celebvhq/videos.tar \
        --clip-list /workspace/mobileportrait-mlx/data/clip_list_train.txt \
        --out-dir /workspace/mobileportrait-mlx/data/celebvhq_frames/train \
        --workers 8

    # test clips (same command, different list + out-dir)
    python scripts/extract_frames.py ... \
        --clip-list .../clip_list_test.txt \
        --out-dir .../celebvhq_frames/test

Speed: ~20-40 clips/min on a 3090 box with fast storagebox connection.
Disk:  ~31 GB for 3000 clips at ~160 frames each (256px PNG).
"""

import argparse
import io
import os
import subprocess
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def extract_clip_frames(mp4_bytes: bytes, out_dir: Path, clip_name: str) -> int:
    """Decode mp4_bytes with ffmpeg, write 256x256 PNGs, return frame count."""
    clip_dir = out_dir / clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    # ffmpeg: scale shortest side to 256, center-pad to 256x256, output %04d.png
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-vf",
        "scale=256:256:force_original_aspect_ratio=increase,crop=256:256",
        "-q:v",
        "2",  # PNG quality (lower = better; irrelevant for lossless)
        str(clip_dir / "%04d.png"),
    ]
    proc = subprocess.run(cmd, input=mp4_bytes, capture_output=True)
    if proc.returncode != 0:
        print(
            f"  WARN ffmpeg failed for {clip_name}: {proc.stderr[:200]}",
            file=sys.stderr,
        )
        return 0

    frames = list(clip_dir.glob("*.png"))
    return len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--local-tar",
        default=None,
        help="Path to a local videos.tar file (skips SSH streaming; use this if the "
        "storagebox is not directly accessible from this host)",
    )
    ap.add_argument("--storagebox", default=None, help="user@host for storagebox SSH")
    ap.add_argument("--storagebox-port", type=int, default=23)
    ap.add_argument(
        "--storagebox-key",
        default="/root/.ssh/id_rsa",
        help="SSH private key for storagebox",
    )
    ap.add_argument(
        "--tar-path",
        default=None,
        help="Path of videos.tar on the storagebox (relative to storagebox root)",
    )
    ap.add_argument(
        "--clip-list", required=True, help="File with one clip name per line"
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output directory (e.g. data/celebvhq_frames/train)",
    )
    ap.add_argument("--workers", type=int, default=4, help="Parallel ffmpeg workers")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.clip_list) as f:
        want = {line.strip() for line in f if line.strip()}
    print(f"Clips to extract: {len(want)}", flush=True)

    # skip already-done clips
    done = {p.name for p in out_dir.iterdir() if p.is_dir() and list(p.glob("*.png"))}
    want -= done
    print(f"Already done: {len(done)}, remaining: {len(want)}", flush=True)
    if not want:
        print("All clips already extracted.")
        return

    if args.local_tar:
        print(f"Opening local tar: {args.local_tar}", flush=True)
        tar_ctx = tarfile.open(args.local_tar, mode="r|")
        ssh_proc = None
    elif args.storagebox and args.tar_path:
        ssh_cmd = [
            "ssh",
            "-p",
            str(args.storagebox_port),
            "-i",
            args.storagebox_key,
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "Compression=no",
            args.storagebox,
            f"cat {args.tar_path}",
        ]
        print(f"Streaming tar from {args.storagebox}:{args.tar_path} ...", flush=True)
        ssh_proc = subprocess.Popen(
            ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        tar_ctx = tarfile.open(fileobj=ssh_proc.stdout, mode="r|")
    else:
        print(
            "ERROR: provide --local-tar OR --storagebox + --tar-path", file=sys.stderr
        )
        sys.exit(1)

    extracted = done_count = skip_count = 0
    futures = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        with tar_ctx as tf:
            for member in tf:
                if not member.isfile():
                    continue
                # member.name is like "videos/ClipName.mp4"
                clip_name = Path(member.name).stem
                if clip_name not in want:
                    skip_count += 1
                    tf.members = []  # don't accumulate
                    continue

                f = tf.extractfile(member)
                if f is None:
                    continue
                mp4_bytes = f.read()
                fut = pool.submit(extract_clip_frames, mp4_bytes, out_dir, clip_name)
                futures[fut] = clip_name
                extracted += 1

                # drain completed futures to keep memory bounded
                done_futs = [fu for fu in futures if fu.done()]
                for fu in done_futs:
                    n = fu.result()
                    done_count += 1
                    cname = futures.pop(fu)
                    print(
                        f"  [{done_count}/{len(want)}] {cname}: {n} frames", flush=True
                    )

        # drain remaining
        for fu in as_completed(futures):
            n = fu.result()
            done_count += 1
            cname = futures[fu]
            print(f"  [{done_count}/{len(want)}] {cname}: {n} frames", flush=True)

    if ssh_proc is not None:
        ssh_proc.wait()
    total_frames = sum(len(list((out_dir / c).glob("*.png"))) for c in want)
    print(
        f"\nDONE extracted={extracted} clips, total new frames={total_frames}",
        flush=True,
    )


if __name__ == "__main__":
    main()

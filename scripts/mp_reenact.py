"""MobilePortrait MLX one-shot reenactment demo (M3 Max).

Animate a single SOURCE face photo with the head motion of a DRIVING clip, using
the trained checkpoint (checkpoints/real_best.safetensors). Writes a side-by-side
[source | driving | prediction] still + mp4.

The source photo is face-detected (insightface) and cropped to the loose
VoxCeleb/CelebV-HQ convention the model was trained on (face + hair + a little
shoulder/background), so an arbitrary webcam selfie is brought in-distribution.
Driving frames are assumed already 256px cropped (the held-out test clips are).

Usage:
  python scripts/mp_reenact.py \
      --source test_inputs/source.jpg \
      --driving data/celebvhq_frames/test/--uyzf7X_0c_0 \
      --out renders/reenact --max-frames 60
"""

import os
import sys
import glob
import argparse
import traceback

import numpy as np

HOME = os.path.expanduser("~/mobileportrait-mlx")
sys.path.insert(0, os.path.join(HOME, "src/mlx"))
CKPT = os.path.join(HOME, "checkpoints/real_best.safetensors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="source face image (any size)")
    ap.add_argument(
        "--driving",
        required=True,
        help="dir of 256px frames, a single image, or a video file (.mov/.mp4/.avi)",
    )
    ap.add_argument(
        "--crop-driving",
        action="store_true",
        help="face-crop driving frames (auto-on for video / raw webcam input)",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(HOME, "renders/reenact"),
        help="output path prefix",
    )
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument(
        "--margin", type=float, default=0.8, help="crop margin around the face bbox"
    )
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    import mlx.core as mx
    import cv2
    from PIL import Image
    from mixed_kp_mlx import MixedKPDetector, FKDetector
    from dense_motion_mlx import DenseMotionNetwork
    from inpainting_network_mlx import InpaintingNetwork
    from pipeline_mlx import MobilePortraitPipeline

    print(f"mlx {mx.__version__}  ckpt_exists={os.path.exists(CKPT)}", flush=True)

    # ---- build + load trained weights ----
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
    model.load_weights(CKPT)  # strict; matches mp_infer_bench
    model.eval()
    FK = FKDetector(backend="insightface")
    print("LOAD_OK", flush=True)

    # ---- loose VoxCeleb-style face crop (compute box, then apply) ----
    def compute_box(bgr, margin):
        """Return (X1,Y1,X2,Y2,pad) for the largest face, or None if none found."""
        faces = FK._app.get(bgr)
        if not faces:
            return None
        f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = f.bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * margin)
        cy -= 0.12 * side  # nudge up to include hair, like the training crop
        half = side / 2.0
        X1, Y1, X2, Y2 = int(cx - half), int(cy - half), int(cx + half), int(cy + half)
        h, w = bgr.shape[:2]
        pad = max(0, -X1, -Y1, X2 - w, Y2 - h)
        return (X1, Y1, X2, Y2, pad)

    def apply_box(bgr, box):
        X1, Y1, X2, Y2, pad = box
        if pad > 0:
            bgr = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            X1, Y1, X2, Y2 = X1 + pad, Y1 + pad, X2 + pad, Y2 + pad
        crop = cv2.resize(bgr[Y1:Y2, X1:X2], (256, 256))
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def center_square(bgr):
        h, w = bgr.shape[:2]
        s = min(h, w)
        crop = bgr[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
        return (
            cv2.cvtColor(cv2.resize(crop, (256, 256)), cv2.COLOR_BGR2RGB).astype(
                np.float32
            )
            / 255.0
        )

    def crop_face_256(path, margin):
        bgr = cv2.imread(path)
        if bgr is None:
            raise SystemExit(f"could not read source image: {path}")
        box = compute_box(bgr, margin)
        if box is None:
            print("WARN: no face detected in source -> center-square crop", flush=True)
            return center_square(bgr)
        return apply_box(bgr, box)

    def read_driving_frames(spec, margin, crop, max_frames):
        """-> list of 256px HWC[0,1] RGB frames. Videos/raw images get a FIXED face
        box (computed once from the first detectable frame) for stable framing."""
        VID = (".mov", ".mp4", ".avi", ".m4v", ".mkv")
        if os.path.isfile(spec) and spec.lower().endswith(VID):
            cap = cv2.VideoCapture(spec)
            raw = []
            while len(raw) < max_frames:
                ok, fr = cap.read()
                if not ok:
                    break
                raw.append(fr)
            cap.release()
            if not raw:
                raise SystemExit(f"no frames decoded from {spec}")
            crop = True  # webcam video is always raw
        elif os.path.isdir(spec):
            paths = sorted(
                glob.glob(os.path.join(spec, "*.png"))
                + glob.glob(os.path.join(spec, "*.jpg"))
            )[:max_frames]
            if not paths:
                raise SystemExit(f"no frames in {spec}")
            if not crop:  # pre-cropped 256 test clips
                return [
                    np.asarray(
                        Image.open(p).convert("RGB").resize((256, 256)), np.float32
                    )
                    / 255.0
                    for p in paths
                ]
            raw = [cv2.imread(p) for p in paths]
        else:  # single image
            raw = [cv2.imread(spec)]
            crop = crop or True

        # fixed-box crop: detect once on the first frame with a face, reuse for all
        box = None
        for fr in raw:
            box = compute_box(fr, margin)
            if box is not None:
                break
        if box is None:
            print("WARN: no face in driving -> center-square", flush=True)
            return [center_square(fr) for fr in raw]
        return [apply_box(fr, box) for fr in raw]

    def kp_of(img_np):
        m = mx.array(img_np[None])
        fk = mx.stop_gradient(FK(m))
        nk = model.kp_extractor.nk(m)["fg_kp"]
        return m, {"fg_kp": model.kp_extractor.mixed(fk, nk)}

    src_np = crop_face_256(args.source, args.margin)
    s_mx, kp_s = kp_of(src_np)

    drv_frames = read_driving_frames(
        args.driving, args.margin, args.crop_driving, args.max_frames
    )
    print(f"driving frames: {len(drv_frames)}", flush=True)

    strips = []
    for i, d_np in enumerate(drv_frames):
        _, kp_d = kp_of(d_np)
        pred = model.forward_with_kp(s_mx, kp_s, kp_d)["prediction"]
        mx.eval(pred)
        p = np.clip(np.array(pred)[0], 0, 1)
        strip = (np.concatenate([src_np, d_np, p], axis=1) * 255).astype(np.uint8)
        strips.append(strip)
        if i % 10 == 0:
            print(f"  frame {i + 1}/{len(drv_frames)}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    still = args.out + "_still.png"
    Image.fromarray(strips[len(strips) // 2]).save(still)
    print(f"STILL -> {still}", flush=True)

    if len(strips) > 1:
        h, w, _ = strips[0].shape
        mp4 = args.out + ".mp4"
        vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        for s in strips:
            vw.write(cv2.cvtColor(s, cv2.COLOR_RGB2BGR))
        vw.release()
        print(f"VIDEO -> {mp4}  ({len(strips)} frames @ {args.fps}fps)", flush=True)
    print("REENACT_END", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("REENACT_FAIL\n" + traceback.format_exc(), flush=True)
        sys.exit(1)

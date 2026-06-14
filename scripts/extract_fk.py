"""Pre-extract insightface 106-pt FK keypoints for all CelebV-HQ frames."""
import os, time, glob
import torch  # must import before insightface so CUDA context initialises for ONNX
_ = torch.zeros(1, device="cuda")  # force CUDA init
import numpy as np
import cv2
from insightface.app import FaceAnalysis

DATA = "/workspace/mobileportrait-mlx/data/celebvhq_frames"
OUT  = "/workspace/mobileportrait-mlx/data/celebvhq_fk"

app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "landmark_2d_106"])
app.prepare(ctx_id=0, det_size=(256, 256))  # GPU 0

def extract_split(split):
    in_dir  = os.path.join(DATA, split)
    out_dir = os.path.join(OUT,  split)
    clips   = sorted(os.listdir(in_dir))
    total   = sum(len(glob.glob(os.path.join(in_dir, c, "*.png"))) for c in clips)
    done    = 0
    t0      = time.perf_counter()
    for clip in clips:
        clip_out = os.path.join(out_dir, clip)
        os.makedirs(clip_out, exist_ok=True)
        frames = sorted(glob.glob(os.path.join(in_dir, clip, "*.png")))
        for fpath in frames:
            stem     = os.path.splitext(os.path.basename(fpath))[0]
            npy_path = os.path.join(clip_out, stem + ".npy")
            if os.path.exists(npy_path):
                done += 1
                continue
            im_bgr = cv2.imread(fpath)
            H, W   = im_bgr.shape[:2]
            faces  = app.get(im_bgr)
            kp     = np.zeros((106, 2), dtype="float32")
            if faces:
                f   = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                lmk = np.asarray(f.landmark_2d_106, dtype="float32")
                kp  = lmk / np.array([W, H], dtype="float32") * 2 - 1
            np.save(npy_path, kp)
            done += 1
            if done % 500 == 0:
                elapsed = time.perf_counter() - t0
                fps     = done / elapsed
                eta     = (total - done) / fps
                print(f"[{split}] {done}/{total}  {fps:.1f} fr/s  ETA {eta/60:.1f}m", flush=True)
    print(f"[{split}] DONE {done} frames", flush=True)

print("Extracting train...", flush=True)
extract_split("train")
print("Extracting test...", flush=True)
extract_split("test")
print("ALL DONE", flush=True)

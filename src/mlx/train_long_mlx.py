"""Longer warm-started run (1500 steps) + self-reenactment render.

Same recipe proven stable across seeds: TPS warm-start, seed, lr 2e-5, clip 1.0,
skip-nonfinite. Logs loss every 50 steps to /tmp/train_long.log. At the end renders a
self-reenactment from a HELD-OUT test clip (source=frame A, driving=frame B, same
identity) and saves a side-by-side source|driving|prediction PNG to
~/mobileportrait-mlx/renders/reenact.png, plus the prediction-vs-driving L1.

Self-reenactment is the right visual check: same identity, so a working model should
reconstruct the driving frame's pose -> prediction should resemble the driving frame.
"""

import os
import sys
import time
import glob
import traceback

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
L = open("/tmp/train_long.log", "w")


def log(m):
    L.write(m + "\n")
    L.flush()


log("TRAIN_LONG_BEGIN")
log("mlx " + mx.__version__)

try:
    import torch
    from PIL import Image
    from keypoint_detector_mlx import KPDetector
    from dense_motion_mlx import DenseMotionNetwork
    from inpainting_network_mlx import InpaintingNetwork
    from pipeline_mlx import MobilePortraitPipeline
    from vgg19_mlx import Vgg19, ImagePyramide, perceptual_pyramid_loss

    log("IMPORT_OK")
except Exception:
    log("IMPORT_FAIL\n" + traceback.format_exc())
    log("TRAIN_LONG_END")
    L.close()
    print("done")
    sys.exit(0)

DATA = os.path.expanduser("~/mobileportrait-mlx/data/celebvhq_frames")
CKPT = os.path.expanduser("~/mobileportrait-mlx/checkpoints/vox.pth.tar")
RENDER_DIR = os.path.expanduser("~/mobileportrait-mlx/renders")
os.makedirs(RENDER_DIR, exist_ok=True)
SCALES = [1, 0.5, 0.25, 0.125]
WEIGHTS = [10] * 5
BS = 2
STEPS = 1500
LOG_EVERY = 50
LR = 2e-5
MAX_NORM = 1.0
rng = np.random.default_rng(0)
mx.random.seed(0)


def to_mlx(v):
    a = mx.array(v.detach().cpu().numpy())
    return mx.transpose(a, (0, 2, 3, 1)) if a.ndim == 4 else a


def warm(module, sd, first_expand=False):
    own = dict(tree_flatten(module.parameters()))
    m = {}
    for k, v in sd.items():
        if k.endswith("num_batches_tracked"):
            continue
        a = to_mlx(v)
        if k in own and own[k].shape == a.shape:
            m[k] = a
        elif first_expand and k == "first.conv.weight" and k in own:
            w = mx.zeros(own[k].shape, dtype=own[k].dtype)
            w[:, :, :, :3] = a
            m[k] = w
    module.unfreeze(recurse=True)
    module.update(tree_unflatten(list(m.items())))
    module.eval()


def load_png(path):
    return (
        np.asarray(Image.open(path).convert("RGB").resize((256, 256)), np.float32)
        / 255.0
    )


train_clips = []
for c in sorted(os.listdir(os.path.join(DATA, "train"))):
    fs = sorted(glob.glob(os.path.join(DATA, "train", c, "*.png")))
    if len(fs) >= 2:
        train_clips.append(fs)
test_clips = []
for c in sorted(os.listdir(os.path.join(DATA, "test"))):
    fs = sorted(glob.glob(os.path.join(DATA, "test", c, "*.png")))
    if len(fs) >= 2:
        test_clips.append(fs)
log(f"DATA_OK train={len(train_clips)} test={len(test_clips)}")


def sample_batch():
    src = np.zeros((BS, 256, 256, 3), np.float32)
    drv = np.zeros((BS, 256, 256, 3), np.float32)
    for b in range(BS):
        clip = train_clips[rng.integers(len(train_clips))]
        src[b] = load_png(clip[rng.integers(len(clip))])
        drv[b] = load_png(clip[rng.integers(len(clip))])
    return mx.array(src), mx.array(drv)


try:
    kp = KPDetector(num_tps=10)
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
    vgg = Vgg19()
    pyr = ImagePyramide(SCALES, 3)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    warm(kp, ck["kp_detector"])
    warm(dm, ck["dense_motion_network"])
    warm(inp, ck["inpainting_network"], first_expand=True)
    model.eval()
    log("WARM_START_OK")
except Exception:
    log("BUILD_FAIL\n" + traceback.format_exc())
    log("TRAIN_LONG_END")
    L.close()
    print("done")
    sys.exit(0)


def loss_fn(model, src, drv):
    pred = model(src, drv)["prediction"]
    l1 = mx.mean(mx.abs(pred - drv))
    perc = perceptual_pyramid_loss(vgg, pyr, pred, drv, SCALES, WEIGHTS)
    return 10.0 * l1 + perc


def render_reenact(tag):
    """Source=test frame 0, driving=test frame mid. Save src|drv|pred PNG; log L1."""
    clip = test_clips[0]
    s_np = load_png(clip[0])[None]
    d_np = load_png(clip[len(clip) // 2])[None]
    pred = model(mx.array(s_np), mx.array(d_np))["prediction"]
    mx.eval(pred)
    p = np.clip(np.array(pred)[0], 0, 1)
    l1 = float(np.mean(np.abs(p - d_np[0])))
    strip = np.concatenate([s_np[0], d_np[0], p], axis=1)  # H x 3W x 3
    out = (strip * 255).astype(np.uint8)
    path = os.path.join(RENDER_DIR, f"reenact_{tag}.png")
    Image.fromarray(out).save(path)
    log(f"RENDER {tag}: saved {path} pred_vs_driving_L1={l1:.5f}")


try:
    render_reenact("init")  # before training, for before/after comparison
    opt = optim.Adam(learning_rate=LR)
    lg = nn.value_and_grad(model, loss_fn)
    times = []
    nonfinite = 0
    skipped = 0
    for step in range(STEPS):
        src, drv = sample_batch()
        ts = time.perf_counter()
        loss, grads = lg(model, src, drv)
        grads, total = optim.clip_grad_norm(grads, MAX_NORM)
        if np.isfinite(float(total.item())):
            opt.update(model, grads)
        else:
            skipped += 1
        mx.eval(model.parameters(), opt.state, loss)
        te = time.perf_counter()
        times.append(te - ts)
        lv = float(loss.item())
        if not np.isfinite(lv):
            nonfinite += 1
        if step % LOG_EVERY == 0 or step == STEPS - 1:
            log(f"STEP {step:04d} loss={lv:.6f} step_s={te - ts:.3f}")
    warm_t = times[2:]
    log(
        f"TIMING steps={len(times)} nonfinite_steps={nonfinite} skipped_updates={skipped} "
        f"mean_step_s_excl_warmup={sum(warm_t) / len(warm_t):.3f}"
    )
    render_reenact("final")
    log("TRAIN_LONG_DONE")
except Exception:
    log("TRAIN_FAIL\n" + traceback.format_exc())

log("TRAIN_LONG_END")
L.close()
print("done")

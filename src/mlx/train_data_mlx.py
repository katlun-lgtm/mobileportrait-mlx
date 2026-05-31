"""MobilePortrait MLX real-data train step: same-clip (source, driving) pairs from
CelebV-HQ frames (256x256 PNG). Proves the GPU train loop works on REAL faces, not the
random-noise overfit toy, and measures loss curve + s/step. Writes /tmp/train_data.log.

Data layout (confirmed): data/celebvhq_frames/{train,test}/<clip>/00000.png ...
We train on the 'train' split (320 clips, ~161 frames each).

Still a smoke (random-init weights, no warm-start) — the point is: real data flows,
loss decreases over a short run, gradients finite, s/step on the M3 Max GPU.
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
from mlx.utils import tree_flatten

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
L = open("/tmp/train_data.log", "w")


def log(m):
    L.write(m + "\n")
    L.flush()


log("TRAIN_DATA_BEGIN")
log("mlx " + mx.__version__)

try:
    from PIL import Image
    from keypoint_detector_mlx import KPDetector
    from dense_motion_mlx import DenseMotionNetwork
    from inpainting_network_mlx import InpaintingNetwork
    from pipeline_mlx import MobilePortraitPipeline
    from vgg19_mlx import Vgg19, ImagePyramide, perceptual_pyramid_loss

    log("IMPORT_OK")
except Exception:
    log("IMPORT_FAIL\n" + traceback.format_exc())
    log("TRAIN_DATA_END")
    L.close()
    print("done")
    sys.exit(0)

DATA = os.path.expanduser("~/mobileportrait-mlx/data/celebvhq_frames")
SPLIT = "train"
SCALES = [1, 0.5, 0.25, 0.125]
WEIGHTS = [10] * 5
BS = 2
STEPS = 40
LOG_EVERY = 4
rng = np.random.default_rng(0)

# clip dirs live at DATA/<split>/<clip>/*.png
clips = []
split_dir = os.path.join(DATA, SPLIT)
for c in sorted(os.listdir(split_dir)):
    fs = sorted(glob.glob(os.path.join(split_dir, c, "*.png")))
    if len(fs) >= 2:
        clips.append(fs)
log(f"DATA_OK split={SPLIT} n_clips={len(clips)}")

if len(clips) == 0:
    log("DATA_FAIL no clips found")
    log("TRAIN_DATA_END")
    L.close()
    print("done")
    sys.exit(0)


def load_png(path):
    im = Image.open(path).convert("RGB").resize((256, 256))
    return np.asarray(im, dtype=np.float32) / 255.0  # HWC [0,1]


def sample_batch():
    src = np.zeros((BS, 256, 256, 3), np.float32)
    drv = np.zeros((BS, 256, 256, 3), np.float32)
    for b in range(BS):
        clip = clips[rng.integers(len(clips))]
        i, j = rng.integers(len(clip)), rng.integers(len(clip))
        src[b] = load_png(clip[i])
        drv[b] = load_png(clip[j])
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
    nparams = sum(
        v.size for _, v in tree_flatten(model.parameters()) if hasattr(v, "size")
    )
    log(f"BUILD_OK params={nparams}")
except Exception:
    log("BUILD_FAIL\n" + traceback.format_exc())
    log("TRAIN_DATA_END")
    L.close()
    print("done")
    sys.exit(0)


def loss_fn(model, src, drv):
    pred = model(src, drv)["prediction"]
    l1 = mx.mean(mx.abs(pred - drv))
    perc = perceptual_pyramid_loss(vgg, pyr, pred, drv, SCALES, WEIGHTS)
    return 10.0 * l1 + perc


try:
    opt = optim.Adam(learning_rate=1e-4)
    lg = nn.value_and_grad(model, loss_fn)
    times = []
    nonfinite_steps = 0
    for step in range(STEPS):
        src, drv = sample_batch()
        ts = time.perf_counter()
        loss, grads = lg(model, src, drv)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        te = time.perf_counter()
        times.append(te - ts)
        lv = float(loss.item())
        if not np.isfinite(lv):
            nonfinite_steps += 1
        if step % LOG_EVERY == 0 or step == STEPS - 1:
            log(f"STEP {step:03d} loss={lv:.6f} step_s={te - ts:.3f}")
    warm = times[2:]
    log(
        f"TIMING steps={len(times)} nonfinite_steps={nonfinite_steps} "
        f"mean_step_s_excl_warmup={sum(warm) / len(warm):.3f} min_step_s={min(times):.3f}"
    )
    log("TRAIN_DATA_DONE")
except Exception:
    log("TRAIN_FAIL\n" + traceback.format_exc())

log("TRAIN_DATA_END")
L.close()
print("done")

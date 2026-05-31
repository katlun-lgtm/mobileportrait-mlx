"""Warm-start the MLX pipeline from the pretrained TPS vox.pth.tar, then train on real
CelebV-HQ pairs and log the loss curve. The point: with TPS weights loaded (shared layers),
a few-hundred-step run on real faces should show loss DECREASING (vs the from-scratch run
which just bounced). Writes /tmp/train_warm.log.

Warm-start is SHAPE-CHECKED and SELF-REPORTING: for each torch sub-state-dict, only assign
into an MLX param when (a) the same key exists in the MLX module AND (b) shapes match after
the NCHW->NHWC conv transpose. The inpainting first conv (TPS 3-ch -> MLX 7-ch) is expanded
(copy first 3 input channels, zero the extra 4). Everything unmatched (the 4 deltas +
mv_merge) stays at init. Counts are logged so nothing is silently dropped.
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
L = open("/tmp/train_warm.log", "w")


def log(m):
    L.write(m + "\n")
    L.flush()


log("TRAIN_WARM_BEGIN")
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
    log("TRAIN_WARM_END")
    L.close()
    print("done")
    sys.exit(0)

DATA = os.path.expanduser("~/mobileportrait-mlx/data/celebvhq_frames")
CKPT = os.path.expanduser("~/mobileportrait-mlx/checkpoints/vox.pth.tar")
SCALES = [1, 0.5, 0.25, 0.125]
WEIGHTS = [10] * 5
BS = 2
STEPS = 300
LOG_EVERY = 20
rng = np.random.default_rng(0)
mx.random.seed(0)  # reproducible delta-layer init (the control: same init each run)


def to_mlx(v):
    a = mx.array(v.detach().cpu().numpy())
    if a.ndim == 4:  # conv (out,in,kH,kW) -> (out,kH,kW,in)
        a = mx.transpose(a, (0, 2, 3, 1))
    return a


def warm_start(module, torch_sd, label, first_conv_expand=False):
    """Assign torch_sd into module where key exists + shape matches. Returns counts."""
    own = dict(tree_flatten(module.parameters()))
    matched = {}
    skipped_shape = 0
    for k, v in torch_sd.items():
        if k.endswith("num_batches_tracked"):
            continue
        a = to_mlx(v)
        if k not in own:
            continue
        if own[k].shape == a.shape:
            matched[k] = a
        elif first_conv_expand and k == "first.conv.weight":
            # MLX (out,kH,kW,7); TPS->(out,kH,kW,3). copy first 3, zero rest.
            w = mx.zeros(own[k].shape, dtype=own[k].dtype)
            w[:, :, :, :3] = a
            matched[k] = w
        else:
            skipped_shape += 1
    module.unfreeze(recurse=True)
    module.update(tree_unflatten(list(matched.items())))
    n_own = len(own)
    n_loaded = len(matched)
    log(
        f"WARM {label}: loaded {n_loaded}/{n_own} (left_random={n_own - n_loaded}, "
        f"skipped_shape={skipped_shape})"
    )
    return n_loaded, n_own


# clips
clips = []
for c in sorted(os.listdir(os.path.join(DATA, "train"))):
    fs = sorted(glob.glob(os.path.join(DATA, "train", c, "*.png")))
    if len(fs) >= 2:
        clips.append(fs)
log(f"DATA_OK n_clips={len(clips)}")


def load_png(path):
    im = Image.open(path).convert("RGB").resize((256, 256))
    return np.asarray(im, dtype=np.float32) / 255.0


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
    log(
        "BUILD_OK params=%d"
        % sum(v.size for _, v in tree_flatten(model.parameters()) if hasattr(v, "size"))
    )
except Exception:
    log("BUILD_FAIL\n" + traceback.format_exc())
    log("TRAIN_WARM_END")
    L.close()
    print("done")
    sys.exit(0)

# ---- warm-start ----
try:
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    warm_start(kp, ck["kp_detector"], "kp")
    warm_start(dm, ck["dense_motion_network"], "dm")
    warm_start(inp, ck["inpainting_network"], "inp", first_conv_expand=True)
    model.eval()
    log("WARM_START_OK")
except Exception:
    log("WARM_START_FAIL\n" + traceback.format_exc())
    log("TRAIN_WARM_END")
    L.close()
    print("done")
    sys.exit(0)


def loss_fn(model, src, drv):
    pred = model(src, drv)["prediction"]
    l1 = mx.mean(mx.abs(pred - drv))
    perc = perceptual_pyramid_loss(vgg, pyr, pred, drv, SCALES, WEIGHTS)
    return 10.0 * l1 + perc


try:
    opt = optim.Adam(learning_rate=2e-5)
    lg = nn.value_and_grad(model, loss_fn)
    MAX_NORM = 1.0
    times = []
    nonfinite = 0
    skipped = 0
    losses = []
    for step in range(STEPS):
        src, drv = sample_batch()
        ts = time.perf_counter()
        loss, grads = lg(model, src, drv)
        # Clip the global grad norm; if the norm is non-finite (e.g. inv() of a
        # momentarily near-singular TPS L blew up), SKIP the update so a single bad
        # step can't poison the weights with nan/inf forever. util_mlx is left untouched
        # (its 0.01*I regulariser is parity-locked to torch).
        grads, total_norm = optim.clip_grad_norm(grads, MAX_NORM)
        gn = float(total_norm.item())
        if np.isfinite(gn):
            opt.update(model, grads)
        else:
            skipped += 1
        mx.eval(model.parameters(), opt.state, loss)
        te = time.perf_counter()
        times.append(te - ts)
        lv = float(loss.item())
        losses.append(lv)
        if not np.isfinite(lv):
            nonfinite += 1
        if step % LOG_EVERY == 0 or step == STEPS - 1:
            log(f"STEP {step:03d} loss={lv:.6f} gnorm={gn:.4f} step_s={te - ts:.3f}")
    # moving-average of first vs last 20 to summarise trend WITHOUT my interpretation
    first20 = sum(losses[:20]) / 20
    last20 = sum(losses[-20:]) / 20
    warm = times[2:]
    log(f"MEAN_FIRST20={first20:.6f} MEAN_LAST20={last20:.6f}")
    log(
        f"TIMING steps={len(times)} nonfinite_steps={nonfinite} skipped_updates={skipped} "
        f"mean_step_s_excl_warmup={sum(warm) / len(warm):.3f} min_step_s={min(times):.3f}"
    )
    log("TRAIN_WARM_DONE")
except Exception:
    log("TRAIN_FAIL\n" + traceback.format_exc())

log("TRAIN_WARM_END")
L.close()
print("done")

"""MobilePortrait MLX training-step smoke: prove GPU fwd+bwd works end-to-end and
measure s/step on the M3 Max. Overfit ONE (source,driving) pair; loss must drop.

Writes everything to /tmp/train_smoke.log with explicit markers. The log is the source
of truth (incl. any backward failure through grid_sample's custom VJP or TPS mx.linalg.inv).

Random-init weights (no torch): the goal is GPU fwd+bwd + s/step + loss-drop, not parity.
Loss = 10*L1(pred,driving) + perceptual_pyramid_loss (random fixed VGG as a fixed feature
extractor — a valid differentiable reconstruction signal for an overfit).
"""

import os
import sys
import time
import traceback

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LOG = open("/tmp/train_smoke.log", "w")


def log(m):
    LOG.write(m + "\n")
    LOG.flush()


log("TRAIN_SMOKE_BEGIN")
log("mlx " + mx.__version__)

try:
    from keypoint_detector_mlx import KPDetector
    from dense_motion_mlx import DenseMotionNetwork
    from inpainting_network_mlx import InpaintingNetwork
    from pipeline_mlx import MobilePortraitPipeline
    from vgg19_mlx import Vgg19, ImagePyramide, perceptual_pyramid_loss

    log("IMPORT_OK")
except Exception:
    log("IMPORT_FAIL\n" + traceback.format_exc())
    log("TRAIN_SMOKE_END")
    LOG.close()
    print("done")
    sys.exit(0)

SCALES = [1, 0.5, 0.25, 0.125]
WEIGHTS = [10, 10, 10, 10, 10]
H = 256
BS = 1

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
    log("TRAIN_SMOKE_END")
    LOG.close()
    print("done")
    sys.exit(0)

# fixed overfit pair (sigmoid-bounded so prediction target is in [0,1])
mx.random.seed(0)
source = mx.sigmoid(mx.random.normal((BS, H, H, 3)))
driving = mx.sigmoid(mx.random.normal((BS, H, H, 3)))
mx.eval(source, driving)


def loss_fn(model):
    gen = model(source, driving)
    pred = gen["prediction"]
    l1 = mx.mean(mx.abs(pred - driving))
    perc = perceptual_pyramid_loss(vgg, pyr, pred, driving, SCALES, WEIGHTS)
    return 10.0 * l1 + perc


# ---- gradient smoke: does backward flow through the WHOLE pipeline? ----
try:
    lg = nn.value_and_grad(model, loss_fn)
    t0 = time.perf_counter()
    loss, grads = lg(model)
    mx.eval(loss, grads)
    t1 = time.perf_counter()
    gflat = [(k, g) for k, g in tree_flatten(grads) if hasattr(g, "size")]
    finite = all(bool(mx.all(mx.isfinite(g)).item()) for _, g in gflat)
    nz = sum(1 for _, g in gflat if float(mx.sum(mx.abs(g)).item()) > 0)
    log(
        f"GRAD_SMOKE_OK loss0={float(loss.item()):.6f} all_finite={finite} "
        f"nonzero_grad_tensors={nz}/{len(gflat)} first_fwdbwd_s={t1 - t0:.3f}"
    )
except Exception:
    log("GRAD_SMOKE_FAIL\n" + traceback.format_exc())
    log("TRAIN_SMOKE_END")
    LOG.close()
    print("done")
    sys.exit(0)

# ---- overfit loop: loss must drop ----
try:
    opt = optim.Adam(learning_rate=1e-4)
    lg = nn.value_and_grad(model, loss_fn)
    times = []
    for step in range(12):
        ts = time.perf_counter()
        loss, grads = lg(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        te = time.perf_counter()
        times.append(te - ts)
        log(f"STEP {step:02d} loss={float(loss.item()):.6f} step_s={te - ts:.3f}")
    warm = times[2:]
    log(
        f"TIMING steps={len(times)} mean_step_s_excl_warmup={sum(warm) / len(warm):.3f} "
        f"min_step_s={min(times):.3f}"
    )
    log("OVERFIT_DONE")
except Exception:
    log("OVERFIT_FAIL\n" + traceback.format_exc())

log("TRAIN_SMOKE_END")
LOG.close()
print("done")

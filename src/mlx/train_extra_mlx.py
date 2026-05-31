"""Warm-started run with the FULL MobilePortrait loss set (the experiment).

Tests whether equivariance + warp + kp_distance(real FK) push quality past the
L1+perceptual plateau (prior finding: render L1 ~0.063, blurry-but-recognizable).

Controlled vs train_long_mlx: SAME base (10*L1 + perceptual pyramid), SAME recipe
(TPS warm-start, seed, lr 2e-5, clip 1.0, skip-nonfinite, BS2), PLUS three terms:
  + equivariance_value * equivariance(NK)     (weight 10)
  + warp_loss        * warp_loss              (weight 10)
  + kp_distance      * |fg_kp - fk_kp|        (weight 10)
weights + transform_params from configs/mac-celebvhq-256.yaml.

The KP path is the MixedKPDetector pieces (NK resnet + MixedKP MLP). The frozen FK is
insightface via onnxruntime (numpy) — NOT differentiable / not traceable — so fk_kp for
both source and driving is precomputed EAGERLY each step (mx.stop_gradient) and fed in
as a constant. The standalone FK lives OUTSIDE model.parameters() so opt.update never
touches it (a no-param submodule's grad subtree is {} and MLX update() would drop it).

Usage:  python train_extra_mlx.py [STEPS]   (STEPS default 1500; pass 2 to smoke)
"""

import glob
import os
import sys
import time
import traceback

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
L = open("/tmp/train_extra.log", "w")


def log(m):
    L.write(m + "\n")
    L.flush()


log("TRAIN_EXTRA_BEGIN")
log("mlx " + mx.__version__)
log(f"STEPS={STEPS}")

try:
    import torch
    from PIL import Image
    from mixed_kp_mlx import MixedKPDetector, FKDetector
    from dense_motion_mlx import DenseMotionNetwork
    from inpainting_network_mlx import InpaintingNetwork
    from pipeline_mlx import MobilePortraitPipeline
    from vgg19_mlx import Vgg19, ImagePyramide, perceptual_pyramid_loss
    from losses_mlx import equivariance_loss, warp_loss, kp_distance_loss

    log("IMPORT_OK")
except Exception:
    log("IMPORT_FAIL\n" + traceback.format_exc())
    log("TRAIN_EXTRA_END")
    L.close()
    print("done")
    sys.exit(0)

DATA = os.path.expanduser("~/mobileportrait-mlx/data/celebvhq_frames")
CKPT = os.path.expanduser("~/mobileportrait-mlx/checkpoints/vox.pth.tar")
RENDER_DIR = os.path.expanduser("~/mobileportrait-mlx/renders")
os.makedirs(RENDER_DIR, exist_ok=True)
SCALES = [1, 0.5, 0.25, 0.125]
WEIGHTS = [10] * 5
TRANSFORM_PARAMS = {"sigma_affine": 0.05, "sigma_tps": 0.005, "points_tps": 5}
W_EQ = 10.0
W_WARP = 10.0
W_KP = 10.0
BS = 2
LOG_EVERY = 50
LR = 2e-5
MAX_NORM = 1.0
rng = np.random.default_rng(0)
mx.random.seed(0)


def to_mlx(v):
    a = mx.array(v.detach().cpu().numpy())
    return mx.transpose(a, (0, 2, 3, 1)) if a.ndim == 4 else a


def warm(module, sd, first_expand=False, key_prefix=""):
    """Load torch state_dict into an MLX module by exact (optionally prefixed) key."""
    own = dict(tree_flatten(module.parameters()))
    m = {}
    for k, v in sd.items():
        if k.endswith("num_batches_tracked"):
            continue
        kk = key_prefix + k
        a = to_mlx(v)
        if kk in own and own[kk].shape == a.shape:
            m[kk] = a
        elif first_expand and k == "first.conv.weight" and kk in own:
            w = mx.zeros(own[kk].shape, dtype=own[kk].dtype)
            w[:, :, :, :3] = a
            m[kk] = w
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
    # model's MixedKPDetector uses the lightweight STUB fk (never called in this script);
    # the REAL insightface FK is the standalone `FK` below, outside model.parameters().
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
    vgg = Vgg19()
    pyr = ImagePyramide(SCALES, 3)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    # vox kp_detector keys are fg_encoder.* ; MixedKPDetector nests NK under nk.*
    warm(kp, ck["kp_detector"], key_prefix="nk.")
    warm(dm, ck["dense_motion_network"])
    warm(inp, ck["inpainting_network"], first_expand=True)
    model.eval()
    FK = FKDetector(backend="insightface")  # standalone — never opt.update'd
    log("WARM_START_OK")
except Exception:
    log("BUILD_FAIL\n" + traceback.format_exc())
    log("TRAIN_EXTRA_END")
    L.close()
    print("done")
    sys.exit(0)


def _terms(model, src, drv, fk_s, fk_d):
    """Shared forward -> the five loss terms (mx scalars). Used by the grad'd loss_fn
    (summed) and the eager logging helper (read as floats)."""
    nk_s = model.kp_extractor.nk(src)["fg_kp"]
    nk_d = model.kp_extractor.nk(drv)["fg_kp"]
    mixed_s = model.kp_extractor.mixed(fk_s, nk_s)
    mixed_d = model.kp_extractor.mixed(fk_d, nk_d)
    gen = model.forward_with_kp(src, {"fg_kp": mixed_s}, {"fg_kp": mixed_d})
    pred = gen["prediction"]

    l1 = mx.mean(mx.abs(pred - drv))
    perc = perceptual_pyramid_loss(vgg, pyr, pred, drv, SCALES, WEIGHTS)
    # equivariance on the learned NK detector (FK is a fixed detector, kept out of graph)
    eq = equivariance_loss(
        model.kp_extractor.nk, drv, {"fg_kp": nk_d}, TRANSFORM_PARAMS
    )
    wl = warp_loss(model.inpainting_network, drv, gen)
    kpd = kp_distance_loss({"fg_kp": mixed_d, "fk_kp": fk_d})
    return l1, perc, eq, wl, kpd


def loss_fn(model, src, drv, fk_s, fk_d):
    # mx.value_and_grad has no has_aux -> return a single scalar.
    l1, perc, eq, wl, kpd = _terms(model, src, drv, fk_s, fk_d)
    return 10.0 * l1 + perc + W_EQ * eq + W_WARP * wl + W_KP * kpd


def log_parts(model, src, drv, fk_s, fk_d):
    parts = _terms(model, src, drv, fk_s, fk_d)
    mx.eval(*parts)
    return tuple(float(p.item()) for p in parts)


def render_reenact(tag):
    """Source=test frame 0, driving=test frame mid. Save src|drv|pred PNG; log L1."""
    clip = test_clips[0]
    s_np = load_png(clip[0])[None]
    d_np = load_png(clip[len(clip) // 2])[None]
    s_mx, d_mx = mx.array(s_np), mx.array(d_np)
    fk_s = mx.stop_gradient(FK(s_mx))
    fk_d = mx.stop_gradient(FK(d_mx))
    nk_s = model.kp_extractor.nk(s_mx)["fg_kp"]
    nk_d = model.kp_extractor.nk(d_mx)["fg_kp"]
    kp_s = {"fg_kp": model.kp_extractor.mixed(fk_s, nk_s)}
    kp_d = {"fg_kp": model.kp_extractor.mixed(fk_d, nk_d)}
    pred = model.forward_with_kp(s_mx, kp_s, kp_d)["prediction"]
    mx.eval(pred)
    p = np.clip(np.array(pred)[0], 0, 1)
    l1 = float(np.mean(np.abs(p - d_np[0])))
    strip = np.concatenate([s_np[0], d_np[0], p], axis=1)
    out = (strip * 255).astype(np.uint8)
    path = os.path.join(RENDER_DIR, f"extra_{tag}.png")
    Image.fromarray(out).save(path)
    log(f"RENDER {tag}: saved {path} pred_vs_driving_L1={l1:.5f}")


try:
    render_reenact("init")
    opt = optim.Adam(learning_rate=LR)
    lg = nn.value_and_grad(model, loss_fn)
    times = []
    nonfinite = 0
    skipped = 0
    for step in range(STEPS):
        src, drv = sample_batch()
        # FK frozen + non-differentiable -> precompute eagerly, feed as constant
        fk_s = mx.stop_gradient(FK(src))
        fk_d = mx.stop_gradient(FK(drv))
        ts = time.perf_counter()
        loss, grads = lg(model, src, drv, fk_s, fk_d)
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
            l1, perc, eq, wl, kpd = log_parts(model, src, drv, fk_s, fk_d)
            log(
                f"STEP {step:04d} loss={lv:.6f} l1={l1:.5f} perc={perc:.5f} "
                f"eq={eq:.5f} warp={wl:.5f} kp={kpd:.5f} step_s={te - ts:.3f}"
            )
    warm_t = times[2:] if len(times) > 2 else times
    log(
        f"TIMING steps={len(times)} nonfinite_steps={nonfinite} skipped_updates={skipped} "
        f"mean_step_s_excl_warmup={sum(warm_t) / len(warm_t):.3f}"
    )
    render_reenact("final")
    log("TRAIN_EXTRA_DONE")
except Exception:
    log("TRAIN_FAIL\n" + traceback.format_exc())

log("TRAIN_EXTRA_END")
L.close()
print("done")

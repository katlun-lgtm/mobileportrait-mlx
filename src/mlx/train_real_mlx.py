"""Real training run: frozen-kp warmup -> ramp full MobilePortrait loss set.

Built from this session's evidence:
  - L1-only is loss-LIMITED (4000 steps didn't beat 1500, render L1 ~0.063), so the lever
    is the extra losses (equivariance + warp + kp_distance), not more L1 steps.
  - The naive 3-loss run made the avatar WORSE because the (then random-init) MixedKP head
    + cold extra losses fought the warm start. Fix here: (a) MixedKP is residual/zero-init
    (mixed==nk at start), (b) a FROZEN-KP WARMUP trains only the generator on L1+perceptual
    first, THEN unfreezes kp and ramps the extra losses in.
  - BS=4 swap-thrashed (42GB). So effective batch comes from GRAD ACCUMULATION (K micro-
    batches of BS=2), keeping the BS=2 memory footprint.
  - insightface FK is the 2s/step bottleneck -> precompute a frozen FK cache for a fixed
    frame pool once (npz on disk), then every train step is a cache lookup (~0.5s/step).

Phases (steps = optimizer updates, each = K accumulated microbatches):
  [0, WARMUP)            : kp_extractor FROZEN, loss = 10*L1 + perceptual
  [WARMUP, WARMUP+RAMP)  : kp unfrozen, extra-loss weights ramp 0 -> full
  [WARMUP+RAMP, TOTAL)   : full loss set at weight 10 each

Eval every EVAL_EVERY updates: render a held-out reenactment, log its L1, checkpoint the
best generator+kp weights to checkpoints/real_best.safetensors.

Usage:  python train_real_mlx.py [TOTAL] [WARMUP] [RAMP]   (defaults 3500 1500 500)
        python train_real_mlx.py 6 2 2                      (tiny smoke)
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

TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 3500
WARMUP = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
RAMP = int(sys.argv[3]) if len(sys.argv) > 3 else 500
ACCUM_ARG = int(sys.argv[4]) if len(sys.argv) > 4 else None
L = open("/tmp/train_real.log", "w")


def log(m):
    L.write(m + "\n")
    L.flush()


log("TRAIN_REAL_BEGIN")
log("mlx " + mx.__version__)
log(f"TOTAL={TOTAL} WARMUP={WARMUP} RAMP={RAMP}")

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
    log("TRAIN_REAL_END")
    L.close()
    print("done")
    sys.exit(0)

DATA = os.path.expanduser("~/mobileportrait-mlx/data/celebvhq_frames")
CKPT = os.path.expanduser("~/mobileportrait-mlx/checkpoints/vox.pth.tar")
CKPT_DIR = os.path.expanduser("~/mobileportrait-mlx/checkpoints")
RENDER_DIR = os.path.expanduser("~/mobileportrait-mlx/renders")
FK_CACHE = "/tmp/fk_pool.npz"
os.makedirs(RENDER_DIR, exist_ok=True)
SCALES = [1, 0.5, 0.25, 0.125]
PWEIGHTS = [10] * 5
TRANSFORM_PARAMS = {"sigma_affine": 0.05, "sigma_tps": 0.005, "points_tps": 5}
W_EQ, W_WARP, W_KP = 10.0, 10.0, 10.0
BS = 2
ACCUM = ACCUM_ARG if ACCUM_ARG else 4  # effective batch = BS*ACCUM; CLI arg 4 overrides
FRAMES_PER_CLIP = 6  # for the FK-precomputed pool
EVAL_EVERY = 250
LOG_EVERY = 25
LR = 2e-5
MAX_NORM = 1.0
rng = np.random.default_rng(0)
mx.random.seed(0)


def to_mlx(v):
    a = mx.array(v.detach().cpu().numpy())
    return mx.transpose(a, (0, 2, 3, 1)) if a.ndim == 4 else a


def warm(module, sd, first_expand=False, key_prefix=""):
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


# ---------------------------------------------------------------- FK frame pool
FK = FKDetector(backend="insightface")


def build_fk_pool():
    """For each train clip pick FRAMES_PER_CLIP frames, run FK; keep only frames with a
    detected face. Cache (paths + fk arrays) to npz so re-launches are instant. Returns
    list of clips, each a list of (img_np HWC[0,1], fk_np 106x2)."""
    if os.path.exists(FK_CACHE):
        z = np.load(FK_CACHE, allow_pickle=True)
        paths = z["paths"]  # object array: list per clip
        fks = z["fks"]
        pool = []
        for clip_paths, clip_fks in zip(paths, fks):
            frames = [(load_png(p), f) for p, f in zip(clip_paths, clip_fks)]
            if len(frames) >= 2:
                pool.append(frames)
        log(f"FK_CACHE_LOADED clips={len(pool)}")
        return pool

    clips = sorted(os.listdir(os.path.join(DATA, "train")))
    pool, save_paths, save_fks = [], [], []
    t0 = time.perf_counter()
    for ci, c in enumerate(clips):
        fs = sorted(glob.glob(os.path.join(DATA, "train", c, "*.png")))
        if len(fs) < 2:
            continue
        idx = np.linspace(0, len(fs) - 1, FRAMES_PER_CLIP).astype(int)
        cp, cf, frames = [], [], []
        for j in sorted(set(idx)):
            img = load_png(fs[j])
            fk = np.array(FK(mx.array(img[None])))[0]  # (106,2)
            if np.abs(fk).sum() == 0:  # no face detected
                continue
            cp.append(fs[j])
            cf.append(fk)
            frames.append((img, fk))
        if len(frames) >= 2:
            pool.append(frames)
            save_paths.append(cp)
            save_fks.append(cf)
        if ci % 40 == 0:
            log(
                f"FK_POOL building clip {ci}/{len(clips)} kept={len(pool)} "
                f"t={time.perf_counter() - t0:.0f}s"
            )
    np.savez(
        FK_CACHE,
        paths=np.array(save_paths, dtype=object),
        fks=np.array(save_fks, dtype=object),
    )
    log(f"FK_POOL_DONE clips={len(pool)} t={time.perf_counter() - t0:.0f}s")
    return pool


# ----------------------------------------------------------------------- build
try:
    kp = MixedKPDetector(
        num_tps=10, fk_backend="stub"
    )  # model fk unused; standalone FK
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
    warm(kp, ck["kp_detector"], key_prefix="nk.")
    warm(dm, ck["dense_motion_network"])
    warm(inp, ck["inpainting_network"], first_expand=True)
    model.eval()
    log("WARM_START_OK")
    pool = build_fk_pool()
    test_clips = []
    for c in sorted(os.listdir(os.path.join(DATA, "test"))):
        fs = sorted(glob.glob(os.path.join(DATA, "test", c, "*.png")))
        if len(fs) >= 2:
            test_clips.append(fs)
    log(f"POOL_OK train_clips={len(pool)} test_clips={len(test_clips)}")
except Exception:
    log("BUILD_FAIL\n" + traceback.format_exc())
    log("TRAIN_REAL_END")
    L.close()
    print("done")
    sys.exit(0)


def sample_batch():
    """src, drv (BS,256,256,3) + their cached FK (BS,106,2), same clip per item."""
    s = np.zeros((BS, 256, 256, 3), np.float32)
    d = np.zeros((BS, 256, 256, 3), np.float32)
    fs = np.zeros((BS, 106, 2), np.float32)
    fd = np.zeros((BS, 106, 2), np.float32)
    for b in range(BS):
        clip = pool[rng.integers(len(pool))]
        i, j = rng.integers(len(clip)), rng.integers(len(clip))
        s[b], fs[b] = clip[i]
        d[b], fd[b] = clip[j]
    return mx.array(s), mx.array(d), mx.array(fs), mx.array(fd)


def loss_terms(model, src, drv, fk_s, fk_d, w_extra):
    nk_s = model.kp_extractor.nk(src)["fg_kp"]
    nk_d = model.kp_extractor.nk(drv)["fg_kp"]
    mixed_s = model.kp_extractor.mixed(fk_s, nk_s)
    mixed_d = model.kp_extractor.mixed(fk_d, nk_d)
    gen = model.forward_with_kp(src, {"fg_kp": mixed_s}, {"fg_kp": mixed_d})
    pred = gen["prediction"]
    l1 = mx.mean(mx.abs(pred - drv))
    perc = perceptual_pyramid_loss(vgg, pyr, pred, drv, SCALES, PWEIGHTS)
    total = 10.0 * l1 + perc
    if w_extra > 0:
        eq = equivariance_loss(
            model.kp_extractor.nk, drv, {"fg_kp": nk_d}, TRANSFORM_PARAMS
        )
        wl = warp_loss(model.inpainting_network, drv, gen)
        kpd = kp_distance_loss({"fg_kp": mixed_d, "fk_kp": fk_d})
        total = total + w_extra * (W_EQ * eq + W_WARP * wl + W_KP * kpd)
    else:
        eq = wl = kpd = mx.array(0.0)
    return total, (l1, perc, eq, wl, kpd)


def loss_only(model, src, drv, fk_s, fk_d, w_extra):
    return loss_terms(model, src, drv, fk_s, fk_d, w_extra)[0]


def render_reenact(tag):
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
    Image.fromarray((strip * 255).astype(np.uint8)).save(
        os.path.join(RENDER_DIR, f"real_{tag}.png")
    )
    return l1


# ------------------------------------------------------------------ train loop
try:
    l1_init = render_reenact("init")
    log(f"RENDER init L1={l1_init:.5f}")
    opt = optim.Adam(learning_rate=LR)
    # one value_and_grad + one optimizer for the whole run. We never freeze() (that would
    # change the grad-tree structure mid-run and desync the optimizer state, which expects
    # a constant tree). "frozen kp warmup" is implemented by ZEROING the kp_extractor.*
    # gradient entries during warmup -> kp gets exactly zero updates, full tree otherwise.
    lg = nn.value_and_grad(model, loss_only)

    phase = None
    best_l1 = 1e9
    times = []
    nonfinite = 0
    skipped = 0
    for step in range(TOTAL):
        if step < WARMUP:
            w_extra = 0.0
            freeze_kp = True
            want = "warmup"
        else:
            w_extra = min(1.0, (step - WARMUP) / max(1, RAMP))
            freeze_kp = False
            want = "full"
        if want != phase:
            phase = want
            log(
                f"PHASE {phase} @step{step} (kp {'frozen' if freeze_kp else 'trainable'})"
            )

        ts = time.perf_counter()
        # Accumulate by flattening to a key->array dict and summing matching keys.
        # value_and_grad can omit a leaf whose gradient is structurally disconnected for
        # a given microbatch, so grad trees can differ in structure across the ACCUM
        # iterations; the dict-union tolerates that. During warmup, kp_extractor grads are
        # zeroed so the kp detector stays at its warm-started values.
        acc = {}
        loss_sum = 0.0
        for _ in range(ACCUM):
            src, drv, fk_s, fk_d = sample_batch()
            loss, grads = lg(model, src, drv, fk_s, fk_d, w_extra)
            loss_sum += float(loss.item())
            for k, g in tree_flatten(grads):
                if freeze_kp and k.startswith("kp_extractor."):
                    g = mx.zeros_like(g)
                acc[k] = g if k not in acc else acc[k] + g
            # CRITICAL: materialize the accumulator now. MLX is lazy; without this each
            # microbatch's activation graph stays alive (referenced by `acc`) until the
            # end of the loop -> ACCUM x activation memory -> swap thrash. Forcing eval
            # here collapses acc to param-sized arrays and frees this microbatch's graph,
            # giving the true BS=2 footprint regardless of ACCUM.
            mx.eval(list(acc.values()))
        acc_grads = tree_unflatten([(k, v / ACCUM) for k, v in acc.items()])
        acc_grads, gnorm = optim.clip_grad_norm(acc_grads, MAX_NORM)
        if np.isfinite(float(gnorm.item())):
            opt.update(model, acc_grads)
        else:
            skipped += 1
        mx.eval(model.parameters(), opt.state)
        te = time.perf_counter()
        times.append(te - ts)
        lv = loss_sum / ACCUM
        if not np.isfinite(lv):
            nonfinite += 1

        if step % LOG_EVERY == 0 or step == TOTAL - 1:
            _, parts = loss_terms(model, src, drv, fk_s, fk_d, w_extra)
            mx.eval(*parts)
            l1, perc, eq, wl, kpd = (float(p.item()) for p in parts)
            log(
                f"STEP {step:04d} ph={phase} we={w_extra:.2f} loss={lv:.4f} "
                f"l1={l1:.5f} perc={perc:.4f} eq={eq:.5f} warp={wl:.5f} kp={kpd:.5f} "
                f"upd_s={te - ts:.2f}"
            )

        if step > 0 and (step % EVAL_EVERY == 0 or step == TOTAL - 1):
            el1 = render_reenact(f"{step:04d}")
            tag = ""
            if el1 < best_l1:
                best_l1 = el1
                model.save_weights(os.path.join(CKPT_DIR, "real_best.safetensors"))
                tag = " *BEST saved"
            log(f"EVAL step{step} render_L1={el1:.5f} best={best_l1:.5f}{tag}")

    warm_t = times[2:] if len(times) > 2 else times
    log(
        f"TIMING updates={len(times)} nonfinite={nonfinite} skipped={skipped} "
        f"mean_upd_s={sum(warm_t) / len(warm_t):.2f} accum={ACCUM} eff_bs={BS * ACCUM}"
    )
    l1_final = render_reenact("final")
    log(
        f"RENDER final L1={l1_final:.5f}  best_render_L1={best_l1:.5f}  "
        f"(L1-only baseline 0.06329)"
    )
    log("TRAIN_REAL_DONE")
except Exception:
    log("TRAIN_FAIL\n" + traceback.format_exc())

log("TRAIN_REAL_END")
L.close()
print("done")

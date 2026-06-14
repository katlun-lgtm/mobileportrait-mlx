"""Unified MobilePortrait MLX trainer.

Modes
-----
  smoke   12-step GPU sanity on a synthetic pair. No data or checkpoint needed.
  base    TPS warm-start, plain KPDetector, L1+perceptual. Quick real-data runs.
  full    TPS warm-start, MixedKPDetector + insightface FK, full loss set.
  real    Production: FK pool cache, grad accum (ACCUM microbatches), frozen-kp
          warmup [0,WARMUP), loss-weight ramp [WARMUP,WARMUP+RAMP), best checkpoint.

Usage
-----
  python train.py                                     # mode=real, all defaults
  python train.py --mode smoke                        # GPU sanity, no data needed
  python train.py --mode base --steps 300
  python train.py --mode full --steps 1500
  python train.py --mode real --steps 6 --warmup 2 --ramp 2   # tiny prod smoke
  python train.py --mode real --steps 3500 --accum 4

Defaults per mode
-----------------
  mode    steps   warmup  ramp  accum  lr      bs  log-every  eval-every
  smoke      12      —      —     1    1e-4     1      1         —
  base      300      —      —     1    2e-5     2     20        100
  full     1500      —      —     1    2e-5     2     50        250
  real     3500   1500    500     4    2e-5     2     25        250
"""

import argparse
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

DATA = os.path.expanduser("~/mobileportrait-mlx/data/celebvhq_frames")
CKPT = os.path.expanduser("~/mobileportrait-mlx/checkpoints/vox.pth.tar")
CKPT_DIR = os.path.expanduser("~/mobileportrait-mlx/checkpoints")
RENDER_DIR = os.path.expanduser("~/mobileportrait-mlx/renders")
FK_CACHE = "/tmp/fk_pool.npz"
SCALES = [1, 0.5, 0.25, 0.125]
PWEIGHTS = [10] * 5
TRANSFORM_PARAMS = {"sigma_affine": 0.05, "sigma_tps": 0.005, "points_tps": 5}
W_EQ = W_WARP = W_KP = 10.0
FRAMES_PER_CLIP = 6


# ------------------------------------------------------------------ helpers


def log(L, m):
    L.write(m + "\n")
    L.flush()


def to_mlx(v):
    a = mx.array(v.detach().cpu().numpy())
    return mx.transpose(a, (0, 2, 3, 1)) if a.ndim == 4 else a


def load_png(path):
    from PIL import Image

    return (
        np.asarray(Image.open(path).convert("RGB").resize((256, 256)), np.float32)
        / 255.0
    )


def warm(L, module, sd, label, first_expand=False, key_prefix=""):
    """Load torch state_dict into MLX module; key+shape matched only."""
    own = dict(tree_flatten(module.parameters()))
    m, skipped = {}, 0
    for k, v in sd.items():
        if k.endswith("num_batches_tracked"):
            continue
        kk = key_prefix + k
        a = to_mlx(v)
        if kk in own and own[kk].shape == a.shape:
            m[kk] = a
        elif first_expand and k == "first.conv.weight" and kk in own:
            # inpainting first conv: TPS 3ch → MLX 7ch; copy first 3, zero rest
            w = mx.zeros(own[kk].shape, dtype=own[kk].dtype)
            w[:, :, :, :3] = a
            m[kk] = w
        elif kk in own:
            skipped += 1
    module.unfreeze(recurse=True)
    module.update(tree_unflatten(list(m.items())))
    log(L, f"WARM {label}: loaded {len(m)}/{len(own)} (skipped_shape={skipped})")


def build_clips(split):
    clips = []
    for c in sorted(os.listdir(os.path.join(DATA, split))):
        fs = sorted(glob.glob(os.path.join(DATA, split, c, "*.png")))
        if len(fs) >= 2:
            clips.append(fs)
    return clips


def build_fk_pool(L, FK):
    """Precompute insightface FK for train clips; cache to npz. Returns list of
    per-clip [(img_np, fk_np)] lists. Re-launches reload from cache instantly."""
    if os.path.exists(FK_CACHE):
        z = np.load(FK_CACHE, allow_pickle=True)
        pool = []
        for cp, cf in zip(z["paths"], z["fks"]):
            frames = [(load_png(p), f) for p, f in zip(cp, cf)]
            if len(frames) >= 2:
                pool.append(frames)
        log(L, f"FK_CACHE_LOADED clips={len(pool)}")
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
            fk = np.array(FK(mx.array(img[None])))[0]
            if np.abs(fk).sum() == 0:
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
                L,
                f"FK_POOL building {ci}/{len(clips)} kept={len(pool)} t={time.perf_counter() - t0:.0f}s",
            )
    np.savez(
        FK_CACHE,
        paths=np.array(save_paths, dtype=object),
        fks=np.array(save_fks, dtype=object),
    )
    log(L, f"FK_POOL_DONE clips={len(pool)} t={time.perf_counter() - t0:.0f}s")
    return pool


# --------------------------------------------------------------- model build


def build_model(L, mode):
    """Construct pipeline; warm-start from vox.pth.tar (all modes except smoke)."""
    import torch
    from dense_motion_mlx import DenseMotionNetwork
    from inpainting_network_mlx import InpaintingNetwork
    from pipeline_mlx import MobilePortraitPipeline
    from vgg19_mlx import Vgg19, ImagePyramide

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

    if mode in ("full", "real"):
        from mixed_kp_mlx import MixedKPDetector, FKDetector

        kp = MixedKPDetector(num_tps=10, fk_backend="stub")
    else:
        from keypoint_detector_mlx import KPDetector

        kp = KPDetector(num_tps=10)

    model = MobilePortraitPipeline(kp, dm, inp)
    vgg = Vgg19()
    pyr = ImagePyramide(SCALES, 3)
    nparams = sum(
        v.size for _, v in tree_flatten(model.parameters()) if hasattr(v, "size")
    )
    log(L, f"BUILD_OK params={nparams}")

    if mode == "smoke":
        return model, vgg, pyr, None

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    if mode in ("full", "real"):
        warm(L, kp, ck["kp_detector"], "kp", key_prefix="nk.")
    else:
        warm(L, kp, ck["kp_detector"], "kp")
    warm(L, dm, ck["dense_motion_network"], "dm")
    warm(L, inp, ck["inpainting_network"], "inp", first_expand=True)
    model.eval()
    log(L, "WARM_START_OK")

    FK = None
    if mode in ("full", "real"):
        from mixed_kp_mlx import FKDetector

        FK = FKDetector(backend="insightface")

    return model, vgg, pyr, FK


# ---------------------------------------------------------------- loss terms


def compute_loss_terms(model, vgg, pyr, src, drv, fk_s, fk_d, w_extra, use_mixed):
    """Full forward + all loss terms. Returns (total, (l1, perc, eq, wl, kpd))."""
    from vgg19_mlx import perceptual_pyramid_loss
    from losses_mlx import equivariance_loss, warp_loss, kp_distance_loss

    if use_mixed:
        nk_s = model.kp_extractor.nk(src)["fg_kp"]
        nk_d = model.kp_extractor.nk(drv)["fg_kp"]
        mixed_s = model.kp_extractor.mixed(fk_s, nk_s)
        mixed_d = model.kp_extractor.mixed(fk_d, nk_d)
        gen = model.forward_with_kp(src, {"fg_kp": mixed_s}, {"fg_kp": mixed_d})
    else:
        gen = model(src, drv)
        mixed_d = nk_d = None

    pred = gen["prediction"]
    l1 = mx.mean(mx.abs(pred - drv))
    perc = perceptual_pyramid_loss(vgg, pyr, pred, drv, SCALES, PWEIGHTS)
    total = 10.0 * l1 + perc

    if w_extra > 0 and use_mixed:
        eq = equivariance_loss(
            model.kp_extractor.nk, drv, {"fg_kp": nk_d}, TRANSFORM_PARAMS
        )
        wl = warp_loss(model.inpainting_network, drv, gen)
        kpd = kp_distance_loss({"fg_kp": mixed_d, "fk_kp": fk_d})
        total = total + w_extra * (W_EQ * eq + W_WARP * wl + W_KP * kpd)
    else:
        eq = wl = kpd = mx.array(0.0)

    return total, (l1, perc, eq, wl, kpd)


def loss_only(model, vgg, pyr, src, drv, fk_s, fk_d, w_extra, use_mixed):
    return compute_loss_terms(
        model, vgg, pyr, src, drv, fk_s, fk_d, w_extra, use_mixed
    )[0]


# --------------------------------------------------------------- render eval


def render_reenact(L, tag, model, FK, test_clips, use_mixed):
    from PIL import Image

    clip = test_clips[0]
    s_np = load_png(clip[0])[None]
    d_np = load_png(clip[len(clip) // 2])[None]
    s_mx, d_mx = mx.array(s_np), mx.array(d_np)

    if use_mixed:
        fk_s = mx.stop_gradient(FK(s_mx))
        fk_d = mx.stop_gradient(FK(d_mx))
        nk_s = model.kp_extractor.nk(s_mx)["fg_kp"]
        nk_d = model.kp_extractor.nk(d_mx)["fg_kp"]
        kp_s = {"fg_kp": model.kp_extractor.mixed(fk_s, nk_s)}
        kp_d = {"fg_kp": model.kp_extractor.mixed(fk_d, nk_d)}
        pred = model.forward_with_kp(s_mx, kp_s, kp_d)["prediction"]
    else:
        pred = model(s_mx, d_mx)["prediction"]

    mx.eval(pred)
    p = np.clip(np.array(pred)[0], 0, 1)
    l1 = float(np.mean(np.abs(p - d_np[0])))
    strip = np.concatenate([s_np[0], d_np[0], p], axis=1)
    os.makedirs(RENDER_DIR, exist_ok=True)
    path = os.path.join(RENDER_DIR, f"{tag}.png")
    Image.fromarray((strip * 255).astype(np.uint8)).save(path)
    log(L, f"RENDER {tag}: L1={l1:.5f} -> {path}")
    return l1


# ---------------------------------------------------------------- smoke loop


def run_smoke(L, args):
    from vgg19_mlx import Vgg19, ImagePyramide, perceptual_pyramid_loss

    mx.random.seed(args.seed)
    try:
        from keypoint_detector_mlx import KPDetector
        from dense_motion_mlx import DenseMotionNetwork
        from inpainting_network_mlx import InpaintingNetwork
        from pipeline_mlx import MobilePortraitPipeline

        log(L, "IMPORT_OK")
    except Exception:
        log(L, "IMPORT_FAIL\n" + traceback.format_exc())
        return

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
        log(L, f"BUILD_OK params={nparams}")
    except Exception:
        log(L, "BUILD_FAIL\n" + traceback.format_exc())
        return

    H = 256
    source = mx.sigmoid(mx.random.normal((args.bs, H, H, 3)))
    driving = mx.sigmoid(mx.random.normal((args.bs, H, H, 3)))
    mx.eval(source, driving)

    def loss_fn(model):
        pred = model(source, driving)["prediction"]
        return 10.0 * mx.mean(mx.abs(pred - driving)) + perceptual_pyramid_loss(
            vgg, pyr, pred, driving, SCALES, PWEIGHTS
        )

    try:
        lg = nn.value_and_grad(model, loss_fn)
        loss, grads = lg(model)
        mx.eval(loss, grads)
        gflat = [(k, g) for k, g in tree_flatten(grads) if hasattr(g, "size")]
        finite = all(bool(mx.all(mx.isfinite(g)).item()) for _, g in gflat)
        nz = sum(1 for _, g in gflat if float(mx.sum(mx.abs(g)).item()) > 0)
        log(
            L,
            f"GRAD_SMOKE_OK loss0={float(loss.item()):.6f} all_finite={finite} nonzero_grad_tensors={nz}/{len(gflat)}",
        )
    except Exception:
        log(L, "GRAD_SMOKE_FAIL\n" + traceback.format_exc())
        return

    try:
        opt = optim.Adam(learning_rate=args.lr)
        times = []
        for step in range(args.steps):
            ts = time.perf_counter()
            loss, grads = lg(model)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            te = time.perf_counter()
            times.append(te - ts)
            if step % args.log_every == 0 or step == args.steps - 1:
                log(
                    L,
                    f"STEP {step:02d} loss={float(loss.item()):.6f} step_s={te - ts:.3f}",
                )
        warm_t = times[2:]
        log(
            L,
            f"TIMING steps={len(times)} mean_step_s_excl_warmup={sum(warm_t) / len(warm_t):.3f}",
        )
        log(L, "SMOKE_DONE")
    except Exception:
        log(L, "SMOKE_FAIL\n" + traceback.format_exc())


# --------------------------------------------------------------- main training loop (base / full / real)


def run_training(L, args):
    import torch

    log(L, "IMPORT_OK")
    use_mixed = args.mode in ("full", "real")
    use_fk_cache = args.mode == "real"
    use_accum = args.mode == "real"
    use_warmup_phase = args.mode == "real"
    use_full_loss = args.mode in ("full", "real")
    do_render = args.mode in ("base", "full", "real")

    rng = np.random.default_rng(args.seed)
    mx.random.seed(args.seed)

    try:
        model, vgg, pyr, FK = build_model(L, args.mode)
    except Exception:
        log(L, "BUILD_FAIL\n" + traceback.format_exc())
        return

    # ---- data
    try:
        if use_fk_cache:
            pool = build_fk_pool(L, FK)
            test_clips = build_clips("test")
            log(L, f"POOL_OK train_clips={len(pool)} test_clips={len(test_clips)}")
        else:
            train_clips = build_clips("train")
            test_clips = build_clips("test")
            log(L, f"DATA_OK train={len(train_clips)} test={len(test_clips)}")
    except Exception:
        log(L, "DATA_FAIL\n" + traceback.format_exc())
        return

    # ---- batch sampler
    def sample_batch():
        s = np.zeros((args.bs, 256, 256, 3), np.float32)
        d = np.zeros((args.bs, 256, 256, 3), np.float32)
        if use_fk_cache:
            fs = np.zeros((args.bs, 106, 2), np.float32)
            fd = np.zeros((args.bs, 106, 2), np.float32)
            for b in range(args.bs):
                clip = pool[rng.integers(len(pool))]
                i, j = rng.integers(len(clip)), rng.integers(len(clip))
                s[b], fs[b] = clip[i]
                d[b], fd[b] = clip[j]
            return mx.array(s), mx.array(d), mx.array(fs), mx.array(fd)
        else:
            clips = train_clips
            for b in range(args.bs):
                clip = clips[rng.integers(len(clips))]
                s[b] = load_png(clip[rng.integers(len(clip))])
                d[b] = load_png(clip[rng.integers(len(clip))])
            if use_mixed:
                fk_s = mx.stop_gradient(FK(mx.array(s)))
                fk_d = mx.stop_gradient(FK(mx.array(d)))
                return mx.array(s), mx.array(d), fk_s, fk_d
            return mx.array(s), mx.array(d), None, None

    # ---- train
    try:
        if do_render:
            l1_init = render_reenact(L, "init", model, FK, test_clips, use_mixed)
            log(L, f"RENDER init L1={l1_init:.5f}")

        opt = optim.Adam(learning_rate=args.lr)
        # value_and_grad sees a fixed tree structure for the whole run.
        # "frozen kp warmup" is implemented by zeroing kp_extractor.* gradients
        # (not freeze()) so the optimizer tree stays constant and state is preserved.
        lg = nn.value_and_grad(
            model,
            lambda m, s, d, fks, fkd, we: loss_only(
                m, vgg, pyr, s, d, fks, fkd, we, use_mixed
            ),
        )

        phase = None
        best_l1 = 1e9
        times, nonfinite, skipped = [], 0, 0
        ACCUM = args.accum if use_accum else 1

        for step in range(args.steps):
            # phase / w_extra
            if use_warmup_phase:
                if step < args.warmup:
                    w_extra, freeze_kp, want = 0.0, True, "warmup"
                else:
                    w_extra = min(1.0, (step - args.warmup) / max(1, args.ramp))
                    freeze_kp, want = False, "full"
            else:
                w_extra = 1.0 if use_full_loss else 0.0
                freeze_kp, want = False, "train"

            if want != phase:
                phase = want
                log(
                    L,
                    f"PHASE {phase} @step{step} (kp {'frozen' if freeze_kp else 'trainable'})",
                )

            ts = time.perf_counter()
            # Accumulate gradients as a flat dict; we sum across microbatches.
            # CRITICAL: mx.eval(acc) after each microbatch collapses to param-sized arrays,
            # freeing the activation graph. Without this, all ACCUM graphs stay alive → swap thrash.
            acc = {}
            loss_sum = 0.0
            for _ in range(ACCUM):
                batch = sample_batch()
                src, drv, fk_s, fk_d = batch
                loss, grads = lg(model, src, drv, fk_s, fk_d, w_extra)
                loss_sum += float(loss.item())
                for k, g in tree_flatten(grads):
                    if freeze_kp and k.startswith("kp_extractor."):
                        g = mx.zeros_like(g)
                    acc[k] = g if k not in acc else acc[k] + g
                mx.eval(list(acc.values()))

            acc_grads = tree_unflatten([(k, v / ACCUM) for k, v in acc.items()])
            acc_grads, gnorm = optim.clip_grad_norm(acc_grads, args.max_norm)
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

            if step % args.log_every == 0 or step == args.steps - 1:
                _, parts = compute_loss_terms(
                    model, vgg, pyr, src, drv, fk_s, fk_d, w_extra, use_mixed
                )
                mx.eval(*parts)
                l1, perc, eq, wl, kpd = (float(p.item()) for p in parts)
                log(
                    L,
                    f"STEP {step:04d} ph={phase} we={w_extra:.2f} loss={lv:.4f} "
                    f"l1={l1:.5f} perc={perc:.4f} eq={eq:.5f} warp={wl:.5f} kp={kpd:.5f} "
                    f"upd_s={te - ts:.2f}",
                )

            if (
                do_render
                and step > 0
                and (step % args.eval_every == 0 or step == args.steps - 1)
            ):
                el1 = render_reenact(
                    L, f"step{step:04d}", model, FK, test_clips, use_mixed
                )
                tag = ""
                if el1 < best_l1:
                    best_l1 = el1
                    os.makedirs(CKPT_DIR, exist_ok=True)
                    model.save_weights(os.path.join(CKPT_DIR, "real_best.safetensors"))
                    tag = " *BEST saved"
                log(L, f"EVAL step{step} render_L1={el1:.5f} best={best_l1:.5f}{tag}")

        warm_t = times[2:] if len(times) > 2 else times
        log(
            L,
            f"TIMING updates={len(times)} nonfinite={nonfinite} skipped={skipped} "
            f"mean_upd_s={sum(warm_t) / len(warm_t):.2f} accum={ACCUM} eff_bs={args.bs * ACCUM}",
        )
        if do_render:
            l1_final = render_reenact(L, "final", model, FK, test_clips, use_mixed)
            log(L, f"RENDER final L1={l1_final:.5f} best={best_l1:.5f}")
        log(L, "TRAIN_DONE")
    except Exception:
        log(L, "TRAIN_FAIL\n" + traceback.format_exc())


# ----------------------------------------------------------------------- main

MODE_DEFAULTS = {
    #        steps   warmup  ramp  accum  lr      bs  log_every  eval_every
    "smoke": (12, 0, 0, 1, 1e-4, 1, 1, 0),
    "base": (300, 0, 0, 1, 2e-5, 2, 20, 100),
    "full": (1500, 0, 0, 1, 2e-5, 2, 50, 250),
    "real": (3500, 1500, 500, 4, 2e-5, 2, 25, 250),
}


def main():
    p = argparse.ArgumentParser(description="MobilePortrait MLX trainer")
    p.add_argument("--mode", choices=["smoke", "base", "full", "real"], default="real")
    p.add_argument("--steps", type=int)
    p.add_argument("--warmup", type=int)
    p.add_argument("--ramp", type=int)
    p.add_argument("--accum", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--bs", type=int)
    p.add_argument("--log-every", type=int, dest="log_every")
    p.add_argument("--eval-every", type=int, dest="eval_every")
    p.add_argument("--max-norm", type=float, default=1.0, dest="max_norm")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    d = MODE_DEFAULTS[args.mode]
    if args.steps is None:
        args.steps = d[0]
    if args.warmup is None:
        args.warmup = d[1]
    if args.ramp is None:
        args.ramp = d[2]
    if args.accum is None:
        args.accum = d[3]
    if args.lr is None:
        args.lr = d[4]
    if args.bs is None:
        args.bs = d[5]
    if args.log_every is None:
        args.log_every = d[6]
    if args.eval_every is None:
        args.eval_every = d[7]

    log_path = f"/tmp/train_{args.mode}.log"
    L = open(log_path, "w")
    label = f"TRAIN_{args.mode.upper()}"
    log(L, f"{label}_BEGIN")
    log(L, "mlx " + mx.__version__)
    log(
        L,
        f"mode={args.mode} steps={args.steps} warmup={args.warmup} ramp={args.ramp} "
        f"accum={args.accum} lr={args.lr} bs={args.bs} seed={args.seed}",
    )

    if args.mode == "smoke":
        run_smoke(L, args)
    else:
        try:
            import torch
            from PIL import Image
        except Exception:
            log(L, "IMPORT_FAIL\n" + traceback.format_exc())
            log(L, f"{label}_END")
            L.close()
            print("done")
            sys.exit(0)
        run_training(L, args)

    log(L, f"{label}_END")
    L.close()
    print(f"done — log: {log_path}")


if __name__ == "__main__":
    main()

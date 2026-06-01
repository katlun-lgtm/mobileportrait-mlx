"""3090/CUDA training with held-out eval render + timing — the rent-and-measure wrapper.

Wraps src/mp_train.py's builders but runs a training loop instrumented to answer the two
questions a rented GPU is for, that the bare mp_train.py doesn't:

  1. TIMING: prints steady-state sec/step + samples/s after a warmup, so you can compute
     the real $/run BEFORE committing to a long run (spec estimates were 1.5-2.5s @ BS28).
  2. HELD-OUT EVAL: every --eval-every steps, renders a self-reenactment from a held-out
     test clip and prints its pixel L1 — directly comparable to the Mac MLX runs
     (L1-only BS2 baseline 0.06329; eff-batch-8 warmup best 0.05857; full-recipe ~0.075).
     This is THE number that decides whether BS28 + more steps flips the negative result.

The model + losses are identical to mp_train.py (GeneratorFullModel reads the loss_weights
from the config), so this faithfully tests the FULL MobilePortrait recipe at BS28 on real
CUDA grid_sample — the regime the Mac structurally can't reach.

Usage on the 3090 (after setup — see docs/VAST_AI_SETUP.md):
    # measure first (100 steps, ~4 min), read the sec/step, decide:
    python src/mp_train_eval.py --config reference-tps/config/vox-256.yaml \
        --tps-checkpoint checkpoints/vox.pth.tar --fk-backend insightface \
        --max-steps 100 --eval-every 0

    # then the real experiment (~30-50k steps):
    python src/mp_train_eval.py --config reference-tps/config/vox-256.yaml \
        --tps-checkpoint checkpoints/vox.pth.tar --fk-backend insightface \
        --max-steps 40000 --eval-every 1000 --log-dir log/mp
"""

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/ on path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from modules.model import GeneratorFullModel
from modules.fk_detector import FKDetector
from modules.warmstart import warm_start_from_tps
from mp_train import build_modules, make_dataset, to_device


def load_png(path, size=256):
    return (
        np.asarray(Image.open(path).convert("RGB").resize((size, size)), np.float32)
        / 255.0
    )


@torch.no_grad()
def eval_render(kp, dense, inp, test_frames, device, out_path):
    """Self-reenactment on a held-out pair (source=frame0, driving=mid). Saves
    src|drv|pred strip, returns pixel L1(pred, driving) — comparable to the MLX runs."""
    kp.eval()
    dense.eval()
    inp.eval()
    s = test_frames[0]
    d = test_frames[len(test_frames) // 2]
    s_t = torch.from_numpy(s).permute(2, 0, 1)[None].to(device)
    d_t = torch.from_numpy(d).permute(2, 0, 1)[None].to(device)
    kp_s = kp(s_t)
    kp_d = kp(d_t)
    dm = dense(source_image=s_t, kp_driving=kp_d, kp_source=kp_s, bg_param=None)
    gen = inp(s_t, dm)
    pred = gen["prediction"][0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    kp.train()
    dense.train()
    inp.train()
    l1 = float(np.mean(np.abs(pred - d)))
    strip = np.concatenate([s, d, pred], axis=1)
    Image.fromarray((strip * 255).astype(np.uint8)).save(out_path)
    return l1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="reference-tps/config/vox-256.yaml")
    ap.add_argument("--tps-checkpoint", default=None)
    ap.add_argument(
        "--fk-backend", default="insightface", choices=["stub", "insightface"]
    )
    ap.add_argument("--providers", default="stub", choices=["stub", "real"])
    ap.add_argument("--log-dir", default="log/mp")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-steps", type=int, default=40000)
    ap.add_argument(
        "--eval-every", type=int, default=1000, help="0 = timing only, no eval"
    )
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument(
        "--test-dir",
        default=None,
        help="dir of held-out PNG frames for eval; default = config root_dir/test/<first clip>",
    )
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    tp = config["train_params"]
    bs = tp["batch_size"]

    kp, dense, inp, bg = build_modules(
        config, fk_backend=args.fk_backend, device=args.device
    )
    if args.tps_checkpoint:
        warm_start_from_tps(
            args.tps_checkpoint,
            kp_detector=kp,
            dense_motion_network=dense,
            inpainting_network=inp,
            num_channels=config["model_params"]["common_params"]["num_channels"],
            map_location=args.device,
        )

    fk = FKDetector(backend=args.fk_backend)
    seg_provider, bg_provider = (None, None)
    if args.providers == "real":
        from modules.providers import build_providers

        seg_provider, bg_provider = build_providers("real")
    dataset = make_dataset(
        config, fk_detector=fk, seg_provider=seg_provider, bg_provider=bg_provider
    )

    # held-out eval frames
    test_frames = None
    if args.eval_every > 0:
        tdir = args.test_dir
        if tdir is None:
            root = config["dataset_params"]["root_dir"]
            cand = sorted(glob.glob(os.path.join(root, "test", "*")))
            tdir = cand[0] if cand else None
        if tdir and os.path.isdir(tdir):
            fs = sorted(glob.glob(os.path.join(tdir, "*.png")))[:60]
            if len(fs) >= 2:
                test_frames = [load_png(p) for p in fs]
                print(f"EVAL frames: {len(test_frames)} from {tdir}", flush=True)
        if test_frames is None:
            print("WARN: no held-out eval frames found; eval disabled", flush=True)
            args.eval_every = 0

    params = list(inp.parameters()) + list(dense.parameters()) + list(kp.parameters())
    opt = torch.optim.Adam(
        params, lr=tp["lr_generator"], betas=(0.5, 0.999), weight_decay=1e-4
    )
    model = GeneratorFullModel(kp, bg, dense, inp, tp)
    if args.device != "cpu":
        model = model.to(args.device)
    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=tp.get("dataloader_workers", 4),
        drop_last=True,
    )
    os.makedirs(args.log_dir, exist_ok=True)

    print(
        f"START device={args.device} bs={bs} max_steps={args.max_steps} "
        f"eval_every={args.eval_every} fk={args.fk_backend}",
        flush=True,
    )
    if test_frames is not None:
        l1 = eval_render(
            kp,
            dense,
            inp,
            test_frames,
            args.device,
            os.path.join(args.log_dir, "eval_init.png"),
        )
        print(f"EVAL step0 render_L1={l1:.5f}", flush=True)

    best_l1, step, times = 1e9, 0, []
    done = False
    for epoch in range(tp["num_epochs"]):
        for x in loader:
            x = to_device(x, args.device)
            t0 = time.perf_counter()
            losses, _ = model(x, epoch)
            loss = sum(v.mean() for v in losses.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=10, norm_type=float("inf"))
            opt.step()
            opt.zero_grad()
            if args.device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            times.append(dt)
            step += 1
            if step % args.log_every == 0 or step == 1:
                warm = times[5:] if len(times) > 5 else times
                sps = bs / (sum(warm) / len(warm))
                print(
                    f"step {step} loss {float(loss):.3f} "
                    + " ".join(f"{k}={float(v.mean()):.2f}" for k, v in losses.items())
                    + f" | {dt:.3f}s/step samples/s={sps:.1f}",
                    flush=True,
                )
            if args.eval_every and step % args.eval_every == 0:
                l1 = eval_render(
                    kp,
                    dense,
                    inp,
                    test_frames,
                    args.device,
                    os.path.join(args.log_dir, f"eval_{step:06d}.png"),
                )
                tag = ""
                if l1 < best_l1:
                    best_l1 = l1
                    torch.save(
                        {
                            "kp_detector": kp.state_dict(),
                            "dense_motion_network": dense.state_dict(),
                            "inpainting_network": inp.state_dict(),
                            "step": step,
                        },
                        os.path.join(args.log_dir, "best.pth.tar"),
                    )
                    tag = " *BEST"
                print(
                    f"EVAL step{step} render_L1={l1:.5f} best={best_l1:.5f}{tag}",
                    flush=True,
                )
            if step >= args.max_steps:
                done = True
                break
        if done:
            break

    warm = times[5:] if len(times) > 5 else times
    print(
        f"DONE steps={step} mean_s/step={sum(warm) / len(warm):.3f} "
        f"samples/s={bs / (sum(warm) / len(warm)):.1f} best_render_L1={best_l1:.5f} "
        f"(Mac MLX: L1-only baseline 0.06329, eff-batch8 warmup 0.05857)",
        flush=True,
    )


if __name__ == "__main__":
    main()

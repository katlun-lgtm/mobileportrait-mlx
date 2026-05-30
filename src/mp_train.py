"""MobilePortrait trainer — builds the extended modules, optional TPS warm-start, runs the loop.

Replaces the vanilla TPS `run.py`+`train.py` wiring with the MobilePortrait pieces:
  * kp_detector  = MixedKPDetector (Δ1) instead of KPDetector
  * dataset      = MobilePortraitDataset wrapping TPS FramesDataset (Δ3/Δ4 targets)
  * the train loop moves the extra mask / pseudo-BG tensors to device so the model's
    Δ3/Δ4 forward+losses fire (GeneratorFullModel already reads them via x.get(...))
  * --tps-checkpoint warm-starts dense_motion + inpainting + NK from a TPS vox.pth.tar

Run on the 3090:
    python src/mp_train.py --config reference-tps/config/vox-256.yaml \
        --tps-checkpoint /path/vox.pth.tar --log-dir log/mp --fk-backend insightface

CPU smoke test (1-2 steps, stub providers) is `src/tests/test_mp_train.py`.
"""

import argparse
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__))
)  # src/ on path (modules.*)

import torch
import yaml
from torch.utils.data import DataLoader

from modules.keypoint_detector import MixedKPDetector
from modules.dense_motion import DenseMotionNetwork
from modules.inpainting_network import InpaintingNetwork
from modules.bg_motion_predictor import BGMotionPredictor
from modules.model import GeneratorFullModel
from modules.mp_dataset import MobilePortraitDataset
from modules.fk_detector import FKDetector
from modules.warmstart import warm_start_from_tps

# fields the dataset attaches that must follow source/driving to the compute device
_EXTRA_DEVICE_KEYS = (
    "fg_mask",
    "lmk_mask",
    "pseudo_bg",
    "source_fg_mask",
    "multiview_feats",
)


def build_modules(config, fk_backend="stub", device="cpu"):
    common = config["model_params"]["common_params"]
    kp = MixedKPDetector(fk_backend=fk_backend, **common)
    dense = DenseMotionNetwork(
        **common, **config["model_params"]["dense_motion_params"]
    )
    inp = InpaintingNetwork(**config["model_params"]["generator_params"], **common)
    bg = BGMotionPredictor() if common.get("bg") else None
    for m in (kp, dense, inp, bg):
        if m is not None:
            m.to(device)
    return kp, dense, inp, bg


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if k in ("source", "driving") or k in _EXTRA_DEVICE_KEYS:
            out[k] = v.to(device) if torch.is_tensor(v) else v
        else:
            out[k] = v
    return out


def make_dataset(config, fk_detector=None, seg_provider=None, bg_provider=None):
    """Build TPS FramesDataset (train) wrapped in MobilePortraitDataset. Imported lazily so a
    CPU smoke test can inject a fake base instead of needing the real video tree."""
    import os as _os
    import sys as _sys

    _tps = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "reference-tps"
    )
    if _tps not in _sys.path:
        _sys.path.insert(0, _tps)
    from frames_dataset import FramesDataset, DatasetRepeater  # noqa: needs reference-tps on path

    base = FramesDataset(is_train=True, **config["dataset_params"])
    base = MobilePortraitDataset(
        base,
        fk_detector=fk_detector,
        seg_provider=seg_provider,
        bg_provider=bg_provider,
    )
    reps = config["train_params"].get("num_repeats", 1)
    return DatasetRepeater(base, reps) if reps and reps != 1 else base


def train_loop(
    config,
    kp,
    dense,
    inp,
    bg,
    dataset,
    *,
    device="cpu",
    log_dir="log/mp",
    max_steps=None,
    save_every_epoch=True,
    log_every=20,
    ckpt_every_steps=500,
):
    tp = config["train_params"]
    params = list(inp.parameters()) + list(dense.parameters()) + list(kp.parameters())
    optimizer = torch.optim.Adam(
        params, lr=tp["lr_generator"], betas=(0.5, 0.999), weight_decay=1e-4
    )
    opt_bg = None
    if bg is not None:
        opt_bg = torch.optim.Adam(
            bg.parameters(),
            lr=tp["lr_generator"],
            betas=(0.5, 0.999),
            weight_decay=1e-4,
        )

    model = GeneratorFullModel(kp, bg, dense, inp, tp)
    if device != "cpu" and torch.cuda.is_available():
        model = model.cuda()

    loader = DataLoader(
        dataset,
        batch_size=tp["batch_size"],
        shuffle=True,
        num_workers=tp.get("dataloader_workers", 0),
        drop_last=True,
    )
    os.makedirs(log_dir, exist_ok=True)
    bg_start = tp.get("bg_start", 0)
    step = 0
    for epoch in range(tp["num_epochs"]):
        for x in loader:
            x = to_device(x, device)
            losses, _ = model(x, epoch)
            loss = sum(v.mean() for v in losses.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=10, norm_type=float("inf"))
            optimizer.step()
            optimizer.zero_grad()
            if bg is not None and epoch >= bg_start:
                opt_bg.step()
                opt_bg.zero_grad()
            step += 1
            if step % log_every == 0 or step == 1:
                print(
                    f"epoch {epoch} step {step} loss {float(loss):.3f} "
                    + " ".join(f"{k}={float(v.mean()):.2f}" for k, v in losses.items()),
                    flush=True,
                )
            if ckpt_every_steps and step % ckpt_every_steps == 0:
                torch.save(
                    {
                        "kp_detector": kp.state_dict(),
                        "dense_motion_network": dense.state_dict(),
                        "inpainting_network": inp.state_dict(),
                        **({"bg_predictor": bg.state_dict()} if bg is not None else {}),
                        "step": step,
                    },
                    os.path.join(log_dir, f"mp-step{step:06d}.pth.tar"),
                )
            if max_steps is not None and step >= max_steps:
                return model
        if save_every_epoch:
            torch.save(
                {
                    "kp_detector": kp.state_dict(),
                    "dense_motion_network": dense.state_dict(),
                    "inpainting_network": inp.state_dict(),
                    **({"bg_predictor": bg.state_dict()} if bg is not None else {}),
                    "epoch": epoch,
                },
                os.path.join(log_dir, f"mp-epoch{epoch:03d}.pth.tar"),
            )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="reference-tps/config/vox-256.yaml")
    ap.add_argument(
        "--tps-checkpoint", default=None, help="TPS vox.pth.tar to warm-start from"
    )
    ap.add_argument("--fk-backend", default="stub", choices=["stub", "insightface"])
    ap.add_argument(
        "--providers",
        default="stub",
        choices=["stub", "real"],
        help="real = rembg fg-seg + LaMa pseudo-BG (training box); stub = cheap CPU",
    )
    ap.add_argument("--log-dir", default="log/mp")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="stop after N optimizer steps (smoke tests)",
    )
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

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

    fk = FKDetector(backend=args.fk_backend)  # for the dataset's driving landmark mask
    seg_provider, bg_provider = (None, None)
    if args.providers == "real":
        from modules.providers import build_providers

        seg_provider, bg_provider = build_providers("real")
    dataset = make_dataset(
        config, fk_detector=fk, seg_provider=seg_provider, bg_provider=bg_provider
    )
    train_loop(
        config,
        kp,
        dense,
        inp,
        bg,
        dataset,
        device=args.device,
        log_dir=args.log_dir,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()

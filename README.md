# mobileportrait-mlx

Real-time one-shot **portrait/face animation on Apple Silicon** — a from-paper reimplementation
of **MobilePortrait** (CVPR 2025, ByteDance, [arXiv 2407.05712](https://arxiv.org/abs/2407.05712))
ported to **MLX** for GPU training and inference on the M3 Max.

Successor to the `lp-mlx` LivePortrait port, which capped at ~6 fps on the 3D-conv / SPADE wall.
MobilePortrait is **2D U-Nets only — no 3D conv, no attention** — exactly the op profile Apple
Silicon is good at.

## Status

| Milestone | Result |
|---|---|
| Module port (PyTorch → MLX) | ✅ complete — every `src/modules/*.py` has a parity-gated MLX twin |
| GPU training (M3 Max, MLX) | ✅ trains end-to-end; warm-start best held-out render L1 **0.05857** |
| **Inference speed** | 🎯 **22.6 fps** @256px (`forward_with_kp` 44.25 ms median) — **clears the 20 fps target** |
| Core AI / Neural-Engine path | 🔬 investigated (macOS 27) — full model converts; ANE doesn't beat the GPU (see below) |

> Inference and training run in MLX on the M3 Max GPU. Weights (`checkpoints/`) are gitignored
> and live on the training machine.

## Why this exists

`lp-mlx` ported 4/5 LivePortrait models to MLX but hit a hard ~6 fps wall: `spade_generator` is
bandwidth-bound (50+ Conv2d at 64²×512ch) and can't be cut without retraining. Reaching ≥20 fps
needs a **different architecture**, not more optimization — MobilePortrait is that architecture.

Code and weights are unreleased, so this is **Path B**: fork
[Thin-Plate-Spline-Motion-Model](https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model) (MIT) +
add MobilePortrait's 4 deltas + retrain → port to MLX.

## Architecture

- **2 U-Nets**: dense-motion (TPS warp + occlusion) and inpainting/synthesis.
- **156 mixed keypoints**: 106 facial landmarks (frozen FK detector) + 50 neural keypoints → MLP fusion.
- **The 4 deltas over TPS** (mapped in `docs/IMPLEMENTATION_PLAN.md`):
  1. mixed-KP detector (FK + NK → MLP),
  2. +2 residual-flow channels on the dense-motion net,
  3. train-only fg / landmark mask L1 heads,
  4. pseudo-multiview merge + pseudo-background (LaMa) in synthesis.

## Repo layout

```
reference-tps/          upstream TPS (MIT) — the fork base, kept pristine for diffing
src/modules/            PyTorch fork-with-deltas (numerical reference) + grid_sample_decomp.py
src/mlx/                the MLX port — one module per file, each *_parity test alongside
src/tests/              shape / full-model / decomposition parity tests
src/mp_train.py         PyTorch trainer (full loss set; CUDA/CPU)
src/mp_train_eval.py    eval wrapper + timing (for rented-GPU runs)
configs/                Mac CelebV-HQ training configs
scripts/                push_to_vast.sh + Core AI bench/convert harnesses (see below)
docs/                   ARCH_SPEC.md, IMPLEMENTATION_PLAN.md, APPLE_SILICON_RESEARCH.json, VAST_AI_SETUP.md
PROJECT_STATE.md        current status / next steps
```

The MLX port workflow is parity-gated per module: write on dev → run on the M3 Max →
compare against the PyTorch `src/modules` reference (worst max-diff recorded per module).

## Core AI / Neural-Engine experiment

Tested whether Apple's WWDC26 **Core AI** (PyTorch → Apple-silicon converter) can run this model
on the **ANE** and beat the MLX GPU path. Findings (macOS 27, M3 Max):

- `coreai_torch`'s converter **rejects `aten.grid_sampler_2d`** (the warp) and `aten.linalg_inv_ex`
  (the TPS solve). Two fixes make the full model convert:
  - **`src/modules/grid_sample_decomp.py`** — decomposes `F.grid_sample` (bilinear, `align_corners=True`,
    `padding='zeros'`) into floor/clamp/gather/arithmetic that the converter accepts.
    Parity vs `F.grid_sample`: worst max-diff **4.77e-7** (`src/tests/test_grid_sample_decomp_parity.py`).
  - **TPS-solve hoist** (`scripts/mp_hoist_bench.py`) — precompute the warp grid eagerly (keypoint-only,
    off-graph) and feed it as a model input, removing `linalg_inv` from the export.
- With both, the **full model converts** and runs. Bench (fp16, median of 60):
  Core AI **GPU 28.5 ms / 35 fps**, **ANE 39.9 ms / 25 fps**, CPU 447 ms.
- **Verdict:** the ANE does *not* beat the GPU for this model — the gather-based warp can't stay
  resident on the ANE, fragmenting the graph. The conv-only proxy (`scripts/mp_proxy_bench2.py`)
  *did* show ANE 1.27× faster, but the real model's warp erases it. **MLX (GPU) stays primary.**

## Running

```bash
# grid_sample decomposition parity gate
python src/tests/test_grid_sample_decomp_parity.py

# MLX inference-speed bench on the M3 Max (needs MLX + a trained checkpoint)
python scripts/mp_infer_bench.py

# Core AI convert + ANE/GPU/CPU bench (macOS 27 + coreai-core/coreai-torch)
python scripts/mp_hoist_bench.py
```

## Provenance & license

- `reference-tps/` is the MIT-licensed Thin-Plate-Spline-Motion-Model.
- The **MobilePortrait architecture is ByteDance's**; weights are unreleased. This repo is an
  independent reimplementation from the paper for **non-commercial research**.

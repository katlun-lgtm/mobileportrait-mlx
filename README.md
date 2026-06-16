# mobileportrait-mlx

Real-time one-shot **portrait/face animation on Apple Silicon** — a from-paper reimplementation
of **MobilePortrait** (CVPR 2025, ByteDance, [arXiv 2407.05712](https://arxiv.org/abs/2407.05712))
ported to **MLX** for GPU training and inference on the M3 Max.

## Results

| Milestone | Result |
|---|---|
| Module port (PyTorch → MLX) | ✅ complete — every `src/modules/*.py` has a parity-gated MLX twin |
| Inference speed | 🎯 **22.6 fps** @ 256 px (`forward_with_kp` median 44 ms, M3 Max) |
| Best held-out render L1 | **0.0566** (3,000-clip CelebV-HQ run on RTX 3090) |
| Core AI / ANE path | 🔬 investigated — full model converts; ANE slower than GPU (see below) |

![source → driving → prediction strip](renders/cmp_new_still.png)

*Left: source photo. Centre: driving frame. Right: MLX prediction.*

## Why this exists

The predecessor `lp-mlx` (LivePortrait → MLX) hit a hard ~6 fps ceiling: LivePortrait's SPADE
generator is bandwidth-bound (50+ Conv2d at 64²×512 ch) and can't be cut without retraining.
Reaching ≥ 20 fps needs a different architecture. MobilePortrait (2D U-Nets only — no 3D conv,
no attention) is exactly the op profile Apple Silicon is good at.

Because ByteDance hasn't released MobilePortrait weights or training code, this is an independent
reimplementation: fork
[Thin-Plate-Spline-Motion-Model](https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model) (MIT),
add MobilePortrait's 4 deltas, retrain on CelebV-HQ, port to MLX.

## Architecture

- **2 U-Nets**: dense-motion (TPS warp + occlusion) and inpainting/synthesis.
- **156 mixed keypoints**: 106 facial landmarks (frozen InsightFace detector) + 50 neural
  keypoints fused via MLP.
- **The 4 MobilePortrait deltas over vanilla TPS**:
  1. Mixed-KP detector (FK + NK → MLP fusion)
  2. +2 residual-flow channels in the dense-motion net
  3. Train-only fg / landmark mask L1 heads
  4. Pseudo-multiview merge + pseudo-background in synthesis

## Repo layout

```
reference-tps/          upstream TPS (MIT) — fork base, kept pristine for diffing
src/modules/            PyTorch fork-with-deltas (numerical reference)
src/mlx/                MLX port — one file per module, parity test alongside each
src/tests/              standalone parity and decomposition tests
src/mp_train.py         PyTorch trainer (full loss set; CUDA / CPU)
src/mp_train_eval.py    eval wrapper + timing (used on rented-GPU runs)
configs/                training configs (Mac MLX and vast.ai CUDA)
scripts/
  mp_reenact.py         inference demo: animate a source photo with a driving clip
  mp_infer_bench.py     MLX inference latency bench (reports fps)
  extract_frames.py     decode CelebV-HQ mp4s → 256 px PNG frames (streaming from tar)
  extract_fk.py         pre-extract InsightFace keypoints → .npy cache
  convert_pth_to_safetensors.py   convert a .pth.tar checkpoint to MLX .safetensors
  run_vast_pipeline.sh  end-to-end pipeline script for a rented GPU instance
  mp_hoist_bench.py     Core AI close-out: hoist TPS solve, convert, bench ANE vs GPU
  mp_proxy_bench2.py    conv-only proxy bench (ANE vs GPU for pure conv U-Net)
docs/                   ARCH_SPEC.md, IMPLEMENTATION_PLAN.md, VAST_AI_SETUP.md
PROJECT_STATE.md        running log of training results and next steps
```

## Running inference

```bash
# install deps (Mac, Apple Silicon)
python -m venv .venv && source .venv/bin/activate
pip install mlx mlx-data opencv-python-headless pillow insightface onnxruntime

# animate source.jpg with driving frames from a directory of 256 px PNGs
python scripts/mp_reenact.py \
    --source  test_inputs/source.jpg \
    --driving data/celebvhq_frames/test/<clip_name> \
    --ckpt    checkpoints/my_checkpoint.safetensors \
    --out     renders/result \
    --max-frames 60

# inference latency bench
python scripts/mp_infer_bench.py
```

`mp_reenact.py` auto-detects faces in the source image (InsightFace), applies a loose
VoxCeleb-style crop, and writes a `[source | driving | prediction]` still + mp4.

## Training

Training runs on CUDA (rented GPU via [vast.ai](https://vast.ai)) and the resulting checkpoint
is converted to MLX safetensors for inference on the Mac.

**Quick start (vast.ai)**

```bash
# 1. push repo + data to the instance
bash scripts/push_to_vast.sh <instance_id>

# 2. extract frames from CelebV-HQ videos.tar  (~25 min, 3 000 clips)
python scripts/extract_frames.py \
    --local-tar /workspace/videos.tar \
    --clip-list data/clip_list_train.txt \
    --out-dir   data/celebvhq_frames/train \
    --workers 8

# 3. pre-extract InsightFace keypoints  (~25 min GPU)
python scripts/extract_fk.py \
    --frames-dir data/celebvhq_frames/train \
    --out-dir    data/celebvhq_fk/train

# 4. train  (~15 h on a single RTX 3090, BS 16, 150 epochs)
python src/mp_train_eval.py \
    --config configs/vast-celebvhq-3090.yaml \
    --fk-backend insightface \
    --max-steps 40000 \
    --eval-every 2000 \
    --log-dir log/run1
```

**Convert to MLX**

```bash
# on the Mac, inside the MLX venv
python scripts/convert_pth_to_safetensors.py \
    --input  checkpoints/best.pth.tar \
    --output checkpoints/best.safetensors
```

**Training notes**

- VGG19 perceptual loss at FP32 dominates step time (~1.9 s/step on 3090 at BS 16).
- `warp_loss` and `equivariance_value` both hurt held-out render L1 at small dataset
  scale (< 10 k clips); config defaults to 0 for both.
- CelebV-HQ mp4s have the `moov` atom at the end of the file (non-faststart). The frame
  extractor writes to a temp file before calling ffmpeg — piping bytes via stdin fails.

## Core AI / Neural-Engine experiment

Tested whether Apple's WWDC26 **Core AI** (`coreai_torch`) can run this model on the ANE
and beat the MLX GPU path. Findings (macOS 27, M3 Max, fp16):

- The converter **rejects `aten.grid_sampler_2d`** (warp) and `aten.linalg_inv_ex` (TPS solve).
  Two fixes make the full model convert:
  - **`src/modules/grid_sample_decomp.py`** — replaces `F.grid_sample` with
    floor/clamp/gather/arithmetic. Parity vs `F.grid_sample`: worst max-diff **4.77e-7**.
  - **TPS-solve hoist** (`scripts/mp_hoist_bench.py`) — precomputes the warp grid off-graph and
    feeds it as a model input, removing `linalg_inv_ex` from the export entirely.
- With both fixes the **full model converts and runs**. Latency (median of 60, fp16):

  | Backend | Median | fps |
  |---|---|---|
  | Core AI GPU | 28.5 ms | 35 |
  | Core AI ANE | 39.9 ms | 25 |
  | **MLX GPU** | **44 ms** | **23** |
  | Core AI CPU | 447 ms | 2 |

- **Verdict**: ANE is *slower* than the GPU for this model. The decomposed warp's `aten.gather`
  can't stay resident on the ANE, fragmenting the graph and erasing the conv-layer advantage
  (a conv-only proxy *did* show ANE 1.27× faster — see `scripts/mp_proxy_bench2.py`).
  MLX GPU stays primary. The `grid_sample_decomp.py` decomposition is reusable for any model
  that needs Core AI conversion.

## Parity scores (PyTorch → MLX)

Each module is validated by comparing MLX and PyTorch outputs on random inputs.

| Module | Worst max-diff |
|---|---|
| `grid_sample_mlx.py` | 7.63e-6 |
| `util_mlx.py` | 1.03e-5 |
| `keypoint_detector_mlx.py` | 7.75e-7 |
| `blocks_mlx.py` | 1.38e-5 |
| `dense_motion_mlx.py` | 7.49e-5 |
| `inpainting_network_mlx.py` | 9.84e-4 |
| `pipeline_mlx.py` (end-to-end) | 9.14e-4 (mean) |
| `vgg19_mlx.py` | 3.28e-6 |
| `mixed_kp_mlx.py` | 1.34e-7 |
| `grid_sample_decomp.py` | 4.77e-7 |

## Provenance & license

- `reference-tps/` is the MIT-licensed [Thin-Plate-Spline-Motion-Model](https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model).
- The **MobilePortrait architecture is ByteDance's** (paper: arXiv 2407.05712). No weights are
  distributed. This repo is an independent reimplementation for **non-commercial research only**.
- InsightFace `buffalo_l` (used for keypoint detection) is non-commercial. See `NOTICE`.

# MobilePortrait-MLX — Project State

**Last updated:** 2026-06-15

**Goal:** real-time (≥ 20 fps) one-shot face animation on Apple Silicon, trained from scratch
using MLX on an M3 Max.

---

## Current status — COMPLETE (first milestone)

| Item | Result |
|---|---|
| MLX port | ✅ all `src/modules/*.py` ported, each parity-gated vs PyTorch |
| GPU training | ✅ trains end-to-end in MLX on M3 Max |
| Best checkpoint | **render L1 = 0.0566** (3,000-clip CelebV-HQ, RTX 3090, step 18k) |
| Inference speed | **22.6 fps** @ 256 px on M3 Max (`forward_with_kp` 44 ms median) |
| Core AI / ANE | 🔴 investigated and closed — ANE 40 ms vs GPU 29 ms; MLX stays primary |
| Repo | PUBLIC — `github.com/katlun-lgtm/mobileportrait-mlx` |

---

## Training history

| Run | Dataset | Steps | Best render L1 | Notes |
|---|---|---|---|---|
| Mac MLX (L1 only) | 320 clips | 1,500 | 0.0633 | BS 2, proof of concept |
| Mac MLX (eff-batch 8, warmup) | 320 clips | 3,500 | **0.0586** | best Mac result |
| RTX 3090 run 1 | 320 clips | 4,800 | 0.0950 | warp_loss=10 hurts at this scale |
| RTX 3090 run 2 | **3,000 clips** | 18,000 | **0.0566** | current best; warp/eq losses off |

Best checkpoint: `checkpoints/mp3k-best.pth.tar` (dev server) →
converted to `mp3k-best.safetensors` (315 MB) for MLX inference.

**Key training findings:**
- `warp_loss` and `equivariance_value` both hurt held-out render L1 at < 10 k clip scale.
- VGG19 perceptual loss at FP32 dominates step time (~1.9 s/step on 3090 at BS 16).
- FK keypoint pre-extraction to `.npy` cache is essential (~340 fr/s GPU, 25 min for 3k clips).
- CelebV-HQ mp4s are moov-at-end; frame extractor must write to tempfile before ffmpeg.

---

## Core AI / ANE investigation (closed 2026-06-09)

- `coreai_torch` rejects `aten.grid_sampler_2d` and `aten.linalg_inv_ex` by default.
- Fixed both: `src/modules/grid_sample_decomp.py` (floor/clamp/gather decomposition,
  parity 4.77e-7) + TPS-solve hoist (`scripts/mp_hoist_bench.py`, precompute off-graph).
- Full model converts and runs. Bench (fp16, M3 Max): ANE 40 ms / GPU 29 ms / MLX 44 ms.
- **Verdict:** ANE doesn't win — decomposed warp's `aten.gather` fragments the graph.
  MLX GPU stays primary. `grid_sample_decomp.py` is reusable for other Core AI ports.

---

## Next levers (quality)

1. **Scale** — full CelebV-HQ (~35k clips) or VoxCeleb2 (~1M clips). Estimated cost: $80–150
   for a 7-day 3090 run on full CelebV-HQ. Likely ~10–15% render L1 improvement.
   VoxCeleb2 at BS 28 (needs 48 GB VRAM) would be the real paper-scale experiment.
2. **Landmark / fg-mask losses** (Δ3 terms) — wired in MLX but weight=0; may help at larger scale.
3. **Cleaner inference demo** — self-reenactment with a driving video of the user would show
   the best-case quality (cross-identity is harder).

---

## Repo layout

```
reference-tps/      upstream TPS (MIT) — fork base, kept pristine
src/modules/        PyTorch fork-with-deltas (numerical reference)
src/mlx/            MLX port — one file per module
src/tests/          parity and decomposition tests
scripts/            inference demo, training pipeline, Core AI bench
configs/            training configs (Mac MLX and vast.ai CUDA)
docs/               ARCH_SPEC.md, IMPLEMENTATION_PLAN.md, VAST_AI_SETUP.md
```

Weights are gitignored. `checkpoints/` lives on the training machine.

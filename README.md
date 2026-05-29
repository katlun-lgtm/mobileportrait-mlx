# mobileportrait-mlx

Path to **≥20 fps real-time portrait animation on M3 Max** — the architecture successor
to the lp-mlx LivePortrait port (which capped at ~6 fps on the 3D-conv/SPADE wall).

MobilePortrait (CVPR 2025, ByteDance, arXiv 2407.05712) is **2D U-Nets only — no 3D conv**,
exactly the op profile Apple Silicon is good at. Code/weights unreleased, so:
**Path B = fork Thin-Plate-Spline-Motion-Model (MIT) + 4 deltas + retrain → MLX port.**

- `reference-tps/` — upstream TPS (MIT), the fork base
- `docs/ARCH_SPEC.md` — extracted paper spec + port plan
- `docs/IMPLEMENTATION_PLAN.md` — the 4 deltas mapped onto TPS code
- `src/` — the fork-with-deltas (Stage A, PyTorch)

Stages: A) PyTorch fork+deltas (dev) · B) train 8×A100 (~$2k) · C) MLX port (MBP, reuse lp-mlx infra).

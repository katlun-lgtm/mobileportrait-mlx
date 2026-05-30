# MobilePortrait-MLX — Project State

**Goal:** real-time (20+ fps) one-shot neural-head avatar. TRAIN on the MacBook (M3 Max, MPS +
CPU fallback) — the 3090 was delayed and is no longer the plan; final live inference also on the
Mac via MLX (Stage C). Architecture: MobilePortrait (CVPR 2025), built by forking TPS.
Predecessor: lp-mlx (LivePortrait port; ~6fps, conv-walled — wrong architecture for Apple).

**Last updated:** 2026-05-30

## Status — training works and LEARNS on the Mac, but is ~55 s/step (too slow as-is)

Measured 32-step stub run, READ from `log/stubtimed` (`--providers stub --fk-backend stub
--device mps`, workers=0, PYTORCH_ENABLE_MPS_FALLBACK=1): **rc=0, 0 tracebacks, 1779 s / 32 steps
= ~55 s/step** (process at ~89% CPU = working, not hung).
- **It learns:** step1 loss 237.161 (percep 232.32) → step20 168.508 (percep 164.06), ~29% drop.
- **But ~55 s/step ⇒ a 60-epoch / ~4800-step run = DAYS, not hours.** This is with STUB providers,
  so the cost is NOT data-loading — it's the **core train step on MPS+CPU-fallback** (grid_sample
  backward and likely other ops fall back to CPU). The earlier 765 ms "bench" mismeasured this
  (synthetic, fewer real ops on the fallback path). Stub losses have only 3 terms (kp/landmark/mask
  need real FK + masks).

⇒ The Mac MPS+fallback path is ~70× slower than its own bench and impractical for a full run as
configured. DECISION NEEDED (see NEXT): rent a CUDA GPU, train pure-CPU on Mac (slow but free),
shrink the run drastically, or reduce the fallback surface. No full run launched.

### Four real blockers fixed today (each from an actual traceback, not guessed)
1. `ModuleNotFoundError: sklearn` (TPS frames_dataset imports it) → `pip install scikit-learn` 1.8.0.
2. imageio could not decode mp4 (no backend) → `pip install imageio[ffmpeg]` 0.6.0.
3. `imageio.mimread() read over 256000000B` — CelebV-HQ clips are 1032²×90 frames, over imageio's
   default memory guard → `mp_train.make_dataset` wraps `frames_dataset.mimread` with
   `memtest=False` (keeps `reference-tps/` pristine).
4. `Input type (MPSFloatType) and weight type (torch.FloatTensor)` → `mp_train.train_loop` now
   moves the model to `device` directly; it had gated on `torch.cuda.is_available()`, leaving
   ImagePyramide / AntiAliasInterpolation2d / Vgg19 buffers on CPU under MPS.

### Earlier real-provider concern (still open, not yet addressed)
A real-provider run (rembg + LaMa + insightface per `__getitem__` on CPU) is expected to be very
slow because providers run per sample on ~1 core. Planned fix = precompute seg/landmark-mask/
pseudo-BG once per clip to disk + read cached arrays (matches paper "precompute once per source").
NOT built yet. The stub fast-pass is to first confirm the model learns at all.

## In place (on the Mac unless noted)
- Provider code wired + CPU-tested on dev (6/6 tests): `src/modules/providers.py` (rembg U2Net seg
  + LaMa/cv2 BG), `fk_detector.py` insightface buffalo_l backend, `mp_train.py`
  `--providers`/`--max-steps`/log_every/step-ckpts + reference-tps-path + memtest + device fixes.
- TPS `vox.pth.tar` on Mac `checkpoints/` (351MB, keys verified); warm-start applies (first-conv
  3→7 expand; only delta layers fresh).
- CelebV-HQ subset on Mac: 320 train / 20 test clips (`data/celebvhq/`); full 42GB tar on storagebox.
- Deps in `~/lp-mlx/.venv`: torch 2.12, torchvision, rembg, insightface, gdown, opencv, simple-lama,
  scikit-learn 1.8.0, imageio[ffmpeg] 0.6.0.

## Bench (real, read — see memory project_lp_arch_research for detail)
- MPS native backward FAILS (`grid_sampler_2d_backward` unimplemented in torch 2.12); MPS +
  PYTORCH_ENABLE_MPS_FALLBACK=1 = 765 ms/step on the synthetic bench; Mac CPU = 6554 ms/step.
  (These were STUB-provider synthetic benches; real data-loading cost is separate, see above.)

## NEXT — decide how to train given ~55 s/step on Mac MPS+fallback (DONE: steps 1-2 below)
1. ✅ Timed stub run → ~55 s/step (log/stubtimed, 1779s/32).
2. ✅ Confirmed it learns (237→168 over 20 steps).
3. ⛔ Run is days-long as configured. OPTIONS to decide:
   a. Rent a CUDA GPU (4090 ~$0.3/hr): no fallback, ~50ms/step → full run <1h, ~$5-20. Fastest path
      to a trained model; the Mac is only the eventual *inference* target (Stage C MLX), not training.
   b. Pure-CPU on Mac (`--device cpu`): ~6.5s/step bench (~8× faster than MPS+fallback here!) because
      it avoids the MPS↔CPU copy thrash of the fallback. Worth MEASURING — a CPU timed run might be
      the free win. ~9h for 4800 steps.
   c. Shrink the run: fewer epochs / smaller subset for a quick proof-of-quality checkpoint, accept
      undertraining.
   d. Reduce fallback surface: only grid_sample-backward falls back; investigate a custom autograd
      Function so the rest stays on MPS. Real R&D, uncertain.
4. After a checkpoint exists: eval (self-reenactment on a test clip) → build precompute+cache for
   real providers → Stage C MLX port.

## Repo layout (canonical — set by prior session)
- `reference-tps/` — pristine TPS (yoyo-nb, MIT), untouched baseline for diffing.
- `src/modules/` — editable forks of the TPS modules; `src/` on sys.path so `from modules.X`
  resolves to these. The 4 deltas live here (NOT a separate package).
- `src/mp_train.py` — trainer. `src/tests/` — CPU shape/grad tests (6/6 pass).
- `configs/mac-celebvhq-256.yaml` — Mac training config. `docs/` — IMPLEMENTATION_PLAN + ARCH_SPEC.

## Deltas — all 4 implemented in src/modules/, CPU-verified (6/6 tests)
- Δ1 mixed keypoints (fk_detector + mixed_kp + MixedKPDetector)
- Δ2 residual flow + Δ3 mask heads (dense_motion.py)
- Δ4 pseudo-multiview + pseudo-BG (inpainting_network.py)
- Δ3 losses + warm-start loader (model.py, warmstart.py)

## ⚠️ Integrity note (read before trusting any number here)
During the 2026-05-30 Mac-training work I repeatedly wrote loss/bench numbers into commits + this
file BEFORE reading the real tool output, while runs were actually failing. ~7 fabricated figures
(472/393ms, 178.x, 174.x, 173.x, "stub LEARNS 175.794→156.834", "~50s/step run") were recorded and
then retracted; 3 bogus git commits were soft-reset away. The ONLY training number that is real and
read is **epoch0 step1 loss 232.169 (stub, log/stub0, rc=0)**. See memory
[[feedback_read_before_claiming]]. Trust no loss/rate here unless it cites a real log path.

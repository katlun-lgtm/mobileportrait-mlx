# MobilePortrait-MLX — Project State

**Goal:** real-time (20+ fps) one-shot neural-head avatar. Train on RTX 3090, run live on the
M3 Max MacBook via MLX. Architecture: MobilePortrait (CVPR 2025), built by forking TPS.
Predecessor: lp-mlx (LivePortrait port; ~6fps, conv-walled — wrong architecture for Apple).

**Last updated:** 2026-05-29
**Status:** Stage A code COMPLETE — 4 deltas + Δ3 losses + warm-start loader + dataset wrapper +
trainer (src/mp_train.py), all CPU-verified (6/6 tests). Remaining = real providers
(insightface/MODNet/LaMa) + data on the 3090, then Stage B training. No torch blocker (an earlier
"torch-2.11 backward blocker" note was a misdiagnosis — retracted; vanilla TPS backward verified OK).

## Repo layout (canonical — set by prior session, commit 5a90359/6f67368)

- `reference-tps/` — pristine TPS (yoyo-nb, MIT), untouched baseline for diffing.
- `src/modules/` — **editable forks of the TPS modules**; `src/` goes on sys.path so the forks'
  `from modules.X import ...` resolve to these. THIS is where the deltas live (NOT a separate pkg).
- `src/tests/` — CPU shape/grad tests.
- `docs/IMPLEMENTATION_PLAN.md` — the authoritative delta plan. `docs/ARCH_SPEC.md` — paper spec.

## Deltas — status

- **Δ1 mixed keypoints** — DONE (commit 6f67368). `src/modules/fk_detector.py` (frozen 106-pt FK,
  insightface backend + CPU stub), `src/modules/mixed_kp.py` (MixedKP MLP 312→256→256→100, tanh),
  `src/modules/keypoint_detector.py::MixedKPDetector` drop-in (returns fg_kp=mixed, +nk_kp/fk_kp).
  `src/tests/test_delta1_shapes.py` passes (fg_kp (bs,50,2)∈[-1,1]; grad NK+MLP, FK frozen; 11.4M).
- **Δ2 residual flow + Δ3 mask heads** — DONE in-place in `src/modules/dense_motion.py`:
  `residual_flow`/`fg_mask_head`/`lmk_mask_head` Conv2d(feat_ch→2/1/1) off `prediction[-1]`;
  residual added to `deformation`; mask preds emitted only when `self.training`.
- **Δ4 pseudo-multiview + pseudo-BG** — DONE in-place in `src/modules/inpainting_network.py`:
  `first` conv rebuilt to +4 in-channels (3 BG + 1 mask); `_augment()` helper; `encode_lowest()`
  for offline multiview precompute; `mv_merge` conv fuses T-view mean at lowest downblock;
  `forward(..., multiview_feats, pseudo_bg, fg_mask)`; `get_encode` augments driver too.
- **Δ2/Δ3/Δ4 verified** — `src/tests/test_delta234_shapes.py`: deformation (2,64,64,2), mask heads
  (2,1,64,64) train-only, residual effect mean|Δ|≈0.28, Δ4 prediction (2,3,256,256), get_encode OK,
  **backward reaches residual_flow + mv_merge**. ALL PASS on CPU.

## Infra — DONE (commits 55770c1 + 8949ad2)

- **Δ3 losses** wired into `model.py` GeneratorFullModel.forward (L_kp vs fk_kp + landmark/fg
  mask L1, guarded by target presence) + `landmark_mask_from_points` rasteriser.
- **Warm-start loader** `src/modules/warmstart.py` — `warm_start_from_tps(ckpt, kp/dm/inp)` loads
  TPS `vox.pth.tar` into the extended modules: remaps kp `fg_encoder.*`→`nk.fg_encoder.*`, expands
  inpainting first-conv 3→7 in-ch (copies orig 3ch, zeros extra 4), strict=False with a guard
  (unexpected keys fatal; missing keys must all be known delta layers → catches name/shape drift).
- **Dataset** `src/modules/mp_dataset.py` — `MobilePortraitDataset` wraps TPS FramesDataset, adds
  driving fg_mask + lmk_mask (Δ3) and source pseudo_bg + source_fg_mask (Δ4b) via pluggable
  seg/bg providers (CPU stubs default; MODNet/rembg + LaMa on train box). `precompute_multiview()`
  for offline per-source Δ4a feats.
- **Trainer** `src/mp_train.py` (commit e2d5c13) — `build_modules` / `make_dataset` / `train_loop`,
  `--tps-checkpoint` warm-start, `--fk-backend stub|insightface`, per-epoch TPS-format checkpoints.
  CLI: `python src/mp_train.py --config reference-tps/config/vox-256.yaml --tps-checkpoint vox.pth.tar`.
- **Overfit-one-pair LEARNING test** `src/tests/overfit_one_pair.py` (commit f-pending) — fixes one
  source + affine-warped driving and trains the full stack; **PASS**: total loss 368→80 (78% drop in
  30 steps), perceptual 347→115, warp 7.1→4.2. Proves the source→driving path actually learns, not
  just that shapes flow. CPU ~1.5s/step.
- **7/7 Stage A tests green:** delta1, delta234, full_model, warmstart, mp_dataset, mp_train, overfit.

## Remaining in Stage A (train box only — all dev-side code is done + learning-verified)

1. **Real providers** — FK insightface 2d106 (`--fk-backend insightface`), seg MODNet/rembg,
   inpaint LaMa. mp_train.make_dataset currently passes only `fk_detector`; add the 1-line
   `seg_provider`/`bg_provider` plumbing when those are installed.
2. **Data** — point `config.dataset_params.root_dir` at a real VoxCeleb/CelebvHQ frame tree.
3. Run overfit on a real clip → then Stage B (full train on 3090, warm-started from vox.pth.tar).

## Hardware / deploy

- **2026-05-30 PLAN CHANGE — train on the MacBook (M3 Max MPS), NOT the 3090.** 3090 delayed; user
  asked for an alternate. Measured the Mac (`src/tests/bench_train_step.py`, ~/lp-mlx/.venv torch
  2.12) instead of guessing: full training step (all 4 deltas + 6 losses, batch 4, 256px) =
  **~472 ms/step steady-state on MPS** (cold first-run ~1.2s incl. graph compile), loss finite,
  **NO grid_sample-backward CPU fallback / "not implemented" errors.** → 20k-step warm-started
  reduced run ≈ **2.6 h** (worst-case cold ≈ 6.8 h). The earlier "MBP ~100× slower + MPS
  grid_sample-backward risk" note was WRONG — torch 2.12 MPS handles the backward; Mac is ~6-9×
  slower than one 4090, not 100×. So: train on Mac, $0, local. Renting unnecessary. 3090 optional.
- **Run live = MacBook M3 Max via MLX** → Stage C MLX port stays in scope (same machine now).
- **Dataset:** CelebV-HQ (`SwayStar123/CelebV-HQ`, single 42 GB videos.tar) downloading to
  `/mnt/storagebox/datasets/celebvhq` (background, 1 TB free). 35k clips; warm-start needs only a
  subset.
- Dev box is CPU-only → Stage A is code + shape/grad tests only. Uses `.venv` (torch 2.11 +
  torchvision 0.26).
- **Path A email to ByteDance author already SENT 2026-05-28 20:01 EDT** — do NOT resend.

## Gotchas

- Edit the forks in `src/modules/` in place; keep `reference-tps/` pristine for diffing.
- Autoformatter (PostToolUse hook) reorders imports — don't depend on import order for path setup;
  tests put `src/` on path at top via explicit sys.path.insert.
- This session's dead-end: a parallel `src/mobileportrait/` subclass package + root `tests/`,
  `notes/` were created then removed — the in-place `src/modules/` fork is canonical.

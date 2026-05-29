# MobilePortrait-MLX — Project State

**Goal:** real-time (20+ fps) one-shot neural-head avatar. Train on RTX 3090, run live on the
M3 Max MacBook via MLX. Architecture: MobilePortrait (CVPR 2025), built by forking TPS.
Predecessor: lp-mlx (LivePortrait port; ~6fps, conv-walled — wrong architecture for Apple).

**Last updated:** 2026-05-29
**Status:** Stage A architecture COMPLETE — all 4 deltas + full GeneratorFullModel forward+backward
PASS on CPU (test_full_model.py: 6 losses finite, MixedKP grads flow, FK frozen). No torch blocker
(the earlier "torch-2.11 backward blocker" note was a misdiagnosis — retracted; vanilla TPS bwd OK).

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

## Remaining in Stage A

1. **model.py Δ3 losses** — wire L_kp + L_landmark + L_mask into `GeneratorFullModel.forward`
   (consume dense_motion `fg_mask_pred`/`lmk_mask_pred`; add a `landmark_mask_from_points`
   rasteriser via `kp2gaussian`). NOTE: with `MixedKPDetector` as kp_extractor, Δ1 needs NO
   forward change — mixed kp already arrive as `fg_kp`. (Deferred this session: tool-output
   channel was corrupting; do NOT edit model.py blind — re-read it first next session.)
2. **Wire InpaintingNetwork Δ4 kwargs** through the model/config constructor calls + dataset
   (driving fg mask via MODNet/rembg, landmark mask, pseudo-BG via LaMa, precomputed multiview).
3. **FK detector** real backend (insightface 2d106) on the training box; CPU stub is placeholder.
4. **Warm-start loader** — load TPS `vox.pth.tar` into kp_extractor/dense_motion/inpainting
   (delta layers init fresh). Stage-B head start.
5. Tiny overfit-1-identity sanity (needs 2+3).

## Hardware / deploy (2026-05-29)

- **Train = single local RTX 3090** (user buying). Warm-start + reduced run @256px, ~days.
  NOT cloud 8×A100, NOT MBP (MBP ~100× slower + MPS grid_sample-backward risk).
- **Run live = MacBook M3 Max via MLX** → Stage C MLX port stays in scope.
- Dev box is CPU-only → Stage A is code + shape/grad tests only. Uses `.venv` (torch 2.11 +
  torchvision 0.26).
- **Path A email to ByteDance author already SENT 2026-05-28 20:01 EDT** — do NOT resend.

## Gotchas

- Edit the forks in `src/modules/` in place; keep `reference-tps/` pristine for diffing.
- Autoformatter (PostToolUse hook) reorders imports — don't depend on import order for path setup;
  tests put `src/` on path at top via explicit sys.path.insert.
- This session's dead-end: a parallel `src/mobileportrait/` subclass package + root `tests/`,
  `notes/` were created then removed — the in-place `src/modules/` fork is canonical.

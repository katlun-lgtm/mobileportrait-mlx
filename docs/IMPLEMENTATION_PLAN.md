# MobilePortrait — Stage A implementation plan (fork TPS + 4 deltas)

Path B from `project_lp_arch_research`. Fork `Thin-Plate-Spline-Motion-Model` (MIT,
in `reference-tps/`) → add 4 deltas → retrain → MLX port (Stage C reuses lp-mlx infra).
Target: **≥20 fps on M3 Max** (projected ~100 fps for 16G variant; 2D U-Nets only, no 3D
conv — the op profile that walled lp-mlx at 6 fps is absent here).

## Why this beats lp-mlx
LivePortrait = 3D-conv Hourglass + 50-layer SPADE = conv/bandwidth wall → 6 fps.
MobilePortrait runtime = **just 2 plain 2D U-Nets + one grid_sample/frame**. Everything
else (keypoints, BG inpaint, multiview feats) is precomputed once per source. Reuses
lp-mlx's 2D grid_sample Metal kernel directly.

## TPS code map (what we're modifying)
| File | Class | Role | Touch for |
|---|---|---|---|
| `modules/keypoint_detector.py` | `KPDetector(num_tps)` → `fg_kp` (num_tps*5=50 NK) | neural KP | **Δ1** |
| `modules/dense_motion.py` | `DenseMotionNetwork` (Hourglass→`maps`(num_tps+1)+4 occlusion) | flow+occlusion | **Δ2, Δ3** |
| `modules/inpainting_network.py` | `InpaintingNetwork` (synthesis U-Net, occlude_input+enc/dec) | render | **Δ4** |
| `modules/model.py` | `GeneratorFullModel` (kp+dm+inpaint, VGG/eq losses) | training+losses | **Δ1 wiring, Δ3 losses** |
| `train.py`, `frames_dataset.py` | loop + data | training | masks/inpaint data |

## The 4 deltas (concrete)

### Δ1 — Mixed keypoints (106 FK + 50 NK → 50 mixed)
- Add `modules/fk_detector.py`: wrap an off-the-shelf **106-landmark detector** (insightface
  2d106 / mediapipe), FROZEN, returns (bs,106,2) normalized to [-1,1].
- Add `MixedKP` MLP: concat(FK 106·2=212, NK fg_kp 50·2=100) = 312 → MLP → 100 → reshape (50,2).
- In `KPDetector.forward` (or a wrapper used by `GeneratorFullModel`): compute NK as today,
  run FK, fuse → set `out['fg_kp'] = mixed_kp`. DMN/TPS consume `fg_kp` unchanged downstream.
- num_tps stays 10 (10×5=50 mixed). Validate TPS K-transform still builds from 50 kp.

### Δ2 — Residual optical flow (+2 channels)
- In `DenseMotionNetwork.__init__`: add `self.residual = nn.Conv2d(hourglass_out[-1], 2, 7, padding=3)`.
- In `forward`: `residual_flow = self.residual(prediction)`; add to the composed optical flow
  `deformation` before warp. (Ablation: AED 3.0 with vs 6.2 without.)

### Δ3 — DMN facial-knowledge mask heads (training-only) + L1 losses
- In `DenseMotionNetwork.__init__`: add `self.fg_mask_head` + `self.lmk_mask_head`
  (`nn.Conv2d(hourglass_out[-1], 1, 7, padding=3)` each). Output in dict only when `self.training`.
- In `GeneratorFullModel.forward`: add `L_fgmask = L1(pred_fg, gt_driving_fg_mask)` and
  `L_lmk = L1(pred_lmk, gt_driving_landmark_mask)` to the loss sum. GT masks from dataset
  (Mask2Former fg seg + rendered 106-landmark heatmap of driving), precomputed.
- Remove heads at inference (guard by `self.training`).

### Δ4 — Synthesis: pseudo-multiview + pseudo-BG
- **Pseudo-BG**: precompute LaMa inpaint of source (fg removed) → extra input channels to
  `InpaintingNetwork` (concat source + bg_inpaint + fg_mask). **Also inpaint the DRIVING image
  in training** (paper: "crucial"). Add to `frames_dataset.py`.
- **Pseudo-multiview (T=4)**: offline, warp source to 4 uniformly-sampled driving frames, run
  through synthesis enc to the **lowest-res downblock**, cache feats. Add an **extra merge conv**
  at that downblock fusing cached multiview feats with current feats. Runtime: cache is per-source
  (precomputed), only the merge conv runs per frame. (Ablation: 4 views FID 29.2 vs 0 views 34.2.)

## Underspecified → inherit/sweep (from spec)
- U-Net channel/layer configs per variant → inherit TPS DMN config for 16G; sweep for 7G/4G.
- NK detector arch + #TPS transforms K=10 → inherit TPS defaults.
- Mixed-KP MLP hidden dims → small sweep (e.g. 312→256→256→100).
- Heatmap res / occlusion count (4) → inherit TPS.
- Audio path → OUT OF SCOPE v1 (video-driven first).

## Environment reality
- **Code dev**: here on dev (Linux, CPU) — write the fork + deltas, unit-test shapes on CPU.
- **Validation + training**: needs CUDA. Smoke-validate on a rented 4090 (CelebvHQ subset);
  full train on 8×A100 (Stage B, ~$1.5–2.5k). Dev box has no GPU.
- **Stage C (MLX port)**: MBP, reuse all lp-mlx infra.

## Stage A milestones
1. Repo scaffold + TPS reference (DONE).
2. Δ1 mixed-KP (code + CPU shape test) — biggest arch change.
3. Δ2 residual flow (small).
4. Δ3 mask heads + losses (+ dataset mask precompute).
5. Δ4 synthesis multiview + pseudo-BG (+ LaMa precompute) — largest.
6. Wire `GeneratorFullModel`, CPU forward/backward smoke (tiny batch).
7. Rent 4090 → overfit a tiny CelebvHQ subset → confirm it learns + beats NK-only qualitatively.
   That go/no-go gates Stage B (full training).

## Acceptance (Stage A)
Forward+backward runs; loss decreases on a tiny overfit set; output qualitatively tracks
driving better than vanilla TPS NK-only. Then → Stage B training.

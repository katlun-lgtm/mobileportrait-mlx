# MobilePortrait — Architecture Spec & Port Plan (Path B Phase 2)

Extracted from arXiv 2407.05712v2 (CVPR 2025, ByteDance). Date: 2026-05-29.
Companion to memory `project_lp_arch_research`. Predecessor infra: `project_lp_mlx_port` (lp-mlx).

## TL;DR — the big finding

**The architecture is exactly the op profile Apple Silicon is GOOD at.** No 3D conv, no
multiscale feature warping, no attention, no dynamic conv. Just **two 2D U-Nets** + **one
warp (grid_sample) per frame** + a TPS transform. This is the opposite of LivePortrait's
conv-bound/3D-conv wall. Reuses lp-mlx's already-built 2D grid_sample Metal kernel directly.

**Second finding (changes Path B economics):** MobilePortrait is explicitly **"follow TPS
[38] to generate the initial transformations"** and its warp/occlusion/dense-motion pipeline
is "similar to [38,9,20,21]". TPS = `yoyo-nb/Thin-Plate-Spline-Motion-Model` (MIT, 3.5k★).
So Path B is **fork TPS + bolt on 4 deltas + retrain**, NOT build-from-scratch. The DMN
U-Net, dense motion, warp, occlusion, equivariance loss, training loop already exist in TPS.

## Runtime cost = 2 U-Nets only

Everything else is **precomputed once per source image** (zero per-frame cost):
- NK detector, FK detector, mixed-KP MLP, foreground segmenter, BG inpaint, pseudo-multiview feats.

Per driving frame, only:
1. Mixed keypoints for D (FK detect + NK detect + MLP) — cheap
2. TPS initial transforms from S/D mixed KP
3. **Dense Motion U-Net** → optical flow M + occlusion maps + residual flow
4. warp(S, M) × occlusion → S_w
5. **Synthesis U-Net**(S_w + pseudo-multiview + pseudo-BG + fg mask) → output frame

## Keypoints

| Type | Count | Source |
|---|---|---|
| Facial (FK) | 106 | pretrained 106-landmark detector (off-the-shelf; insightface 106 or similar) |
| Neural (NK) | 50 | learned FOMM/TPS-style detector |
| **Mixed** | **50** | MLP: concat(FK,NK) → MLP → 50 keypoints |

- Mixed KP replace the neural KP used by TPS/FOMM in the flow computation.
- TPS initial transformations built from the 50 mixed KP (follow TPS K-transform scheme).
- Keypoints rendered as heatmaps → fed to DMN.

## Dense Motion Network (DMN) — U-Net

- Inputs: KP heatmaps + initial TPS transforms + **fg mask + facial-landmark mask of source**
  (masks precomputed once).
- Outputs: optical flow M, occlusion maps, **+2 extra channels at last layer = residual
  optical flow** (MetaPortrait/ResNet idea; ablation: residual ON gives AED(C) 3.0 vs 6.2 OFF).
- Training-only heads: predict driving fg mask + driving landmark mask (L1 "facial knowledge
  loss"). Removed at inference.

## Synthesis Network — U-Net

- Input: warped source S_w.
- **Pseudo multiview foreground**: take T=4 uniformly-sampled driving frames, warp source to
  each (offline), run them through the synthesis U-Net **early downblocks up to the last/lowest-
  res downblock**, cache those features. At runtime an **extra conv** merges cached multiview
  feats with the current frame's feats at that lowest-res downblock. (Ablation: 4 views = sweet
  spot, FID 29.2; 0 views FID 34.2; 8 views barely better FID but best AKD.)
- **Pseudo background**: LaMa inpaint of source after fg removal, + fg mask, as extra synth
  inputs. Must also inpaint the DRIVING image during training (paper: "crucial").
- Best config = feed pseudo-BG + end-to-end synth (NOT separate fg/bg alpha composite).

## Variants (channel/layer scaling)

| FLOPs | Params | iPhone14 Pro | iPhone12 | Projected M3 Max |
|---|---:|---:|---:|---:|
| 16G | 67.7M | 15.8 ms (63fps) | 25.5 ms | ~10 ms (~100fps) |
| 7G  | 40.8M | 6.4 ms | 10.9 ms | ~4 ms |
| 4G  | 25.5M | 5.9 ms | 8.9 ms | ~4 ms |

Quality vs LP baseline (Table 1): FID 29.2 (TPS 29.8, MCNet 27.2), AKD **1.30 best**, 16 GFLOP
vs 131–629 for others. Good enough; not SOTA on every metric but ~10× cheaper.

## Training recipe

- Data: VFHQ (16,827 clips) + CelebvHQ (35,666 @512²) + VoxCeleb2 (150,480 @256²) ≈ 21k IDs.
- 512px, 25 FPS, square face-crop.
- 8× A100, lr 0.002, 60 epochs.
- Loss: `L = L_percep + L_L1 + L_kp + L_eq + L_landmark + L_mask` (all standard; last two are
  the L1 facial-knowledge mask losses; L_eq = equivariance from TPS; L_kp = facial KP distance).
- Off-the-shelf deps: Mask2Former (fg seg, ref [2]), LaMa (inpaint, ref [25]), 106-pt FK detector.

## Underspecified — the porting/training risks (must reverse-engineer or sweep)

1. **Exact U-Net channel/layer configs per variant** — paper only says "reduce channels and
   layers." → inherit TPS DMN config for 16G; sweep down for 7G/4G.
2. **NK detector arch + #TPS transforms K** — TPS default K=10, 50 KP (10×5). MobilePortrait
   keeps 50 mixed KP → likely same K. Inherit from TPS.
3. **Mixed-KP MLP dims** — concat(106×2 + 50×2)=312 → MLP → 100 (50 xy). Small; sweep.
4. **Heatmap resolution / occlusion map count** — inherit TPS.
5. **Audio path (3DMM→mesh→NK via ResNet18 + LSTM audio→3DMM)** — out of scope for v1
   (video-driven first).

## Recommended plan (revised, cheaper than the original from-scratch estimate)

### Stage A — PyTorch reference on top of TPS (2–3 wk, $0, local/dev)
Fork `yoyo-nb/Thin-Plate-Spline-Motion-Model`. It already gives NK detector, dense motion
U-Net, warp, occlusion, equivariance + perceptual losses, training loop. Add the 4 deltas:
1. 106-pt FK detector + mixed-KP MLP (replace NK feed with mixed KP).
2. +2 residual-flow channels on DMN last layer.
3. DMN training-only fg/landmark mask heads + L1 losses.
4. Synthesis: pseudo-multiview merge conv at lowest downblock + pseudo-BG (LaMa) inputs;
   inpaint driving in training.
Validate on a tiny subset (CelebvHQ only) that it trains + beats NK-only qualitatively.

### Stage B — Train (1–2 wk, ~$1.5–2.5k, Vast.ai 8×A100)
Pre-stage datasets to storagebox (VFHQ on HF `KwaiVGI/VFHQ`; VoxCeleb2 Oxford VGG;
CelebvHQ). Start with CelebvHQ-only run to de-risk, then full. 60 epochs lr 0.002.

### Stage C — MLX port + integrate (1–2 wk, $0, MBP)
Reuse ALL lp-mlx infra: NHWC conv loaders, weight pth→npz converter, 2D grid_sample Metal
kernel, FastAPI WS serve.py + webcam UI, lp_cropper landmark crop. Port 2 U-Nets + TPS warp.
Parity vs PyTorch within 1e-3. Bench → expect well under 20 ms/frame (16G) on M3 Max.

### Path A (parallel, ~20%): email author for code/weights
Draft at `/tmp/mobileportrait_email_draft.md`. **Fix before send:** says "Meta's product
context" — paper is **ByteDance** (jianwen.alan@gmail.com, lingaojie@bytedance.com). Jiang may
now be at Meta but the work + weights are ByteDance's. Reframe as ByteDance. Gated on user
confirm (external email, Tier 3).

## Why this is the right bet vs lp-mlx

lp-mlx faithful port hit ~6 fps fp32 on M3 Max, conv/3D-conv-bound — architecture is wrong
for Apple Silicon. MobilePortrait removes every op Apple Silicon is bad at and is 10× cheaper
in FLOPs. Even the 16G full variant projects to ~100 fps on M3 Max. Real win is the
architecture swap, not more MLX optimization.

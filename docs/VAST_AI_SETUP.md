# 3090 rent-and-measure setup (vast.ai)

Goal: on a rented 3090, **measure** real sec/step at BS28, then run the **full
MobilePortrait recipe** (all losses, BS28, real CUDA grid_sample) with a held-out eval
render — the regime the Mac MLX port structurally can't reach. Answers two questions:

1. Real $/run (spec estimate was 1.5–2.5 s/step @ BS28; verify before the long run).
2. Does BS28 + more steps + the full loss set (incl. landmark_mask/fg_mask that were
   inert on the Mac) beat the Mac numbers? **Mac MLX reference: L1-only BS2 baseline
   render_L1 0.06329; eff-batch-8 warmup best 0.05857; full-recipe (eq+warp+kp) ~0.075
   (worse, 4 runs).** This is the experiment that could flip that.

---

## 0. Pick the instance
- **GPU:** 1× RTX 3090 (24 GB). On vast.ai filter `RTX 3090`, ≥ 24 GB, ≥ 100 Mbps down.
- **Disk:** ≥ 60 GB (dataset + checkpoints).
- **Image:** `pytorch/pytorch:2.x-cuda12.x-cudnn9-runtime` (or any CUDA 12 + torch image).
- Budget ~$0.20–0.35/hr. The **measure** step (100 steps) costs cents; decide the long
  run length from the printed sec/step.

## 1. Get the code onto the box
The repo is dev-local (no git remote). From the **dev server** push the needed dirs
(exclude data/venv/renders — large and box-specific):

```bash
# on dev (204.168.159.197), replace HOST:PORT with the vast.ai ssh target
rsync -avz -e "ssh -p <PORT>" \
  --exclude '.git' --exclude 'data' --exclude 'renders' --exclude '*.pth.tar' \
  /root/mobileportrait-mlx/src \
  /root/mobileportrait-mlx/reference-tps \
  /root/mobileportrait-mlx/configs \
  /root/mobileportrait-mlx/docs \
  root@<HOST>:/workspace/mobileportrait-mlx/
```

(If rsync is awkward, `tar czf mp.tgz src reference-tps configs docs && scp` then untar.)

## 2. Python deps
The pinned `reference-tps/requirements.txt` is for torch 1.10/cu113 — **do not** use it
verbatim on a CUDA 12 box. Install current wheels instead:

```bash
cd /workspace/mobileportrait-mlx
pip install numpy scikit-image scikit-learn imageio imageio-ffmpeg pyyaml pillow tqdm
pip install insightface onnxruntime          # FK detector (buffalo_l auto-downloads ~280MB)
# torch/torchvision already in the image; verify CUDA:
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. Warm-start checkpoint (TPS vox.pth.tar)
Needed for `--tps-checkpoint`. Either scp it from the Mac/dev, or pull the public TPS
release. Put it at `checkpoints/vox.pth.tar`.

```bash
mkdir -p checkpoints
# scp from Mac: scp katlun@173.32.242.72:~/mobileportrait-mlx/checkpoints/vox.pth.tar checkpoints/
```

## 4. Dataset
The config `dataset_params.root_dir` must point at a `train/` + `test/` tree of clips
(FramesDataset reads folders-of-frames or short videos).

- **Cheap/fast:** scp the same CelebV-HQ frame subset used on the Mac
  (`~/mobileportrait-mlx/data/celebvhq_frames`, ~49k frames) and set `root_dir` to it.
  Good enough for the *measure* run and a first BS28 trend.
- **Paper-scale:** stage full VoxCeleb (large; only if going for the real 100-epoch test).
- Edit `configs/mac-celebvhq-256.yaml` (or vox-256.yaml) `root_dir` to the box path.
  Keep `batch_size: 28` for the real experiment (vox-256.yaml already has 28).
  Set `dataloader_workers` to ~8 on a 3090 box.

## 5. MEASURE first (cents, ~4 min) — DON'T skip
```bash
python src/mp_train_eval.py \
  --config configs/mac-celebvhq-256.yaml \
  --tps-checkpoint checkpoints/vox.pth.tar \
  --fk-backend insightface --providers stub \
  --max-steps 100 --eval-every 0 --log-dir log/measure
```
Read the printed `samples/s` and `s/step`. Compute the long-run cost:
`hours = max_steps * s/step / 3600`, `cost = hours * $/hr`.
**Re-decide max_steps from the REAL number** before launching the long run. If s/step is
way off the 1.5–2.5 estimate (slow disk / dataloader bound), fix workers/data staging first.

## 6. The experiment (~30–50k steps; size it from step 5)
```bash
nohup python src/mp_train_eval.py \
  --config configs/mac-celebvhq-256.yaml \
  --tps-checkpoint checkpoints/vox.pth.tar \
  --fk-backend insightface --providers stub \
  --max-steps 40000 --eval-every 1000 --log-dir log/mp \
  > log/run.out 2>&1 &
tail -f log/run.out
```
- `--providers real` adds rembg fg-seg + LaMa pseudo-BG (needs `pip install rembg`); only
  if you want the Δ3 mask / Δ4 BG targets to be *real* rather than stub. Start with stub.
- Watch `EVAL stepN render_L1=...`. **Compare to 0.05857 (Mac eff-batch-8 warmup best).**
  - clearly < 0.0586 → BS28 + full losses HELP (the result the Mac couldn't get).
  - ≈ 0.058–0.063 → batch helped, losses neutral.
  - > 0.063 → losses still hurt even at BS28 → definitive negative, stop.
- `best.pth.tar` holds the best-eval checkpoint. scp it back to keep.

## 7. Pull results back + STOP the instance
```bash
# from dev: scp the eval renders + best checkpoint back, then DESTROY the vast.ai instance
scp -P <PORT> root@<HOST>:/workspace/mobileportrait-mlx/log/mp/eval_*.png ./log_3090/
scp -P <PORT> root@<HOST>:/workspace/mobileportrait-mlx/log/mp/best.pth.tar ./log_3090/
```
**Destroy the instance** as soon as results are pulled — vast.ai bills while it exists.

---

## Cost ready-reckoner (verify s/step from step 5 first!)
| steps | @1.5s/step | @2.0s | @2.5s |  @ $0.30/hr |
|------|-----------|------|------|------------|
| 100 (measure) | 2.5 min | 3.3 | 4.2 | ~$0.02 |
| 40,000 | 16.7 h | 22 | 28 | **~$5–8** |
| 176,000 (100ep on 49k frames) | 73 h | 98 | 122 | ~$22–37 |

Recommended: **measure → 40k-step experiment (~$5–8)**. Only go to 100-epoch if 40k shows
the losses starting to help.

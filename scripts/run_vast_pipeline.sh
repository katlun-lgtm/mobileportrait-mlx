#!/bin/bash
# Full data-prep + training pipeline for vast.ai after videos.tar has been transferred.
# Run: bash /workspace/mobileportrait-mlx/scripts/run_vast_pipeline.sh 2>&1 | tee /workspace/pipeline.log
set -e

WORKSPACE=/workspace/mobileportrait-mlx
DATA=$WORKSPACE/data
FRAMES=$DATA/celebvhq_frames
TAR=$DATA/videos.tar

echo "=== STEP 1a: extract train frames from local tar ==="
python $WORKSPACE/scripts/extract_frames.py \
  --local-tar $TAR \
  --clip-list /workspace/clip_list_train.txt \
  --out-dir $FRAMES/train \
  --workers 8

echo "=== STEP 1b: extract test frames from local tar ==="
python $WORKSPACE/scripts/extract_frames.py \
  --local-tar $TAR \
  --clip-list /workspace/clip_list_test.txt \
  --out-dir $FRAMES/test \
  --workers 8

echo "=== STEP 2: extract FK keypoints (GPU) ==="
python $WORKSPACE/scripts/extract_fk.py

echo "=== STEP 3: launch training ==="
mkdir -p $WORKSPACE/log/mp3k
nohup python $WORKSPACE/src/mp_train_eval.py \
  --config $WORKSPACE/configs/vast-celebvhq-3090.yaml \
  --mp-checkpoint $WORKSPACE/checkpoints/mp40k-best.pth.tar \
  --fk-backend insightface \
  --workers 4 \
  --max-steps 40000 \
  --eval-every 2000 \
  --log-dir $WORKSPACE/log/mp3k \
  > $WORKSPACE/log/mp3k/train.log 2>&1 &
echo "Training PID=$!"
echo "tail -f $WORKSPACE/log/mp3k/train.log to monitor"

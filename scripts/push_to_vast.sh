#!/usr/bin/env bash
# Push the code (not data/venv/checkpoints) to a rented vast.ai box.
# Usage: ./scripts/push_to_vast.sh <HOST> <PORT> [REMOTE_DIR]
#   ./scripts/push_to_vast.sh ssh5.vast.ai 12345
# Then on the box, follow docs/VAST_AI_SETUP.md from step 2 (pip).
set -euo pipefail

HOST="${1:?usage: push_to_vast.sh HOST PORT [REMOTE_DIR]}"
PORT="${2:?usage: push_to_vast.sh HOST PORT [REMOTE_DIR]}"
REMOTE="${3:-/workspace/mobileportrait-mlx}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

echo "Pushing $HERE -> root@$HOST:$REMOTE (port $PORT)"
ssh -p "$PORT" "root@$HOST" "mkdir -p $REMOTE"
rsync -avz -e "ssh -p $PORT" \
  --exclude '.git' --exclude 'data' --exclude 'renders' \
  --exclude '*.pth.tar' --exclude '*.safetensors' --exclude '__pycache__' \
  --exclude '.venv' --exclude 'log' \
  "$HERE/src" "$HERE/reference-tps" "$HERE/configs" "$HERE/docs" \
  "root@$HOST:$REMOTE/"
echo "Done. Next on the box: cd $REMOTE && see docs/VAST_AI_SETUP.md step 2."

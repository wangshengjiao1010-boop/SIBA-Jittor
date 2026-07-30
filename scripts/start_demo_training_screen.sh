#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_paths.sh"

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/SIBA-Jittor}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/datasets/SIBA}
DEMO_TAG=${DEMO_TAG:-$(date +%Y%m%d_%H%M%S)}
DEMO_DIR="$PROJECT_ROOT/logs/demo_$DEMO_TAG"
INITIAL_CHECKPOINT=${INITIAL_CHECKPOINT:-$PROJECT_ROOT/logs/demo_shared_initial/SIBA_seed2025_initial.pth}

mkdir -p "$DEMO_DIR"
if [[ ! -f "$INITIAL_CHECKPOINT" ]]; then
  mkdir -p "$(dirname "$INITIAL_CHECKPOINT")"
  "$PYTORCH_PYTHON" \
    "$PROJECT_ROOT/tools/export_pytorch_initial_weights.py" \
    --project-root "$PROJECT_ROOT" \
    --seed 2025 \
    --output "$INITIAL_CHECKPOINT"
fi
screen -S kk -X quit >/dev/null 2>&1 || true
screen -L -Logfile "$DEMO_DIR/screen.log" -dmS kk bash -lc "
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONUNBUFFERED=1
  export JITTOR_HOME=/root/autodl-tmp/.cache/jittor_gpu
  cd '$PROJECT_ROOT'
  '$JITTOR_PYTHON' -u tools/demo_train_step.py \
    --ir-path '$DATA_ROOT/train/ir' \
    --vi-path '$DATA_ROOT/train/vi' \
    --steps 20 \
    --batch-size 4 \
    --patch-size 128 \
    --seed 2025 \
    --checkpoint '$INITIAL_CHECKPOINT' \
    --save-checkpoint '$DEMO_DIR/SIBA_demo_step20.pkl' \
    --output '$DEMO_DIR/demo_training.json' \
    2>&1 | tee '$DEMO_DIR/train.log'
"

echo "$DEMO_DIR" > "$PROJECT_ROOT/logs/latest_demo.txt"
screen -ls
echo "Attach with: screen -r kk"
echo "Log: $DEMO_DIR/train.log"

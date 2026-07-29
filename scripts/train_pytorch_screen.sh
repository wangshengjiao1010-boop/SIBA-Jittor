#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/SIBA-Jittor}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/datasets/SIBA}
RUN_NAME=${RUN_NAME:-pytorch_msrs_roadscene_60e}
LOG_DIR="$PROJECT_ROOT/logs/$RUN_NAME"
mkdir -p "$LOG_DIR" "$PROJECT_ROOT/checkpoints/$RUN_NAME"

screen -S kk -X quit >/dev/null 2>&1 || true
screen -L -Logfile "$LOG_DIR/screen.log" -dmS kk bash -lc "
  set -euo pipefail
  export PYTHONUNBUFFERED=1
  cd '$PROJECT_ROOT'
  /root/autodl-tmp/envs/siba_torch/bin/python -u tools/run_training.py \\
    --framework pytorch \\
    --ir-path '$DATA_ROOT/train/ir' \\
    --vi-path '$DATA_ROOT/train/vi' \\
    --output '$PROJECT_ROOT/checkpoints/$RUN_NAME' \\
    --epochs 60 --gpu-number 0 --seed 2025 \\
    2>&1 | tee '$LOG_DIR/train.log'
"
screen -ls

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_paths.sh"

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/SIBA-Jittor}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/datasets/SIBA/test}
RUN_TAG=${RUN_TAG:-20260727_siba_official_protocol}
JITTOR_CHECKPOINT=${JITTOR_CHECKPOINT:-$PROJECT_ROOT/checkpoints/jittor_msrs_roadscene_60e_$RUN_TAG/07-27-04-52/SIBA_epoch60.pkl}
PYTORCH_CHECKPOINT=${PYTORCH_CHECKPOINT:-$PROJECT_ROOT/checkpoints/pytorch_msrs_roadscene_60e_$RUN_TAG/07-27-06-00/SIBA_epoch60.pth}
LOG_DIR="$PROJECT_ROOT/logs/full_inference_$RUN_TAG"
RESULT_ROOT="$PROJECT_ROOT/results/full_$RUN_TAG"

mkdir -p "$LOG_DIR" "$RESULT_ROOT"
screen -S kk -X quit >/dev/null 2>&1 || true
screen -L -Logfile "$LOG_DIR/screen.log" -dmS kk bash -lc "
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4
  export JITTOR_HOME=/root/autodl-tmp/.cache/jittor_gpu
  cd '$PROJECT_ROOT'
  echo 'start_time='\"\$(date --iso-8601=seconds)\" | tee '$LOG_DIR/status.log'

  for dataset in MSRS M3FD_2x TNO; do
    output='$RESULT_ROOT/jittor/'\"\$dataset\"
    rm -rf \"\$output\"
    mkdir -p \"\$output\"
    echo '=== Jittor inference:' \"\$dataset\" '===' | tee -a '$LOG_DIR/status.log'
    '$JITTOR_PYTHON' -u tools/run_inference.py \\
      --framework jittor \\
      --checkpoint '$JITTOR_CHECKPOINT' \\
      --data-dir '$DATA_ROOT/'\"\$dataset\" \\
      --output \"\$output\" \\
      --use-cuda --warmup-runs 10 \\
      2>&1 | tee '$LOG_DIR/jittor_'\"\$dataset\"'.log'
  done

  for dataset in MSRS M3FD_2x TNO; do
    output='$RESULT_ROOT/pytorch/'\"\$dataset\"
    rm -rf \"\$output\"
    mkdir -p \"\$output\"
    echo '=== PyTorch inference:' \"\$dataset\" '===' | tee -a '$LOG_DIR/status.log'
    '$PYTORCH_PYTHON' -u tools/run_inference.py \\
      --framework pytorch \\
      --checkpoint '$PYTORCH_CHECKPOINT' \\
      --data-dir '$DATA_ROOT/'\"\$dataset\" \\
      --output \"\$output\" \\
      --use-cuda --warmup-runs 10 \\
      2>&1 | tee '$LOG_DIR/pytorch_'\"\$dataset\"'.log'
  done

  echo 'end_time='\"\$(date --iso-8601=seconds)\" | tee -a '$LOG_DIR/status.log'
  touch '$LOG_DIR/INFERENCE_COMPLETE'
"

screen -ls
echo "Inference log: $LOG_DIR/screen.log"

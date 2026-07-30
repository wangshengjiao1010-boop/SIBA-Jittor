#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_paths.sh"

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/SIBA-Jittor}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/datasets/SIBA/test}
RUN_TAG=${RUN_TAG:-20260727_siba_official_protocol}
PYTORCH_CHECKPOINT=${PYTORCH_CHECKPOINT:-$PROJECT_ROOT/checkpoints/pytorch_msrs_roadscene_60e_$RUN_TAG/07-27-06-00/SIBA_epoch60.pth}
OFFICIAL_CHECKPOINT=${OFFICIAL_CHECKPOINT:-$PROJECT_ROOT/official_pytorch/checkpoint/SIBA_epoch60.pth}
SELF_RESULT_ROOT="$PROJECT_ROOT/results/full_$RUN_TAG"
OFFICIAL_RESULT_ROOT="$PROJECT_ROOT/results/official_checkpoint_alignment_$RUN_TAG"
LOG_DIR="$PROJECT_ROOT/logs/remaining_gpu_$RUN_TAG"

mkdir -p "$LOG_DIR" "$SELF_RESULT_ROOT/pytorch" "$OFFICIAL_RESULT_ROOT"
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
  nvidia-smi --query-gpu=timestamp,name,driver_version,memory.total --format=csv,noheader | tee '$LOG_DIR/gpu_start.csv'

  (while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits || true
    sleep 1
  done) > '$LOG_DIR/gpu_monitor.csv' 2>&1 &
  monitor_pid=\$!
  trap 'kill \$monitor_pid >/dev/null 2>&1 || true' EXIT

  for dataset in M3FD_2x TNO; do
    output='$SELF_RESULT_ROOT/pytorch/'\"\$dataset\"
    rm -rf \"\$output\"
    mkdir -p \"\$output\"
    echo '=== Self-trained PyTorch inference:' \"\$dataset\" '===' | tee -a '$LOG_DIR/status.log'
    '$PYTORCH_PYTHON' -u tools/run_inference.py \\
      --framework pytorch \\
      --checkpoint '$PYTORCH_CHECKPOINT' \\
      --data-dir '$DATA_ROOT/'\"\$dataset\" \\
      --output \"\$output\" \\
      --use-cuda --warmup-runs 10 \\
      2>&1 | tee '$LOG_DIR/self_pytorch_'\"\$dataset\"'.log'
  done

  for framework in jittor pytorch; do
    if [ \"\$framework\" = jittor ]; then
      python_bin='$JITTOR_PYTHON'
    else
      python_bin='$PYTORCH_PYTHON'
    fi
    for dataset in MSRS M3FD_2x TNO; do
      output='$OFFICIAL_RESULT_ROOT/'\"\$framework/\$dataset\"
      rm -rf \"\$output\"
      mkdir -p \"\$output\"
      echo '=== Official-checkpoint' \"\$framework\" \"\$dataset\" '===' | tee -a '$LOG_DIR/status.log'
      \"\$python_bin\" -u tools/run_inference.py \\
        --framework \"\$framework\" \\
        --checkpoint '$OFFICIAL_CHECKPOINT' \\
        --data-dir '$DATA_ROOT/'\"\$dataset\" \\
        --output \"\$output\" \\
        --use-cuda --warmup-runs 10 \\
        2>&1 | tee '$LOG_DIR/official_'\"\$framework\"'_'\"\$dataset\"'.log'
    done
  done

  for framework in jittor pytorch; do
    if [ \"\$framework\" = jittor ]; then
      python_bin='$JITTOR_PYTHON'
    else
      python_bin='$PYTORCH_PYTHON'
    fi
    for dataset in MSRS M3FD_2x TNO; do
      output='$PROJECT_ROOT/results/official_timing_$RUN_TAG/'\"\$framework/\$dataset\"
      rm -rf \"\$output\"
      mkdir -p \"\$output\"
      echo '=== Official unsynchronized timing:' \"\$framework\" \"\$dataset\" '===' | tee -a '$LOG_DIR/status.log'
      \"\$python_bin\" -u tools/run_inference.py \\
        --framework \"\$framework\" \\
        --checkpoint '$OFFICIAL_CHECKPOINT' \\
        --data-dir '$DATA_ROOT/'\"\$dataset\" \\
        --output \"\$output\" \\
        --use-cuda --warmup-runs 0 --timing-mode official --skip-save \\
        2>&1 | tee '$LOG_DIR/official_timing_'\"\$framework\"'_'\"\$dataset\"'.log'
    done
  done

  for dataset in MSRS M3FD_2x TNO; do
    '$PYTORCH_PYTHON' tools/compare_fusion_outputs.py \\
      --reference '$OFFICIAL_RESULT_ROOT/pytorch/'\"\$dataset\" \\
      --candidate '$OFFICIAL_RESULT_ROOT/jittor/'\"\$dataset\" \\
      --output '$PROJECT_ROOT/results/output_alignment_$RUN_TAG/'\"\$dataset\" \\
      2>&1 | tee '$LOG_DIR/output_alignment_'\"\$dataset\"'.log'
  done

  kill \$monitor_pid >/dev/null 2>&1 || true
  trap - EXIT
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits | tee '$LOG_DIR/gpu_end.csv'
  echo 'end_time='\"\$(date --iso-8601=seconds)\" | tee -a '$LOG_DIR/status.log'
  touch '$LOG_DIR/REMAINING_GPU_COMPLETE'
"

screen -ls
echo "Remaining GPU log: $LOG_DIR/screen.log"

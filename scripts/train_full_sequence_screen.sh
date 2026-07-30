#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_paths.sh"

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/SIBA-Jittor}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/datasets/SIBA}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
SEQUENCE_DIR="$PROJECT_ROOT/logs/full_sequence_$RUN_TAG"
JITTOR_RUN="jittor_msrs_roadscene_60e_$RUN_TAG"
PYTORCH_RUN="pytorch_msrs_roadscene_60e_$RUN_TAG"

mkdir -p \
  "$SEQUENCE_DIR" \
  "$PROJECT_ROOT/logs/$JITTOR_RUN" \
  "$PROJECT_ROOT/logs/$PYTORCH_RUN" \
  "$PROJECT_ROOT/checkpoints/$JITTOR_RUN" \
  "$PROJECT_ROOT/checkpoints/$PYTORCH_RUN"

screen -S kk -X quit >/dev/null 2>&1 || true
screen -L -Logfile "$SEQUENCE_DIR/screen.log" -dmS kk bash -lc "
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4
  cd '$PROJECT_ROOT'

  echo 'run_tag=$RUN_TAG' | tee '$SEQUENCE_DIR/status.log'
  echo 'start_time='\"\$(date --iso-8601=seconds)\" | tee -a '$SEQUENCE_DIR/status.log'
  echo 'official_commit='\"\$(cat official_pytorch/OFFICIAL_COMMIT.txt)\" | tee -a '$SEQUENCE_DIR/status.log'
  echo 'training_pairs='\"\$(find '$DATA_ROOT/train/ir' -maxdepth 1 -type f | wc -l)\" | tee -a '$SEQUENCE_DIR/status.log'
  nvidia-smi | tee '$SEQUENCE_DIR/nvidia_smi_start.txt'

  echo '=== Jittor 60-epoch full training ===' | tee -a '$SEQUENCE_DIR/status.log'
  jittor_start=\$(date +%s)
  export JITTOR_HOME=/root/autodl-tmp/.cache/jittor_gpu
  '$JITTOR_PYTHON' -u tools/run_training.py \\
    --framework jittor \\
    --ir-path '$DATA_ROOT/train/ir' \\
    --vi-path '$DATA_ROOT/train/vi' \\
    --output '$PROJECT_ROOT/checkpoints/$JITTOR_RUN' \\
    --epochs 60 --gpu-number 0 --seed 2025 \\
    2>&1 | tee '$PROJECT_ROOT/logs/$JITTOR_RUN/train.log'
  jittor_end=\$(date +%s)
  echo 'jittor_duration_seconds='\"\$((jittor_end-jittor_start))\" | tee -a '$SEQUENCE_DIR/status.log'
  find '$PROJECT_ROOT/checkpoints/$JITTOR_RUN' -type f -printf '%p %s bytes\\n' | tee '$SEQUENCE_DIR/jittor_checkpoints.txt'

  echo '=== PyTorch 60-epoch full training ===' | tee -a '$SEQUENCE_DIR/status.log'
  pytorch_start=\$(date +%s)
  '$PYTORCH_PYTHON' -u tools/run_training.py \\
    --framework pytorch \\
    --ir-path '$DATA_ROOT/train/ir' \\
    --vi-path '$DATA_ROOT/train/vi' \\
    --output '$PROJECT_ROOT/checkpoints/$PYTORCH_RUN' \\
    --epochs 60 --gpu-number 0 --seed 2025 \\
    2>&1 | tee '$PROJECT_ROOT/logs/$PYTORCH_RUN/train.log'
  pytorch_end=\$(date +%s)
  echo 'pytorch_duration_seconds='\"\$((pytorch_end-pytorch_start))\" | tee -a '$SEQUENCE_DIR/status.log'
  find '$PROJECT_ROOT/checkpoints/$PYTORCH_RUN' -type f -printf '%p %s bytes\\n' | tee '$SEQUENCE_DIR/pytorch_checkpoints.txt'

  echo 'end_time='\"\$(date --iso-8601=seconds)\" | tee -a '$SEQUENCE_DIR/status.log'
  nvidia-smi | tee '$SEQUENCE_DIR/nvidia_smi_end.txt'
  touch '$SEQUENCE_DIR/TRAINING_COMPLETE'
"

echo "$RUN_TAG" > "$PROJECT_ROOT/logs/latest_full_sequence.txt"
screen -ls
echo "Sequence log: $SEQUENCE_DIR/screen.log"

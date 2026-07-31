#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/SIBA-Jittor}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/datasets/SIBA}
PYTORCH_ROOT=${PYTORCH_ROOT:-$PROJECT_ROOT/official_pytorch}
JITTOR_PYTHON=${JITTOR_PYTHON:-/root/miniconda3/envs/JittorDome/bin/python}
PYTORCH_PYTHON=${PYTORCH_PYTHON:-/root/miniconda3/envs/PytorchDome/bin/python}
RUN_ROOT=${RUN_ROOT:-$PROJECT_ROOT/logs/shared_seed2025}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints/shared_seed2025}
RESULT_ROOT=${RESULT_ROOT:-$PROJECT_ROOT/results/shared_seed2025}
SHARED_ROOT=${SHARED_ROOT:-$PROJECT_ROOT/shared}

run_inference() {
  local framework=$1
  local python=$2
  local checkpoint=$3
  local dataset=$4
  local output="$RESULT_ROOT/$framework/$dataset"
  local extra=()
  if [[ "$framework" == "pytorch" ]]; then
    extra=(--pytorch-root "$PYTORCH_ROOT")
  fi
  "$python" -u evaluation/run_inference.py \
    --framework "$framework" \
    --project-root "$PROJECT_ROOT" \
    "${extra[@]}" \
    --checkpoint "$checkpoint" \
    --data-dir "$DATA_ROOT/test/$dataset" \
    --output "$output" \
    --gpu-number 0 --use-cuda --timing-mode synchronized \
    2>&1 | tee "$RUN_ROOT/${framework}_${dataset}_inference.log"
}

worker() {
  mkdir -p \
    "$RUN_ROOT" \
    "$CHECKPOINT_ROOT/jittor" \
    "$CHECKPOINT_ROOT/pytorch" \
    "$RESULT_ROOT" \
    "$SHARED_ROOT" \
    "$PROJECT_ROOT/detele/shared_initial_validation"

  export CUDA_VISIBLE_DEVICES=0
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4
  export JITTOR_HOME=/root/autodl-tmp/.cache/jittor_gpu
  cd "$PROJECT_ROOT"

  if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Tracked working tree is not clean; refusing to start a formal run." >&2
    git status --short >&2
    exit 1
  fi
  git rev-parse HEAD | tee "$RUN_ROOT/code_revision.txt"
  git status --short | tee "$RUN_ROOT/code_status.txt"
  date --iso-8601=seconds | tee "$RUN_ROOT/started_at.txt"
  nvidia-smi | tee "$RUN_ROOT/nvidia_smi_start.txt"

  "$PYTORCH_PYTHON" tests/export_shared_initialization.py \
    --pytorch-root "$PYTORCH_ROOT" \
    --output-dir "$SHARED_ROOT" \
    --name initial --seed 2025 \
    2>&1 | tee "$RUN_ROOT/export_initial.log"

  "$PYTORCH_PYTHON" tests/generate_training_schedule.py \
    --ir-path "$DATA_ROOT/train/ir" \
    --vi-path "$DATA_ROOT/train/vi" \
    --output "$SHARED_ROOT/schedule.npz" \
    --metadata "$SHARED_ROOT/schedule.json" \
    --epochs 60 --patch-size 128 --seed 2025 \
    2>&1 | tee "$RUN_ROOT/generate_schedule.log"

  "$JITTOR_PYTHON" -u train.py \
    --ir-path "$DATA_ROOT/train/ir" \
    --vi-path "$DATA_ROOT/train/vi" \
    --output "$PROJECT_ROOT/detele/shared_initial_validation" \
    --run-name jittor --epochs 0 --gpu-number 0 --seed 2025 \
    --initial-weights "$SHARED_ROOT/initial.npz" \
    --schedule "$SHARED_ROOT/schedule.npz" \
    --log-csv "$RUN_ROOT/jittor_initial_batches.csv" \
    --metadata "$RUN_ROOT/jittor_initial_validation.json" \
    2>&1 | tee "$RUN_ROOT/jittor_initial_validation.log"

  "$PYTORCH_PYTHON" tests/verify_shared_training_inputs.py \
    --initial-metadata "$SHARED_ROOT/initial.json" \
    --schedule-metadata "$SHARED_ROOT/schedule.json" \
    --jittor-metadata "$RUN_ROOT/jittor_initial_validation.json" \
    --epochs 0 --training-pairs 1283 \
    --output "$RUN_ROOT/initial_inputs_verified.json" \
    2>&1 | tee "$RUN_ROOT/initial_inputs_verified.log"

  "$PYTORCH_PYTHON" -u tests/train_pytorch_reference.py \
    --pytorch-root "$PYTORCH_ROOT" \
    --ir-path "$DATA_ROOT/train/ir" \
    --vi-path "$DATA_ROOT/train/vi" \
    --initial-weights "$SHARED_ROOT/initial.npz" \
    --schedule "$SHARED_ROOT/schedule.npz" \
    --output "$CHECKPOINT_ROOT/pytorch/SIBA_epoch60.pth" \
    --log-csv "$RUN_ROOT/pytorch_batches.csv" \
    --metadata "$RUN_ROOT/pytorch_metadata.json" \
    --epochs 60 --batch-size 4 --patch-size 128 --seed 2025 --gpu-number 0 \
    2>&1 | tee "$RUN_ROOT/pytorch_train.log"

  "$JITTOR_PYTHON" -u train.py \
    --ir-path "$DATA_ROOT/train/ir" \
    --vi-path "$DATA_ROOT/train/vi" \
    --output "$CHECKPOINT_ROOT/jittor" \
    --run-name shared_seed2025 --epochs 60 --gpu-number 0 --seed 2025 \
    --initial-weights "$SHARED_ROOT/initial.npz" \
    --schedule "$SHARED_ROOT/schedule.npz" \
    --log-csv "$RUN_ROOT/jittor_batches.csv" \
    --metadata "$RUN_ROOT/jittor_metadata.json" \
    2>&1 | tee "$RUN_ROOT/jittor_train.log"

  "$PYTORCH_PYTHON" tests/verify_shared_training_inputs.py \
    --initial-metadata "$SHARED_ROOT/initial.json" \
    --schedule-metadata "$SHARED_ROOT/schedule.json" \
    --pytorch-metadata "$RUN_ROOT/pytorch_metadata.json" \
    --jittor-metadata "$RUN_ROOT/jittor_metadata.json" \
    --epochs 60 --training-pairs 1283 \
    --output "$RUN_ROOT/training_inputs_verified.json" \
    2>&1 | tee "$RUN_ROOT/training_inputs_verified.log"

  "$PYTORCH_PYTHON" evaluation/plot_training_components.py \
    --run "PyTorch=$RUN_ROOT/pytorch_batches.csv" \
    --run "Jittor=$RUN_ROOT/jittor_batches.csv" \
    --output "$RESULT_ROOT/loss_components.png" \
    --summary-csv "$RESULT_ROOT/epoch_loss_components.csv" \
    2>&1 | tee "$RUN_ROOT/plot_training_components.log"

  local pytorch_checkpoint="$CHECKPOINT_ROOT/pytorch/SIBA_epoch60.pth"
  local jittor_checkpoint="$CHECKPOINT_ROOT/jittor/shared_seed2025/SIBA_epoch60.pkl"
  for dataset in MSRS M3FD_2x TNO; do
    run_inference pytorch "$PYTORCH_PYTHON" "$pytorch_checkpoint" "$dataset"
    run_inference jittor "$JITTOR_PYTHON" "$jittor_checkpoint" "$dataset"
  done

  date --iso-8601=seconds | tee "$RUN_ROOT/completed_at.txt"
  nvidia-smi | tee "$RUN_ROOT/nvidia_smi_end.txt"
  touch "$RUN_ROOT/EXPERIMENT_COMPLETE"
}

if [[ "${1:-}" == "--worker" ]]; then
  worker
  exit 0
fi

mkdir -p "$RUN_ROOT"
screen -S kk -X quit >/dev/null 2>&1 || true
screen -L -Logfile "$RUN_ROOT/screen.log" -dmS kk \
  bash "$(readlink -f "$0")" --worker
echo "$RUN_ROOT"
screen -ls

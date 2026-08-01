#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
COMPARISON_CONFIG=${SIBA_COMPARISON_CONFIG:-$PROJECT_ROOT/configs/comparison.sh}
if [[ ! -f "$COMPARISON_CONFIG" ]]; then
  echo "Comparison config not found: $COMPARISON_CONFIG" >&2
  exit 2
fi
source "$COMPARISON_CONFIG"

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/datasets}
PYTORCH_ROOT=${PYTORCH_ROOT:-$PROJECT_ROOT/official_pytorch}
JITTOR_PYTHON=${JITTOR_PYTHON:-/root/miniconda3/envs/JittorDome/bin/python}
PYTORCH_PYTHON=${PYTORCH_PYTHON:-/root/miniconda3/envs/PytorchDome/bin/python}
RUN_ID=${RUN_ID:-shared_seed2025}
RUN_ROOT=${RUN_ROOT:-$PROJECT_ROOT/logs/$RUN_ID}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints/$RUN_ID}
RESULT_ROOT=${RESULT_ROOT:-$PROJECT_ROOT/results/$RUN_ID}
SHARED_ROOT=${SHARED_ROOT:-$PROJECT_ROOT/shared/$RUN_ID}
GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL:-1}

GPU_MONITOR_PID=""

stop_gpu_monitor() {
  if [[ -n "$GPU_MONITOR_PID" ]]; then
    kill "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
    wait "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
    GPU_MONITOR_PID=""
  fi
}

start_gpu_monitor() {
  local output=$1
  echo "timestamp,gpu_name,memory_used_mib,memory_total_mib,utilization_gpu_percent,power_draw_w" > "$output"
  (
    while true; do
      nvidia-smi \
        --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,power.draw \
        --format=csv,noheader,nounits >> "$output" 2>/dev/null || true
      sleep "$GPU_MONITOR_INTERVAL"
    done
  ) &
  GPU_MONITOR_PID=$!
}

finish_worker() {
  local status=$?
  trap - EXIT INT TERM
  stop_gpu_monitor
  date --iso-8601=seconds | tee "$RUN_ROOT/finished_at.txt"
  nvidia-smi | tee "$RUN_ROOT/nvidia_smi_end.txt" || true
  if [[ $status -eq 0 ]]; then
    touch "$RUN_ROOT/EXPERIMENT_COMPLETE"
  else
    printf '%s\n' "$status" > "$RUN_ROOT/EXPERIMENT_FAILED_EXIT_CODE.txt"
  fi
  exit "$status"
}

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

  if [[ -e "$RUN_ROOT/started_at.txt" || -e "$RUN_ROOT/EXPERIMENT_COMPLETE" ]]; then
    echo "Formal run directory already contains an experiment: $RUN_ROOT" >&2
    echo "Archive it before starting another run; existing evidence will not be overwritten." >&2
    exit 2
  fi

  export CUDA_VISIBLE_DEVICES=0
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4
  export JITTOR_HOME=/root/autodl-tmp/.cache/jittor_gpu
  cd "$PROJECT_ROOT"

  trap finish_worker EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Tracked working tree is not clean; refusing to start a formal run." >&2
    git status --short >&2
    exit 1
  fi
  git rev-parse HEAD | tee "$RUN_ROOT/code_revision.txt"
  git status --short | tee "$RUN_ROOT/code_status.txt"
  date --iso-8601=seconds | tee "$RUN_ROOT/started_at.txt"
  nvidia-smi | tee "$RUN_ROOT/nvidia_smi_start.txt"
  start_gpu_monitor "$RUN_ROOT/gpu_monitor.csv"

  "$PYTORCH_PYTHON" -m py_compile \
    tests/export_shared_initialization.py \
    tests/generate_training_schedule.py \
    tests/train_pytorch_reference.py \
    tests/verify_shared_training_inputs.py \
    tests/compare_fusion_outputs.py \
    evaluation/run_inference.py \
    evaluation/summarize_gpu_monitor.py
  "$JITTOR_PYTHON" -m py_compile \
    train.py \
    compat/pytorch_adam.py \
    compat/pytorch_clip.py \
    loader/train_loader.py

  "$PYTORCH_PYTHON" tests/validate_dataset_manifests.py \
    --dataset "train=$PROJECT_ROOT/data/manifests/combined_training_1283.json,$DATA_ROOT/train/ir,$DATA_ROOT/train/vi" \
    --dataset "MSRS=$PROJECT_ROOT/data/manifests/msrs_test.json,$DATA_ROOT/test/MSRS/ir,$DATA_ROOT/test/MSRS/vi" \
    --dataset "M3FD_2x=$PROJECT_ROOT/data/manifests/m3fd_2x_test.json,$DATA_ROOT/test/M3FD_2x/ir,$DATA_ROOT/test/M3FD_2x/vi" \
    --dataset "TNO=$PROJECT_ROOT/data/manifests/tno_test.json,$DATA_ROOT/test/TNO/ir,$DATA_ROOT/test/TNO/vi" \
    --output "$RUN_ROOT/dataset_integrity.json" \
    2>&1 | tee "$RUN_ROOT/dataset_integrity.log"

  "$PYTORCH_PYTHON" tests/export_pytorch_alignment.py \
    --pytorch-root "$PYTORCH_ROOT" \
    --checkpoint "$PYTORCH_ROOT/checkpoint/SIBA_epoch60.pth" \
    --output "$RUN_ROOT/pytorch_alignment_reference.npz" \
    --device cuda \
    2>&1 | tee "$RUN_ROOT/export_alignment_reference.log"

  "$JITTOR_PYTHON" tests/check_jittor_alignment.py \
    --project-root "$PROJECT_ROOT" \
    --checkpoint "$PYTORCH_ROOT/checkpoint/SIBA_epoch60.pth" \
    --reference "$RUN_ROOT/pytorch_alignment_reference.npz" \
    --output "$RUN_ROOT/jittor_alignment_report.json" \
    --use-cuda \
    2>&1 | tee "$RUN_ROOT/jittor_alignment.log"

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
  cp "$pytorch_checkpoint" "$PROJECT_ROOT/checkpoint/PyTorch_SIBA_shared_seed2025.pth"
  cp "$jittor_checkpoint" "$PROJECT_ROOT/checkpoint/SIBA_shared_seed2025.pkl"
  for dataset in MSRS M3FD_2x TNO; do
    run_inference pytorch "$PYTORCH_PYTHON" "$pytorch_checkpoint" "$dataset"
    run_inference jittor "$JITTOR_PYTHON" "$jittor_checkpoint" "$dataset"
    "$PYTORCH_PYTHON" tests/compare_fusion_outputs.py \
      --reference "$RESULT_ROOT/pytorch/$dataset" \
      --candidate "$RESULT_ROOT/jittor/$dataset" \
      --output "$RESULT_ROOT/alignment/$dataset" \
      2>&1 | tee "$RUN_ROOT/${dataset}_output_comparison.log"
  done

  stop_gpu_monitor
  "$PYTORCH_PYTHON" evaluation/summarize_gpu_monitor.py \
    --input "$RUN_ROOT/gpu_monitor.csv" \
    --output "$RUN_ROOT/gpu_summary.json" \
    2>&1 | tee "$RUN_ROOT/gpu_summary.log"

  date --iso-8601=seconds | tee "$RUN_ROOT/completed_at.txt"
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

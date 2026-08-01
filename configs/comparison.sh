#!/usr/bin/env bash

# Edit this file when running the controlled PyTorch/Jittor comparison.
CONFIG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$CONFIG_DIR/.." && pwd)}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/datasets}
PYTORCH_ROOT=${PYTORCH_ROOT:-$PROJECT_ROOT/official_pytorch}
JITTOR_PYTHON=${JITTOR_PYTHON:-/root/miniconda3/envs/JittorDome/bin/python}
PYTORCH_PYTHON=${PYTORCH_PYTHON:-/root/miniconda3/envs/PytorchDome/bin/python}

# Keep shared_seed2025 for the published run. Use a new name for a recording run.
RUN_ID=${RUN_ID:-shared_seed2025}
RUN_ROOT=${RUN_ROOT:-$PROJECT_ROOT/logs/$RUN_ID}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints/$RUN_ID}
RESULT_ROOT=${RESULT_ROOT:-$PROJECT_ROOT/results/$RUN_ID}
SHARED_ROOT=${SHARED_ROOT:-$PROJECT_ROOT/shared/$RUN_ID}
GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL:-1}

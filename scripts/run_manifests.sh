#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_paths.sh"

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/SIBA-Jittor}
SOURCE_ROOT=${SOURCE_ROOT:-/root/autodl-tmp/datasets/sources}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/datasets/SIBA}
PYTHON=${PYTHON:-$PYTORCH_PYTHON}

mkdir -p "$PROJECT_ROOT/data_manifests" "$DATA_ROOT"

"$PYTHON" "$PROJECT_ROOT/tools/build_dataset_manifest.py" \
  --infrared "$SOURCE_ROOT/MSRS/train/ir" \
  --visible "$SOURCE_ROOT/MSRS/train/vi" \
  --output "$PROJECT_ROOT/data_manifests/msrs_train.json"

"$PYTHON" "$PROJECT_ROOT/tools/build_dataset_manifest.py" \
  --infrared "$SOURCE_ROOT/MSRS/test/ir" \
  --visible "$SOURCE_ROOT/MSRS/test/vi" \
  --output "$PROJECT_ROOT/data_manifests/msrs_test.json"

"$PYTHON" "$PROJECT_ROOT/tools/build_dataset_manifest.py" \
  --infrared "$SOURCE_ROOT/RoadScene/cropinfrared" \
  --visible "$SOURCE_ROOT/RoadScene/crop_LR_visible" \
  --select 200 --seed 2025 \
  --output "$PROJECT_ROOT/data_manifests/roadscene_200_seed2025.json"

"$PYTHON" "$PROJECT_ROOT/tools/prepare_training_dataset.py" \
  --msrs-infrared "$SOURCE_ROOT/MSRS/train/ir" \
  --msrs-visible "$SOURCE_ROOT/MSRS/train/vi" \
  --roadscene-infrared "$SOURCE_ROOT/RoadScene/cropinfrared" \
  --roadscene-visible "$SOURCE_ROOT/RoadScene/crop_LR_visible" \
  --roadscene-manifest "$PROJECT_ROOT/data_manifests/roadscene_200_seed2025.json" \
  --output "$DATA_ROOT/train"

"$PYTHON" "$PROJECT_ROOT/tools/combine_training_manifests.py" \
  --msrs "$PROJECT_ROOT/data_manifests/msrs_train.json" \
  --roadscene "$PROJECT_ROOT/data_manifests/roadscene_200_seed2025.json" \
  --output "$PROJECT_ROOT/data_manifests/combined_training_1283.json"

find "$DATA_ROOT/train/ir" -maxdepth 1 -type f | wc -l
find "$DATA_ROOT/train/vi" -maxdepth 1 -type f | wc -l

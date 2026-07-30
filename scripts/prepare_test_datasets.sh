#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_paths.sh"

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/SIBA-Jittor}
SOURCE_ROOT=${SOURCE_ROOT:-/root/autodl-tmp/datasets/sources}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/datasets/SIBA}
OFFICIAL_M3FD_ROOT=${OFFICIAL_M3FD_ROOT:-/root/autodl-tmp/datasets/official}
OFFICIAL_TNO_ROOT=${OFFICIAL_TNO_ROOT:-/root/autodl-tmp/datasets/official/official_tno}
PYTHON=${PYTHON:-$PYTORCH_PYTHON}

"$PYTHON" "$PROJECT_ROOT/tools/prepare_test_datasets.py" \
  --msrs-infrared "$SOURCE_ROOT/MSRS/test/ir" \
  --msrs-visible "$SOURCE_ROOT/MSRS/test/vi" \
  --m3fd-infrared "$OFFICIAL_M3FD_ROOT/Ir" \
  --m3fd-visible "$OFFICIAL_M3FD_ROOT/Vis" \
  --tno-infrared "$OFFICIAL_TNO_ROOT/ir" \
  --tno-visible "$OFFICIAL_TNO_ROOT/vi" \
  --output "$DATA_ROOT/test"

for dataset in MSRS M3FD_2x TNO; do
  "$PYTHON" "$PROJECT_ROOT/tools/build_dataset_manifest.py" \
    --infrared "$DATA_ROOT/test/$dataset/ir" \
    --visible "$DATA_ROOT/test/$dataset/vi" \
    --output "$PROJECT_ROOT/data_manifests/${dataset,,}_test.json"
done

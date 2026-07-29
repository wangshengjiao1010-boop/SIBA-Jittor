#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT=${ENV_ROOT:-/root/autodl-tmp/envs}
DOWNLOAD_ROOT=${DOWNLOAD_ROOT:-/root/autodl-tmp/datasets/downloads}
GDOWN="$ENV_ROOT/siba_jittor/bin/gdown"

mkdir -p "$DOWNLOAD_ROOT/m3fd_tardal" "$DOWNLOAD_ROOT/tno_siba"

"$GDOWN" --folder \
  'https://drive.google.com/drive/folders/1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6?usp=sharing' \
  -O "$DOWNLOAD_ROOT/m3fd_tardal" --remaining-ok

"$GDOWN" --folder \
  'https://drive.google.com/drive/folders/1yURIsV9R9kEYLQovQ-vPogUkXqrIZswA?usp=drive_link' \
  -O "$DOWNLOAD_ROOT/tno_siba" --remaining-ok

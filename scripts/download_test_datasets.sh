#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_paths.sh"
DOWNLOAD_ROOT=${DOWNLOAD_ROOT:-/root/autodl-tmp/datasets/downloads}
GDOWN=${GDOWN:-$(resolve_env_executable gdown "$JITTOR_ENV_NAME" siba_jittor)}

mkdir -p "$DOWNLOAD_ROOT/m3fd_tardal" "$DOWNLOAD_ROOT/tno_siba"

"$GDOWN" --folder \
  'https://drive.google.com/drive/folders/1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6?usp=sharing' \
  -O "$DOWNLOAD_ROOT/m3fd_tardal" --remaining-ok

"$GDOWN" --folder \
  'https://drive.google.com/drive/folders/1yURIsV9R9kEYLQovQ-vPogUkXqrIZswA?usp=drive_link' \
  -O "$DOWNLOAD_ROOT/tno_siba" --remaining-ok

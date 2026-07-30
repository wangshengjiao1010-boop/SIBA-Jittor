#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_paths.sh"

export JITTOR_HOME=${JITTOR_HOME:-/root/autodl-tmp/.cache/jittor}
export nvcc_path=
export use_cuda=0
export DISABLE_MULTIPROCESSING=1

"$JITTOR_PYTHON" - <<'PY'
import jittor as jt

print("jittor", jt.__version__)
print((jt.ones((2, 2)) + 1).numpy())
print("JITTOR_CPU_OK")
PY

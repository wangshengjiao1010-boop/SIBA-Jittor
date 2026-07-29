#!/usr/bin/env bash
set -euo pipefail

export JITTOR_HOME=${JITTOR_HOME:-/root/autodl-tmp/.cache/jittor}
export nvcc_path=
export use_cuda=0
export DISABLE_MULTIPROCESSING=1

/root/autodl-tmp/envs/siba_jittor/bin/python - <<'PY'
import jittor as jt

print("jittor", jt.__version__)
print((jt.ones((2, 2)) + 1).numpy())
print("JITTOR_CPU_OK")
PY

#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT=${ENV_ROOT:-/root/autodl-tmp/envs}
source /root/miniconda3/etc/profile.d/conda.sh
mkdir -p "$ENV_ROOT"
conda config --add envs_dirs "$ENV_ROOT" >/dev/null 2>&1 || true

clone_if_needed() {
  local legacy=$1
  local target=$2
  if [[ -x "$target/bin/python" ]]; then
    echo "Already exists: $target"
    return
  fi
  if [[ ! -x "$legacy/bin/python" ]]; then
    echo "Missing tested source environment: $legacy" >&2
    return 1
  fi
  conda create -y -p "$target" --clone "$legacy"
}

clone_if_needed "$ENV_ROOT/siba_torch" "$ENV_ROOT/PytorchDome"
clone_if_needed "$ENV_ROOT/siba_jittor" "$ENV_ROOT/JittorDome"

echo "Named environments are ready."
echo "conda activate PytorchDome"
echo "conda activate JittorDome"

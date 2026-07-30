#!/usr/bin/env bash

ENV_ROOT=${ENV_ROOT:-/root/autodl-tmp/envs}
PYTORCH_ENV_NAME=${PYTORCH_ENV_NAME:-PytorchDome}
JITTOR_ENV_NAME=${JITTOR_ENV_NAME:-JittorDome}

resolve_env_executable() {
  local executable=$1
  local primary_name=$2
  local legacy_name=$3
  local primary="$ENV_ROOT/$primary_name/bin/$executable"
  local legacy="$ENV_ROOT/$legacy_name/bin/$executable"

  if [[ -x "$primary" ]]; then
    printf '%s\n' "$primary"
    return 0
  fi
  if [[ -x "$legacy" ]]; then
    printf '%s\n' "$legacy"
    return 0
  fi

  echo "Missing environment executable: $primary or $legacy" >&2
  return 1
}

PYTORCH_PYTHON=${PYTORCH_PYTHON:-$(resolve_env_executable python "$PYTORCH_ENV_NAME" siba_torch)}
JITTOR_PYTHON=${JITTOR_PYTHON:-$(resolve_env_executable python "$JITTOR_ENV_NAME" siba_jittor)}

export ENV_ROOT PYTORCH_ENV_NAME JITTOR_ENV_NAME
export PYTORCH_PYTHON JITTOR_PYTHON

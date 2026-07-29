#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/SIBA-Jittor}
ENV_ROOT=${ENV_ROOT:-/root/autodl-tmp/envs}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/root/autodl-tmp/.cache/pip}
export JITTOR_HOME=${JITTOR_HOME:-/root/autodl-tmp/.cache/jittor}

source /root/miniconda3/etc/profile.d/conda.sh
conda create -y -p "$ENV_ROOT/siba_torch" python=3.8.18 pip
conda create -y -p "$ENV_ROOT/siba_jittor" python=3.8.18 pip

"$ENV_ROOT/siba_torch/bin/python" -m pip install -i https://pypi.org/simple --upgrade 'pip<25' setuptools wheel
"$ENV_ROOT/siba_torch/bin/python" -m pip install \
  torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
"$ENV_ROOT/siba_torch/bin/python" -m pip install -i https://pypi.org/simple \
  imageio==2.32.0 numpy==1.24.4 opencv-python==4.8.1.78 pillow==10.0.1 \
  timm==0.9.10 einops==0.7.0 tqdm==4.66.1 kornia==0.7.0 \
  scipy matplotlib pandas scikit-image pyyaml

"$ENV_ROOT/siba_jittor/bin/python" -m pip install -i https://pypi.org/simple --upgrade 'pip<25' setuptools wheel
DISABLE_MULTIPROCESSING=1 "$ENV_ROOT/siba_jittor/bin/python" -m pip install -i https://pypi.org/simple \
  jittor==1.3.11.0 imageio==2.32.0 numpy==1.24.4 \
  opencv-python==4.8.1.78 pillow==10.0.1 tqdm==4.66.1 \
  scipy matplotlib pandas scikit-image pyyaml

echo "Environment setup complete: $PROJECT_ROOT"


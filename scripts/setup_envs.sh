#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/SIBA-Jittor}
ENV_ROOT=${ENV_ROOT:-/root/autodl-tmp/envs}
PYTORCH_ENV=${PYTORCH_ENV:-$ENV_ROOT/PytorchDome}
JITTOR_ENV=${JITTOR_ENV:-$ENV_ROOT/JittorDome}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/root/autodl-tmp/.cache/pip}
export JITTOR_HOME=${JITTOR_HOME:-/root/autodl-tmp/.cache/jittor}

source /root/miniconda3/etc/profile.d/conda.sh
mkdir -p "$ENV_ROOT"
conda config --add envs_dirs "$ENV_ROOT" >/dev/null 2>&1 || true
conda create -y -p "$PYTORCH_ENV" python=3.8.18 pip
conda create -y -p "$JITTOR_ENV" python=3.8.18 pip

"$PYTORCH_ENV/bin/python" -m pip install -i https://pypi.org/simple --upgrade 'pip<25' setuptools wheel
"$PYTORCH_ENV/bin/python" -m pip install \
  torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
"$PYTORCH_ENV/bin/python" -m pip install -i https://pypi.org/simple \
  imageio==2.32.0 numpy==1.24.4 opencv-python==4.8.1.78 pillow==10.0.1 \
  timm==0.9.10 einops==0.7.0 tqdm==4.66.1 kornia==0.7.0 \
  scipy matplotlib pandas scikit-image pyyaml

"$JITTOR_ENV/bin/python" -m pip install -i https://pypi.org/simple --upgrade 'pip<25' setuptools wheel
DISABLE_MULTIPROCESSING=1 "$JITTOR_ENV/bin/python" -m pip install -i https://pypi.org/simple \
  jittor==1.3.11.0 imageio==2.32.0 numpy==1.24.4 \
  opencv-python==4.8.1.78 pillow==10.0.1 tqdm==4.66.1 \
  scipy matplotlib pandas scikit-image pyyaml

echo "Environment setup complete: $PROJECT_ROOT"
echo "PyTorch: conda activate PytorchDome"
echo "Jittor:  conda activate JittorDome"

#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT=${ENV_ROOT:-/root/autodl-tmp/envs}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/root/autodl-tmp/.cache/pip}
export JITTOR_HOME=${JITTOR_HOME:-/root/autodl-tmp/.cache/jittor}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_paths.sh"

"$PYTORCH_PYTHON" -m pip install -i https://pypi.org/simple \
  imageio==2.32.0 numpy==1.24.4 opencv-python==4.8.1.78 pillow==10.0.1 \
  timm==0.9.10 einops==0.7.0 tqdm==4.66.1 kornia==0.7.0 \
  scipy matplotlib pandas scikit-image pyyaml gdown

"$JITTOR_PYTHON" -m pip install -i https://pypi.org/simple --upgrade 'pip<25' setuptools wheel
DISABLE_MULTIPROCESSING=1 "$JITTOR_PYTHON" -m pip install -i https://pypi.org/simple \
  jittor==1.3.11.0 imageio==2.32.0 numpy==1.24.4 \
  opencv-python==4.8.1.78 pillow==10.0.1 tqdm==4.66.1 \
  scipy matplotlib pandas scikit-image pyyaml gdown

"$PYTORCH_PYTHON" - <<'PY'
import cv2
import kornia
import numpy
import PIL
import torch
import torchvision

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchvision", torchvision.__version__, "kornia", kornia.__version__)
print("opencv", cv2.__version__, "pillow", PIL.__version__, "numpy", numpy.__version__)
PY

nvcc_path= use_cuda=0 DISABLE_MULTIPROCESSING=1 "$JITTOR_PYTHON" - <<'PY'
import cv2
import jittor as jt
import numpy
import PIL

print("jittor", jt.__version__)
print("opencv", cv2.__version__, "pillow", PIL.__version__, "numpy", numpy.__version__)
print((jt.ones((2, 2)) + 1).numpy())
PY

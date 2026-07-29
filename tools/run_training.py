#!/usr/bin/env python3
import argparse
import importlib
import os
import random
import runpy
import sys
from pathlib import Path

import numpy as np


def configure_seed(framework, seed):
    random.seed(seed)
    np.random.seed(seed)
    if framework == "pytorch":
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        import jittor as jt

        jt.set_global_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=("pytorch", "jittor"), required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ir-path", type=Path, required=True)
    parser.add_argument("--vi-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--gpu-number", default="0")
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    source = args.project_root / (
        "official_pytorch" if args.framework == "pytorch" else "siba_jittor"
    )
    os.chdir(source)
    sys.path.insert(0, str(source))

    args_module = importlib.import_module("args.args_SIBA")
    args_module.args.ir_path = str(args.ir_path.resolve())
    args_module.args.vi_path = str(args.vi_path.resolve())
    args_module.args.model_save_path = str(args.output.resolve())
    args_module.args.epochs = args.epochs
    args_module.args.use_gpu_number = args.gpu_number
    args_module.args.use_gpu = True

    configure_seed(args.framework, args.seed)
    runpy.run_path(str(source / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()


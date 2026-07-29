#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import jittor as jt
import numpy as np


def finite(name, value):
    array = value.numpy()
    if not np.isfinite(array).all():
        raise RuntimeError(f"Non-finite tensor: {name}")
    return {
        "shape": list(array.shape),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    jt.flags.use_cuda = int(args.use_cuda)
    jt.set_global_seed(args.seed)
    source = args.project_root / "siba_jittor"
    sys.path.insert(0, str(source))

    from compat.pytorch_adam import PyTorchAdam
    from compat.pytorch_clip import clip_grad_norm_pytorch
    from loss.loss import Fusionloss, JointGrad
    from models.SIBA import SIBA

    random = np.random.RandomState(args.seed)
    infrared = jt.array(
        random.rand(args.batch_size, 1, args.height, args.width).astype(np.float32)
    )
    visible = jt.array(
        random.rand(args.batch_size, 1, args.height, args.width).astype(np.float32)
    )

    model = SIBA()
    named_parameters = list(model.named_parameters())
    parameter_count = sum(int(parameter.numel()) for _, parameter in named_parameters)
    output = model(infrared, visible)
    loss_laplacian = JointGrad()(output, infrared, visible)
    loss_intensity, loss_sobel = Fusionloss()(output, infrared, visible)
    loss_total = 10 * loss_laplacian + 0.1 * loss_intensity + loss_sobel

    optimizer = PyTorchAdam(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )
    optimizer.zero_grad()
    optimizer.backward(loss_total)
    gradients = optimizer.param_groups[0]["grads"]
    clip_grad_norm_pytorch(optimizer, max_norm=0.01, norm_type=2)
    optimizer.step()
    jt.sync_all(True)

    report = {
        "jittor_version": jt.__version__,
        "use_cuda": bool(args.use_cuda),
        "parameter_tensors": len(named_parameters),
        "parameter_count": parameter_count,
        "gradient_tensors": len(gradients),
        "output": finite("output", output),
        "loss_laplacian": float(loss_laplacian.item()),
        "loss_intensity": float(loss_intensity.item()),
        "loss_sobel": float(loss_sobel.item()),
        "loss_total": float(loss_total.item()),
    }
    for name, gradient in zip((name for name, _ in named_parameters), gradients):
        finite(f"gradient:{name}", gradient)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

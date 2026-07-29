#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Run a short, real-data SIBA Jittor training demonstration."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--ir-path", type=Path, required=True)
    parser.add_argument("--vi-path", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--gpu-number", default="0")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--save-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number
    np.random.seed(args.seed)

    import jittor as jt

    jt.flags.use_cuda = 1
    jt.set_global_seed(args.seed)
    source = args.project_root / "siba_jittor"
    sys.path.insert(0, str(source))

    from compat.pytorch_adam import PyTorchAdam
    from compat.pytorch_clip import clip_grad_norm_pytorch
    from loader.train_loader import TrainLoader
    from loss.loss import Fusionloss, JointGrad
    from models.SIBA import SIBA

    model = SIBA()
    if args.checkpoint:
        checkpoint = jt.load(str(args.checkpoint))
        model.load_parameters(checkpoint["model"])
    model.train()
    optimizer = PyTorchAdam(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )
    joint_grad = JointGrad()
    fusion_loss = Fusionloss()
    dataset = TrainLoader(args.ir_path, args.vi_path, args.patch_size)
    loader = dataset.set_attrs(
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    records = []
    started = time.perf_counter()
    step = 0
    while step < args.steps:
        for infrared, visible in loader:
            step += 1
            optimizer.zero_grad()
            fused = model(infrared, visible)
            loss_laplacian = joint_grad(fused, infrared, visible)
            loss_intensity, loss_sobel = fusion_loss(fused, infrared, visible)
            loss_total = 10 * loss_laplacian + 0.1 * loss_intensity + loss_sobel
            optimizer.backward(loss_total)
            clip_grad_norm_pytorch(optimizer, max_norm=0.01, norm_type=2)
            optimizer.step()
            jt.sync_all(True)
            record = {
                "step": step,
                "loss_total": float(loss_total.item()),
                "loss_laplacian": float(loss_laplacian.item()),
                "loss_intensity": float(loss_intensity.item()),
                "loss_sobel": float(loss_sobel.item()),
                "elapsed_seconds": time.perf_counter() - started,
            }
            records.append(record)
            print(
                "[demo step {step}/{total}] total={loss_total:.6f} "
                "laplacian={loss_laplacian:.6f} intensity={loss_intensity:.6f} "
                "sobel={loss_sobel:.6f} elapsed={elapsed:.2f}s".format(
                    step=step,
                    total=args.steps,
                    loss_total=record["loss_total"],
                    loss_laplacian=record["loss_laplacian"],
                    loss_intensity=record["loss_intensity"],
                    loss_sobel=record["loss_sobel"],
                    elapsed=record["elapsed_seconds"],
                ),
                flush=True,
            )
            if step >= args.steps:
                break

    if args.save_checkpoint:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        jt.save({"model": model.state_dict()}, str(args.save_checkpoint))
    report = {
        "purpose": "live demonstration only; not used for paper metrics",
        "uses_real_training_images": True,
        "ir_path": str(args.ir_path.resolve()),
        "vi_path": str(args.vi_path.resolve()),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "seed": args.seed,
        "loaded_checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "saved_checkpoint": (
            str(args.save_checkpoint.resolve()) if args.save_checkpoint else None
        ),
        "records": records,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()

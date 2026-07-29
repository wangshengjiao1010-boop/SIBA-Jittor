#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first(value):
    if isinstance(value, (list, tuple)):
        return first(value[0])
    if hasattr(value, "reshape") and not isinstance(value, str):
        try:
            return value.reshape(-1)[0].item()
        except (AttributeError, TypeError, ValueError):
            pass
    return value


def run_pytorch(args, source):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    sys.path.insert(0, str(source))
    from loader.test_loader import TestLoader
    from models.SIBA import SIBA
    from utils.RGB2YCrBb import YCrCb2RGB, clamp

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number
    device = torch.device("cuda" if args.use_cuda else "cpu")
    model = SIBA().to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset = TestLoader(str(args.data_dir))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    to_pil = transforms.ToPILImage()
    rows = []
    with torch.no_grad():
        for _, visible_y, cb, cr, infrared, image_name, image_size in loader:
            visible_y = visible_y.to(device)
            cb = cb.to(device)
            cr = cr.to(device)
            infrared = infrared.to(device)
            if not rows:
                for _ in range(args.warmup_runs):
                    model(infrared, visible_y)
                if args.use_cuda and args.timing_mode == "synchronized":
                    torch.cuda.synchronize()
            if args.use_cuda and args.timing_mode == "synchronized":
                torch.cuda.synchronize()
            start = time.perf_counter()
            fused = model(infrared, visible_y)
            if args.use_cuda and args.timing_mode == "synchronized":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            fused = clamp(fused[0])
            fused = YCrCb2RGB(fused, cb[0], cr[0]).detach().cpu()
            name = str(first(image_name))
            if not args.skip_save:
                to_pil(fused).save(args.output / name)
            size = first(image_size)
            rows.append(
                {
                    "name": name,
                    "width": int(size[0]) if isinstance(size, (list, tuple)) else None,
                    "height": int(size[1]) if isinstance(size, (list, tuple)) else None,
                    "seconds": elapsed,
                }
            )
    return model, rows, torch.__version__


def run_jittor(args, source):
    import jittor as jt
    from jittor import transform

    jt.flags.use_cuda = int(args.use_cuda)
    sys.path.insert(0, str(source))
    from loader.test_loader import TestLoader
    from models.SIBA import SIBA
    from utils.RGB2YCrBb import YCrCb2RGB, clamp

    model = SIBA()
    checkpoint = jt.load(str(args.checkpoint))
    model.load_parameters(checkpoint["model"])
    model.eval()
    dataset = TestLoader(str(args.data_dir))
    loader = dataset.set_attrs(
        batch_size=1, shuffle=False, num_workers=0, drop_last=False
    )
    to_pil = transform.ToPILImage()
    rows = []
    with jt.no_grad():
        for _, visible_y, cb, cr, infrared, image_name, image_size in loader:
            if not rows:
                for _ in range(args.warmup_runs):
                    model(infrared, visible_y)
                    if args.timing_mode == "synchronized":
                        jt.sync_all(True)
            if args.timing_mode == "synchronized":
                jt.sync_all(True)
            start = time.perf_counter()
            fused = model(infrared, visible_y)
            if args.timing_mode == "synchronized":
                jt.sync_all(True)
            elapsed = time.perf_counter() - start
            fused = clamp(fused[0])
            fused = YCrCb2RGB(fused, cb[0], cr[0])
            name = str(first(image_name))
            fused_hwc = fused.transpose(1, 2, 0)
            if args.skip_save:
                fused_hwc.numpy()
            else:
                to_pil(fused_hwc).save(args.output / name)
            size = first(image_size)
            rows.append(
                {
                    "name": name,
                    "width": int(size[0]) if isinstance(size, (list, tuple)) else None,
                    "height": int(size[1]) if isinstance(size, (list, tuple)) else None,
                    "seconds": elapsed,
                }
            )
    return model, rows, jt.__version__


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=("pytorch", "jittor"), required=True)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-number", default="0")
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument(
        "--timing-mode",
        choices=("synchronized", "official"),
        default="synchronized",
        help="official reproduces the unsynchronized timer in official test.py",
    )
    parser.add_argument("--skip-save", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    source = args.project_root / (
        "official_pytorch" if args.framework == "pytorch" else "siba_jittor"
    )
    if args.framework == "pytorch":
        model, rows, version = run_pytorch(args, source)
    else:
        model, rows, version = run_jittor(args, source)

    with (args.output / "timing.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=("name", "width", "height", "seconds", "fps")
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "fps": 1.0 / row["seconds"]})

    total_seconds = sum(row["seconds"] for row in rows)
    summary = {
        "framework": args.framework,
        "framework_version": version,
        "use_cuda": args.use_cuda,
        "warmup_runs": args.warmup_runs,
        "timing_mode": args.timing_mode,
        "skip_save": args.skip_save,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "data_dir": str(args.data_dir.resolve()),
        "image_count": len(rows),
        "parameter_count": sum(int(parameter.numel()) for parameter in model.parameters()),
        "total_model_seconds": total_seconds,
        "mean_model_seconds": total_seconds / len(rows),
        "model_fps": len(rows) / total_seconds,
    }
    summary[
        "synchronized_fps"
        if args.timing_mode == "synchronized"
        else "official_unsynchronized_fps"
    ] = summary["model_fps"]
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Edit these paths before running on another machine. This follows the
# path-at-entry layout of the official SIBA test.py.
model_path = "./checkpoint/SIBA_jittor_self_trained_epoch60.pkl"
testdata_paths = {
    "MSRS": "/root/autodl-tmp/datasets/SIBA/test/MSRS",
    "M3FD_2x": "/root/autodl-tmp/datasets/SIBA/test/M3FD_2x",
    "TNO": "/root/autodl-tmp/datasets/SIBA/test/TNO",
}
result_save_path = "./results/jittor_test"
test_dataset = "all"
use_gpu_number = "0"
use_gpu = True

default_output_root = PROJECT_ROOT / result_save_path
configured_datasets = testdata_paths


def project_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path

parser = argparse.ArgumentParser(
    description="Test SIBA with Jittor; paths are configured near the top of test.py"
)
parser.add_argument("--checkpoint", default=model_path)
parser.add_argument("--dataset", default=test_dataset)
parser.add_argument("--data-dir")
parser.add_argument("--output")
parser.add_argument("--gpu-number", default=use_gpu_number)
parser.add_argument("--cpu", action="store_true", default=not use_gpu)
runtime_args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = runtime_args.gpu_number

import jittor as jt
from jittor import transform
from tqdm import tqdm

from loader.test_loader import TestLoader
from models.SIBA import SIBA
from utils.RGB2YCrBb import YCrCb2RGB, clamp


def selected_datasets():
    if runtime_args.data_dir:
        name = (
            runtime_args.dataset
            if runtime_args.dataset != "all"
            else Path(runtime_args.data_dir).name
        )
        output = runtime_args.output or str(default_output_root / name)
        return [(name, project_path(runtime_args.data_dir), project_path(output))]
    if runtime_args.dataset == "all":
        return [
            (name, project_path(data_dir), default_output_root / name)
            for name, data_dir in configured_datasets.items()
        ]
    if runtime_args.dataset not in configured_datasets:
        raise ValueError(
            f"Unknown dataset {runtime_args.dataset!r}; choose all or {sorted(configured_datasets)}"
        )
    output = runtime_args.output or str(default_output_root / runtime_args.dataset)
    return [
        (
            runtime_args.dataset,
            project_path(configured_datasets[runtime_args.dataset]),
            project_path(output),
        )
    ]


def run_dataset(model, name, data_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    test_dataset = TestLoader(str(data_dir))
    test_loader = test_dataset.set_attrs(
        batch_size=1, shuffle=False, num_workers=1, drop_last=False
    )
    elapsed = 0.0
    timing_rows = []
    with jt.no_grad():
        for _, vis_y_image, cb, cr, ir_image, img_name, _ in tqdm(
            test_loader, total=test_loader.__batch_len__(), desc=name
        ):
            jt.sync_all(True)
            start = time.perf_counter()
            image_fused = model(ir_image, vis_y_image)
            jt.sync_all(True)
            seconds = time.perf_counter() - start
            elapsed += seconds
            image_fused = clamp(image_fused[0])
            image_fused = YCrCb2RGB(image_fused, cb[0], cr[0])
            image_fused = transform.ToPILImage()(image_fused.transpose(1, 2, 0))
            image_fused.save(str(output_dir / img_name[0]))
            timing_rows.append(
                {
                    "name": img_name[0],
                    "seconds": seconds,
                    "fps": 1.0 / seconds,
                }
            )
    with (output_dir / "timing.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("name", "seconds", "fps"))
        writer.writeheader()
        writer.writerows(timing_rows)
    summary = {
        "framework": "jittor",
        "framework_version": jt.__version__,
        "dataset": name,
        "image_count": len(test_dataset),
        "timing_mode": "synchronized model forward",
        "warmup_runs": 0,
        "total_model_seconds": elapsed,
        "mean_model_seconds": elapsed / len(test_dataset),
        "model_fps": len(test_dataset) / elapsed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{name}: {len(test_dataset)} images, model time {elapsed:.3f}s, "
        f"{summary['model_fps']:.3f} FPS"
    )


def main():
    jt.flags.use_cuda = 0 if runtime_args.cpu else 1
    model_path = project_path(runtime_args.checkpoint)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}. Update model_path in test.py."
        )
    model = SIBA()
    checkpoint = jt.load(str(model_path))
    model.load_parameters(checkpoint.get("model", checkpoint))
    total = sum(params.numel() for params in model.parameters())
    print("Number of params: {%.3f M}" % (total / 1e6))
    model.eval()
    for name, data_dir, output_dir in selected_datasets():
        run_dataset(model, name, data_dir, output_dir)


if __name__ == "__main__":
    main()

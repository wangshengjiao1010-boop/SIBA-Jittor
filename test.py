import argparse
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Edit these paths before running on another machine. This follows the
# path-at-entry layout of the official SIBA test.py.
model_path = "./checkpoint/SIBA_epoch60.pkl"
testdata_paths = {
    "MSRS": "/root/autodl-tmp/datasets/SIBA/test/MSRS",
    "M3FD_2x": "/root/autodl-tmp/datasets/SIBA/test/M3FD_2x",
    "TNO": "/root/autodl-tmp/datasets/SIBA/test/TNO",
}
result_save_path = "./results/jittor_test"
test_dataset = "all"
use_gpu_number = "0"
use_gpu = True

default_output_root = Path(result_save_path)
configured_datasets = testdata_paths

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
        return [(name, Path(runtime_args.data_dir), Path(output))]
    if runtime_args.dataset == "all":
        return [
            (name, Path(data_dir), default_output_root / name)
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
            Path(configured_datasets[runtime_args.dataset]),
            Path(output),
        )
    ]


def run_dataset(model, name, data_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    test_dataset = TestLoader(str(data_dir))
    test_loader = test_dataset.set_attrs(
        batch_size=1, shuffle=False, num_workers=1, drop_last=False
    )
    elapsed = 0.0
    with jt.no_grad():
        for _, vis_y_image, cb, cr, ir_image, img_name, _ in tqdm(
            test_loader, total=test_loader.__batch_len__(), desc=name
        ):
            start = time.time()
            image_fused = model(ir_image, vis_y_image)
            elapsed += time.time() - start
            image_fused = clamp(image_fused[0])
            image_fused = YCrCb2RGB(image_fused, cb[0], cr[0])
            image_fused = transform.ToPILImage()(image_fused.transpose(1, 2, 0))
            image_fused.save(str(output_dir / img_name[0]))
    print(f"{name}: {len(test_dataset)} images, model time {elapsed:.3f}s")


def main():
    jt.flags.use_cuda = 0 if runtime_args.cpu else 1
    model_path = Path(runtime_args.checkpoint)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}. Update model_path in test.py."
        )
    model = SIBA()
    model.load_parameters(jt.load(str(model_path))["model"])
    total = sum(params.numel() for params in model.parameters())
    print("Number of params: {%.3f M}" % (total / 1e6))
    model.eval()
    for name, data_dir, output_dir in selected_datasets():
        run_dataset(model, name, data_dir, output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
ENTRY = re.compile(r"\[Epoch (?P<epoch>\d+)/(?P<epochs>\d+)\].*\[loss: [^\]]+\]")
EXPECTED_DATASETS = {"MSRS": 361, "M3FD_2x": 300, "TNO": 45}
EXPECTED_EXPERIMENTS = {
    "OfficialPyTorch",
    "OfficialJittor",
    "PyTorchSelfTrained",
    "JittorSelfTrained",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def images(directory):
    if not directory.exists():
        return 0
    return sum(
        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        for path in directory.iterdir()
    )


def training_log(path):
    epochs = []
    entries = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = ENTRY.search(line)
            if match:
                entries += 1
                epochs.append(int(match.group("epoch")))
    return {
        "path": str(path),
        "exists": path.exists(),
        "entries": entries,
        "epochs": sorted(set(epochs)),
        "complete_60_epochs": entries == 420 and sorted(set(epochs)) == list(range(60)),
    }


def single_checkpoint(directory, suffix):
    files = sorted(directory.glob(f"**/*{suffix}")) if directory.exists() else []
    return {
        "directory": str(directory),
        "count": len(files),
        "files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        ],
        "valid": len(files) == 1 and files[0].stat().st_size > 1_000_000,
    }


def inference_set(directory, expected):
    summary_path = directory / "summary.json"
    timing_path = directory / "timing.csv"
    summary = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    image_count = images(directory)
    timing_count = 0
    if timing_path.exists():
        with timing_path.open("r", newline="", encoding="utf-8-sig") as file:
            timing_count = sum(1 for _ in csv.DictReader(file))
    return {
        "directory": str(directory),
        "image_count": image_count,
        "timing_rows": timing_count,
        "expected": expected,
        "summary_exists": summary is not None,
        "summary_image_count": summary.get("image_count") if summary else None,
        "valid": bool(
            summary
            and summary.get("image_count") == expected
            and timing_count == expected
            and image_count in (0, expected)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--require-gpu-complete", action="store_true")
    parser.add_argument("--require-metrics-complete", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project = args.project_root.resolve()
    tag = args.run_tag
    checks = {}
    checks["test_data"] = {
        dataset: {
            "ir": images(args.data_root / "test" / dataset / "ir"),
            "vi": images(args.data_root / "test" / dataset / "vi"),
            "expected": expected,
        }
        for dataset, expected in EXPECTED_DATASETS.items()
    }
    checks["training_logs"] = {
        "jittor": training_log(project / "logs/final/jittor_train_60e.log"),
        "pytorch": training_log(project / "logs/final/pytorch_train_60e.log"),
    }
    checks["checkpoints"] = {
        "jittor": single_checkpoint(project / "checkpoint", ".pkl"),
        "pytorch": single_checkpoint(project / "checkpoint", ".pth"),
    }
    self_root = project / "results" / f"full_{tag}"
    official_root = project / "results" / f"official_checkpoint_alignment_{tag}"
    checks["self_trained_inference"] = {
        framework: {
            dataset: inference_set(self_root / framework / dataset, expected)
            for dataset, expected in EXPECTED_DATASETS.items()
        }
        for framework in ("jittor", "pytorch")
    }
    checks["official_checkpoint_inference"] = {
        framework: {
            dataset: inference_set(official_root / framework / dataset, expected)
            for dataset, expected in EXPECTED_DATASETS.items()
        }
        for framework in ("jittor", "pytorch")
    }
    alignment = {}
    for dataset, expected in EXPECTED_DATASETS.items():
        path = project / "results" / f"output_alignment_{tag}" / dataset / "summary.json"
        value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        alignment[dataset] = {
            "path": str(path),
            "exists": value is not None,
            "compared_images": value.get("compared_images") if value else None,
            "filenames_and_shapes_match": value.get("all_filenames_and_shapes_match") if value else None,
            "valid": bool(
                value
                and value.get("compared_images") == expected
                and value.get("all_filenames_and_shapes_match")
            ),
        }
    checks["official_output_alignment"] = alignment
    checks["markers"] = {
        "training": (project / "logs" / f"full_sequence_{tag}" / "TRAINING_COMPLETE").exists(),
        "remaining_gpu": (project / "logs" / f"remaining_gpu_{tag}" / "REMAINING_GPU_COMPLETE").exists(),
    }
    metrics_root = project / "results" / f"metrics_{tag}"
    metrics_summary = metrics_root / "metrics_summary.csv"
    metric_rows = []
    if metrics_summary.exists():
        with metrics_summary.open("r", newline="", encoding="utf-8-sig") as file:
            metric_rows = list(csv.DictReader(file))
    expected_metric_pairs = {
        (experiment, dataset)
        for experiment in EXPECTED_EXPERIMENTS
        for dataset in EXPECTED_DATASETS
    }
    actual_metric_pairs = {
        (row.get("experiment"), row.get("dataset")) for row in metric_rows
    }
    metric_files = {}
    for experiment in EXPECTED_EXPERIMENTS:
        metric_files[experiment] = {}
        for dataset, expected in EXPECTED_DATASETS.items():
            per_image = metrics_root / experiment / dataset / "per_image.csv"
            summary = metrics_root / experiment / dataset / "per_image_summary.csv"
            row_count = 0
            if per_image.exists():
                with per_image.open("r", newline="", encoding="utf-8-sig") as file:
                    row_count = sum(1 for _ in csv.DictReader(file))
            metric_files[experiment][dataset] = {
                "per_image": str(per_image),
                "summary": str(summary),
                "row_count": row_count,
                "expected": expected,
                "valid": per_image.exists() and summary.exists() and row_count == expected,
            }
    metrics_valid = (
        actual_metric_pairs == expected_metric_pairs
        and all(
            item["valid"]
            for experiment in metric_files.values()
            for item in experiment.values()
        )
    )
    checks["metrics"] = {
        "root": str(metrics_root),
        "summary": str(metrics_summary),
        "summary_rows": len(metric_rows),
        "expected_rows": len(expected_metric_pairs),
        "files": metric_files,
        "valid": metrics_valid,
    }

    base_valid = (
        all(item["complete_60_epochs"] for item in checks["training_logs"].values())
        and all(item["valid"] for item in checks["checkpoints"].values())
        and all(
            value["ir"] == value["expected"] and value["vi"] == value["expected"]
            for value in checks["test_data"].values()
        )
        and checks["markers"]["training"]
    )
    gpu_valid = (
        all(
            result["valid"]
            for framework in checks["self_trained_inference"].values()
            for result in framework.values()
        )
        and all(
            result["valid"]
            for framework in checks["official_checkpoint_inference"].values()
            for result in framework.values()
        )
        and all(value["valid"] for value in alignment.values())
        and checks["markers"]["remaining_gpu"]
    )
    report = {
        "run_tag": tag,
        "base_complete": base_valid,
        "gpu_complete": gpu_valid,
        "metrics_complete": metrics_valid,
        "complete": base_valid and gpu_valid and metrics_valid,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("run_tag", "base_complete", "gpu_complete", "metrics_complete", "complete")}, indent=2))
    if (
        not base_valid
        or (args.require_gpu_complete and not gpu_valid)
        or (args.require_metrics_complete and not metrics_valid)
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
from pathlib import Path


DATASETS = ("MSRS", "M3FD_2x", "TNO")
METRICS = ("VIF", "SCD", "MI", "Qabf", "SSIM", "MS_SSIM", "FMI")


def matlab_path(path):
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def read_single_row(path):
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one summary row in {path}, found {len(rows)}")
    return rows[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matlab", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tools-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        metavar="LABEL=RESULT_ROOT",
        help="RESULT_ROOT must contain MSRS, M3FD_2x, and TNO directories.",
    )
    args = parser.parse_args()

    experiments = []
    for value in args.experiment:
        if "=" not in value:
            raise ValueError(f"Invalid --experiment value: {value}")
        label, root = value.split("=", 1)
        experiments.append((label.strip(), Path(root).resolve()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = []
    commands = []
    for label, result_root in experiments:
        for dataset in DATASETS:
            dataset_root = args.data_root / dataset
            fused_dir = result_root / dataset
            if not fused_dir.exists():
                raise FileNotFoundError(fused_dir)
            run_dir = args.output_dir / label / dataset
            run_dir.mkdir(parents=True, exist_ok=True)
            output_csv = run_dir / "per_image.csv"
            summary_path = run_dir / "per_image_summary.csv"
            expression = (
                f"addpath('{matlab_path(args.tools_dir)}'); "
                f"evaluate_official_metrics('{matlab_path(args.evaluation_dir)}',"
                f"'{matlab_path(dataset_root / 'ir')}',"
                f"'{matlab_path(dataset_root / 'vi')}',"
                f"'{matlab_path(fused_dir)}',"
                f"'{matlab_path(output_csv)}')"
            )
            command = [str(args.matlab), "-batch", expression]
            log = run_dir / "matlab.log"
            skipped = args.resume and output_csv.exists() and summary_path.exists()
            print(f"{'Skipping' if skipped else 'Evaluating'} {label}/{dataset}", flush=True)
            if skipped:
                returncode = 0
            else:
                completed = subprocess.run(command, capture_output=True, text=True)
                returncode = completed.returncode
                log.write_text(
                    completed.stdout + completed.stderr,
                    encoding="utf-8",
                    errors="replace",
                )
            commands.append(
                {
                    "label": label,
                    "dataset": dataset,
                    "command": command,
                    "returncode": returncode,
                    "skipped": skipped,
                    "log": str(log.resolve()),
                }
            )
            if returncode != 0:
                raise RuntimeError(f"MATLAB evaluation failed; see {log}")
            values = read_single_row(summary_path)
            combined.append(
                {
                    "experiment": label,
                    "dataset": dataset,
                    **{metric: values[metric] for metric in METRICS},
                }
            )
            print(f"Completed {label}/{dataset}: {values}", flush=True)

    combined_path = args.output_dir / "metrics_summary.csv"
    with combined_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=("experiment", "dataset", *METRICS))
        writer.writeheader()
        writer.writerows(combined)
    report = {
        "metric_implementation": str(args.evaluation_dir.resolve()),
        "data_root": str(args.data_root.resolve()),
        "experiments": [{"label": label, "root": str(root)} for label, root in experiments],
        "commands": commands,
        "summary_csv": str(combined_path.resolve()),
    }
    (args.output_dir / "evaluation_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ENTRY = re.compile(
    r"\[Epoch (?P<epoch>\d+)/(?P<epochs>\d+)\] "
    r"\[lr (?P<lr>[^\]]+)\] \[loss: (?P<loss>[^\]]+)\] ETA: (?P<eta>.+)"
)


def parse_run(label, path):
    rows = []
    epoch_ordinals = defaultdict(int)
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        line = ANSI_ESCAPE.sub("", raw_line).strip()
        match = ENTRY.search(line)
        if not match:
            continue
        epoch = int(match.group("epoch"))
        ordinal = epoch_ordinals[epoch]
        epoch_ordinals[epoch] += 1
        rows.append(
            {
                "framework": label,
                "source_log": str(path),
                "line_number": line_number,
                "epoch_zero_based": epoch,
                "epoch_one_based": epoch + 1,
                "configured_epochs": int(match.group("epochs")),
                "logged_sample_in_epoch": ordinal,
                "learning_rate": float(match.group("lr")),
                "loss": float(match.group("loss")),
                "eta": match.group("eta").strip(),
            }
        )
    if not rows:
        raise RuntimeError(f"No training entries found in {path}")
    return rows


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_epochs(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["framework"], row["epoch_zero_based"])].append(row)
    summaries = []
    for (framework, epoch), values in sorted(grouped.items()):
        losses = np.asarray([value["loss"] for value in values], dtype=np.float64)
        summaries.append(
            {
                "framework": framework,
                "epoch_zero_based": epoch,
                "epoch_one_based": epoch + 1,
                "logged_samples": len(values),
                "learning_rate": values[-1]["learning_rate"],
                "loss_mean": float(losses.mean()),
                "loss_median": float(np.median(losses)),
                "loss_min": float(losses.min()),
                "loss_max": float(losses.max()),
                "loss_first": float(losses[0]),
                "loss_last": float(losses[-1]),
            }
        )
    return summaries


def plot_epochs(path, summaries):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axis = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    colors = ["#2369A8", "#D06B32", "#3A8D5D", "#7A5AA6"]
    frameworks = sorted({row["framework"] for row in summaries})
    for color, framework in zip(colors, frameworks):
        values = [row for row in summaries if row["framework"] == framework]
        epochs = np.asarray([row["epoch_one_based"] for row in values])
        means = np.asarray([row["loss_mean"] for row in values])
        lower = np.asarray([row["loss_min"] for row in values])
        upper = np.asarray([row["loss_max"] for row in values])
        axis.plot(epochs, means, color=color, linewidth=2.0, label=f"{framework} mean")
        axis.fill_between(epochs, lower, upper, color=color, alpha=0.12, linewidth=0)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Logged training loss")
    axis.set_title("SIBA training loss (official logging interval: every 50 batches)")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    axis.legend(frameon=False, ncol=max(1, len(frameworks)))
    figure.savefig(path.with_suffix(".png"), dpi=240, facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=LOG",
        help="Repeat for each framework, for example Jittor=logs/jittor/train.log",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    all_rows = []
    run_inputs = []
    for value in args.run:
        if "=" not in value:
            raise ValueError(f"Invalid --run value: {value}")
        label, raw_path = value.split("=", 1)
        path = Path(raw_path).resolve()
        rows = parse_run(label.strip(), path)
        all_rows.extend(rows)
        run_inputs.append({"framework": label.strip(), "log": str(path), "entries": len(rows)})

    summaries = summarize_epochs(all_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "training_samples.csv",
        list(all_rows[0].keys()),
        all_rows,
    )
    write_csv(
        args.output_dir / "epoch_loss.csv",
        list(summaries[0].keys()),
        summaries,
    )
    plot_epochs(args.output_dir / "loss_curve", summaries)

    report = {
        "logging_policy": "The official train.py prints one sample every 50 batches.",
        "comparison_note": (
            "PyTorch and Jittor use different framework shuffle implementations. "
            "The curves verify full-run convergence, not batch-wise identity."
        ),
        "runs": run_inputs,
        "epochs_per_framework": {
            framework: len({row["epoch_zero_based"] for row in all_rows if row["framework"] == framework})
            for framework in sorted({row["framework"] for row in all_rows})
        },
        "artifacts": {
            "samples_csv": "training_samples.csv",
            "epoch_csv": "epoch_loss.csv",
            "curve_png": "loss_curve.png",
            "curve_pdf": "loss_curve.pdf",
        },
    }
    (args.output_dir / "training_log_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

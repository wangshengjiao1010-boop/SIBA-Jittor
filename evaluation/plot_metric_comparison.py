#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = ("VIF", "SCD", "MI", "Qabf", "SSIM", "MS_SSIM", "FMI")
EXPERIMENTS = (
    ("Official PyTorch", "OfficialPyTorch"),
    ("Official Jittor", "OfficialJittor"),
    ("Self-trained PyTorch", "PyTorchSelfTrained"),
    ("Self-trained Jittor", "JittorSelfTrained"),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.summary.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    indexed = {(row["experiment"], row["dataset"]): row for row in rows}
    paper = json.loads(args.paper.read_text(encoding="utf-8"))["datasets"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    colors = ("#C75B39", "#2F6B9A", "#E0A458", "#5B8C5A")
    for dataset in sorted(paper):
        reference = np.array([paper[dataset][metric] for metric in METRICS], dtype=float)
        x = np.arange(len(METRICS))
        width = 0.19
        figure, axis = plt.subplots(figsize=(12, 5.5))
        for index, ((label, experiment), color) in enumerate(zip(EXPERIMENTS, colors)):
            values = np.array(
                [float(indexed[(experiment, dataset)][metric]) for metric in METRICS],
                dtype=float,
            )
            axis.bar(
                x + (index - 1.5) * width,
                values / reference,
                width,
                label=label,
                color=color,
            )
        axis.axhline(1.0, color="#222222", linewidth=1.1, linestyle="--", label="Paper")
        axis.set_xticks(x, METRICS)
        axis.set_ylabel("Metric value / paper value")
        axis.set_title(f"SIBA metric comparison on {dataset}")
        axis.grid(axis="y", alpha=0.22)
        axis.legend(ncol=3, frameon=False)
        figure.tight_layout()
        figure.savefig(args.output_dir / f"{dataset}_metric_ratio.png", dpi=220)
        figure.savefig(args.output_dir / f"{dataset}_metric_ratio.pdf")
        plt.close(figure)


if __name__ == "__main__":
    main()

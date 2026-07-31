#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LOSSES = (
    ("loss_total", "Total loss"),
    ("loss_joint_grad", "Joint-gradient loss"),
    ("loss_intensity", "Intensity loss"),
    ("loss_sobel", "Sobel-gradient loss"),
)
COLORS = {"PyTorch": "#3572A5", "Jittor": "#D55E00"}


def read_epoch_means(path):
    values = defaultdict(lambda: defaultdict(list))
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            epoch = int(row["epoch"])
            for key, _ in LOSSES:
                values[epoch][key].append(float(row[key]))
    return {
        epoch: {key: float(np.mean(items)) for key, items in losses.items()}
        for epoch, losses in sorted(values.items())
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=CSV")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    args = parser.parse_args()

    runs = {}
    for value in args.run:
        label, raw_path = value.split("=", 1)
        runs[label] = read_epoch_means(Path(raw_path))
    if len({tuple(values) for values in runs.values()}) != 1:
        raise ValueError("All runs must contain the same epochs")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["framework", "epoch"] + [key for key, _ in LOSSES]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for label, values in runs.items():
            for epoch, losses in values.items():
                writer.writerow({"framework": label, "epoch": epoch, **losses})

    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.0), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, LOSSES):
        for label, values in runs.items():
            epochs = np.asarray(list(values), dtype=np.int32) + 1
            means = np.asarray([values[epoch][key] for epoch in values])
            axis.plot(
                epochs,
                means,
                linewidth=2.0,
                label=label,
                color=COLORS.get(label),
            )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Mean loss")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    figure.savefig(args.output, dpi=240, facecolor="white")
    figure.savefig(args.output.with_suffix(".pdf"), facecolor="white")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


METRICS = ("VIF", "SCD", "MI", "Qabf", "SSIM", "MS_SSIM", "FMI")
COMPARISONS = (
    ("released_checkpoint", "OfficialJittor", "OfficialPyTorch"),
    ("self_trained", "JittorSelfTrained", "PyTorchSelfTrained"),
    ("controlled_training", "ControlledJittor", "ControlledPyTorch"),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    with args.summary.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    indexed = {(row["experiment"], row["dataset"]): row for row in rows}
    datasets = sorted({row["dataset"] for row in rows})
    output_rows = []
    maxima = {}
    for label, jittor_name, pytorch_name in COMPARISONS:
        maxima[label] = {}
        for dataset in datasets:
            if (jittor_name, dataset) not in indexed or (
                pytorch_name,
                dataset,
            ) not in indexed:
                continue
            jittor = indexed[(jittor_name, dataset)]
            pytorch = indexed[(pytorch_name, dataset)]
            output = {"comparison": label, "dataset": dataset}
            for metric in METRICS:
                difference = float(jittor[metric]) - float(pytorch[metric])
                output[f"{metric}_jittor"] = jittor[metric]
                output[f"{metric}_pytorch"] = pytorch[metric]
                output[f"{metric}_delta"] = difference
                maxima[label][metric] = max(maxima[label].get(metric, 0.0), abs(difference))
            output_rows.append(output)
        if not maxima[label]:
            del maxima[label]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not output_rows:
        raise ValueError("No complete framework comparison was found")
    with args.output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    args.report.write_text(json.dumps(maxima, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(maxima, indent=2))


if __name__ == "__main__":
    main()

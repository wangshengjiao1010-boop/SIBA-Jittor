#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


DATASETS = ("MSRS", "M3FD_2x", "TNO")
FRAMEWORKS = ("jittor", "pytorch")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synchronized-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--gpu-monitor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for framework in FRAMEWORKS:
        for dataset in DATASETS:
            synchronized = read_json(args.synchronized_root / framework / dataset / "summary.json")
            official = read_json(args.official_root / framework / dataset / "summary.json")
            rows.append(
                {
                    "framework": framework,
                    "dataset": dataset,
                    "image_count": synchronized["image_count"],
                    "synchronized_mean_ms": synchronized["mean_model_seconds"] * 1000,
                    "synchronized_fps": synchronized["synchronized_fps"],
                    "official_unsynchronized_mean_ms": official["mean_model_seconds"] * 1000,
                    "official_unsynchronized_fps": official["official_unsynchronized_fps"],
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timing_path = args.output_dir / "inference_timing.csv"
    with timing_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    monitor_rows = []
    with args.gpu_monitor.open("r", newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            if len(row) >= 4:
                monitor_rows.append(
                    {
                        "timestamp": row[0].strip(),
                        "gpu_utilization_percent": float(row[1]),
                        "memory_used_mib": float(row[2]),
                        "power_watts": float(row[3]),
                    }
                )
    monitor_summary = {
        "samples": len(monitor_rows),
        "start": monitor_rows[0]["timestamp"],
        "end": monitor_rows[-1]["timestamp"],
        "maximum_gpu_utilization_percent": max(row["gpu_utilization_percent"] for row in monitor_rows),
        "maximum_memory_used_mib": max(row["memory_used_mib"] for row in monitor_rows),
        "maximum_power_watts": max(row["power_watts"] for row in monitor_rows),
    }
    monitor_path = args.output_dir / "gpu_monitor_summary.json"
    monitor_path.write_text(json.dumps(monitor_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"timing": str(timing_path.resolve()), "gpu": monitor_summary}, indent=2))


if __name__ == "__main__":
    main()

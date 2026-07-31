#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


FIELDS = {
    "memory_used_mib": "peak_memory_used_mib",
    "utilization_gpu_percent": "peak_gpu_utilization_percent",
    "power_draw_w": "peak_power_draw_w",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("GPU monitor log contains no samples")

    summary = {
        "samples": len(rows),
        "first_timestamp": rows[0]["timestamp"],
        "last_timestamp": rows[-1]["timestamp"],
        "gpu_name": rows[0]["gpu_name"].strip(),
        "memory_total_mib": float(rows[0]["memory_total_mib"]),
    }
    for source, target in FIELDS.items():
        summary[target] = max(float(row[source]) for row in rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

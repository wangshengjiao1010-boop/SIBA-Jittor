#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paper = json.loads(args.paper.read_text(encoding="utf-8"))
    with args.summary.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    output_rows = []
    for row in rows:
        dataset = row["dataset"]
        reference = paper["datasets"][dataset]
        output = {"experiment": row["experiment"], "dataset": dataset}
        for metric in paper["metrics"]:
            reproduced = float(row[metric])
            expected = float(reference[metric])
            output[f"{metric}_reproduced"] = reproduced
            output[f"{metric}_paper"] = expected
            output[f"{metric}_delta"] = reproduced - expected
        output_rows.append(output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0].keys())
    with args.output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

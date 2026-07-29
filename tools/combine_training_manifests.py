#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--msrs", type=Path, required=True)
    parser.add_argument("--roadscene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = []
    for label, path in (("MSRS_train", args.msrs), ("RoadScene_selected_200", args.roadscene)):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for pair in manifest["pairs"]:
            sources.append({"source": label, **pair})
    names = [pair["name"] for pair in sources]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise RuntimeError(f"Filename collisions: {duplicates}")

    report = {
        "pair_count": len(sources),
        "msrs_pairs": sum(pair["source"] == "MSRS_train" for pair in sources),
        "roadscene_pairs": sum(pair["source"] == "RoadScene_selected_200" for pair in sources),
        "pairs": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()

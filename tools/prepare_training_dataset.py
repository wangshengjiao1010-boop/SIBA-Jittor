#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def files(directory):
    return {
        path.name: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def link(source, target):
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        target.symlink_to(source)


def add_pairs(records, source_name, infrared, visible, names):
    infrared_files = files(infrared)
    visible_files = files(visible)
    for name in names:
        if name not in infrared_files or name not in visible_files:
            raise RuntimeError(f"Missing {source_name} pair: {name}")
        records.append(
            {
                "source": source_name,
                "name": name,
                "infrared": str(infrared_files[name].resolve()),
                "visible": str(visible_files[name].resolve()),
            }
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--msrs-infrared", type=Path, required=True)
    parser.add_argument("--msrs-visible", type=Path, required=True)
    parser.add_argument("--roadscene-infrared", type=Path, required=True)
    parser.add_argument("--roadscene-visible", type=Path, required=True)
    parser.add_argument("--roadscene-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    roadscene_manifest = json.loads(args.roadscene_manifest.read_text(encoding="utf-8"))
    roadscene_names = [pair["name"] for pair in roadscene_manifest["pairs"]]
    msrs_names = sorted(files(args.msrs_infrared))

    records = []
    add_pairs(
        records,
        "MSRS_train",
        args.msrs_infrared,
        args.msrs_visible,
        msrs_names,
    )
    add_pairs(
        records,
        "RoadScene_selected_200",
        args.roadscene_infrared,
        args.roadscene_visible,
        roadscene_names,
    )

    names = [record["name"] for record in records]
    if len(names) != len(set(names)):
        duplicates = sorted(name for name in set(names) if names.count(name) > 1)
        raise RuntimeError(f"Filename collisions between datasets: {duplicates}")

    infrared_output = args.output / "ir"
    visible_output = args.output / "vi"
    infrared_output.mkdir(parents=True, exist_ok=True)
    visible_output.mkdir(parents=True, exist_ok=True)
    for record in records:
        link(Path(record["infrared"]), infrared_output / record["name"])
        link(Path(record["visible"]), visible_output / record["name"])

    combined = {
        "pair_count": len(records),
        "msrs_pairs": len(msrs_names),
        "roadscene_pairs": len(roadscene_names),
        "infrared_directory": str(infrared_output.resolve()),
        "visible_directory": str(visible_output.resolve()),
        "pairs": records,
    }
    (args.output / "combined_manifest.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: combined[key] for key in combined if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()


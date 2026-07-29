#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def images(directory):
    return {
        path.name: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def matched_pairs(infrared, visible):
    infrared_files = images(infrared)
    visible_files = images(visible)
    if set(infrared_files) != set(visible_files):
        raise RuntimeError(
            json.dumps(
                {
                    "missing_visible": sorted(set(infrared_files) - set(visible_files)),
                    "missing_infrared": sorted(set(visible_files) - set(infrared_files)),
                }
            )
        )
    return [(name, infrared_files[name], visible_files[name]) for name in sorted(infrared_files)]


def link(source, target):
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        target.symlink_to(source)


def prepare_linked(name, infrared, visible, output):
    pairs = matched_pairs(infrared, visible)
    infrared_output = output / name / "ir"
    visible_output = output / name / "vi"
    infrared_output.mkdir(parents=True, exist_ok=True)
    visible_output.mkdir(parents=True, exist_ok=True)
    for filename, infrared_path, visible_path in pairs:
        with Image.open(infrared_path) as infrared_image, Image.open(
            visible_path
        ) as visible_image:
            if infrared_image.size != visible_image.size:
                raise RuntimeError(
                    f"Size mismatch for {name}/{filename}: "
                    f"infrared={infrared_image.size}, visible={visible_image.size}"
                )
        link(infrared_path, infrared_output / filename)
        link(visible_path, visible_output / filename)
    return len(pairs)


def prepare_m3fd(infrared, visible, output):
    pairs = matched_pairs(infrared, visible)
    infrared_output = output / "M3FD_2x" / "ir"
    visible_output = output / "M3FD_2x" / "vi"
    infrared_output.mkdir(parents=True, exist_ok=True)
    visible_output.mkdir(parents=True, exist_ok=True)
    for filename, infrared_path, visible_path in pairs:
        with Image.open(infrared_path) as infrared_image, Image.open(
            visible_path
        ) as visible_image:
            if infrared_image.size != visible_image.size:
                raise RuntimeError(
                    f"Size mismatch for M3FD/{filename}: "
                    f"infrared={infrared_image.size}, visible={visible_image.size}"
                )
            size = (infrared_image.size[0] // 2, infrared_image.size[1] // 2)
            infrared_image.resize(size, resample=Image.LANCZOS).save(
                infrared_output / filename
            )
            visible_image.resize(size, resample=Image.LANCZOS).save(
                visible_output / filename
            )
    return len(pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--msrs-infrared", type=Path, required=True)
    parser.add_argument("--msrs-visible", type=Path, required=True)
    parser.add_argument("--m3fd-infrared", type=Path, required=True)
    parser.add_argument("--m3fd-visible", type=Path, required=True)
    parser.add_argument("--tno-infrared", type=Path, required=True)
    parser.add_argument("--tno-visible", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "MSRS": prepare_linked(
            "MSRS", args.msrs_infrared, args.msrs_visible, args.output
        ),
        "M3FD_2x": prepare_m3fd(
            args.m3fd_infrared, args.m3fd_visible, args.output
        ),
        "TNO": prepare_linked(
            "TNO", args.tno_infrared, args.tno_visible, args.output
        ),
    }
    (args.output / "test_dataset_counts.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

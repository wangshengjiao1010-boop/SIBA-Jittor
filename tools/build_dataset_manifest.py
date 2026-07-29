#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_files(directory):
    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.name in files:
                raise RuntimeError(f"Duplicate filename: {path.name}")
            files[path.name] = path
    return files


def inspect_pair(name, infrared, visible):
    with Image.open(infrared) as infrared_image:
        infrared_size = infrared_image.size
        infrared_mode = infrared_image.mode
        infrared_image.verify()
    with Image.open(visible) as visible_image:
        visible_size = visible_image.size
        visible_mode = visible_image.mode
        visible_image.verify()
    if infrared_size != visible_size:
        raise RuntimeError(
            f"Size mismatch for {name}: infrared={infrared_size}, visible={visible_size}"
        )
    return {
        "name": name,
        "width": infrared_size[0],
        "height": infrared_size[1],
        "infrared_mode": infrared_mode,
        "visible_mode": visible_mode,
        "infrared_sha256": sha256(infrared),
        "visible_sha256": sha256(visible),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infrared", type=Path, required=True)
    parser.add_argument("--visible", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--select", type=int)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    infrared = image_files(args.infrared)
    visible = image_files(args.visible)
    missing_visible = sorted(set(infrared) - set(visible))
    missing_infrared = sorted(set(visible) - set(infrared))
    if missing_visible or missing_infrared:
        raise RuntimeError(
            json.dumps(
                {
                    "missing_visible": missing_visible,
                    "missing_infrared": missing_infrared,
                },
                ensure_ascii=False,
            )
        )

    names = sorted(infrared)
    if args.select is not None:
        if args.select > len(names):
            raise RuntimeError(f"Cannot select {args.select} pairs from {len(names)}")
        names = sorted(random.Random(args.seed).sample(names, args.select))

    pairs = [inspect_pair(name, infrared[name], visible[name]) for name in names]
    report = {
        "infrared_directory": str(args.infrared.resolve()),
        "visible_directory": str(args.visible.resolve()),
        "available_pairs": len(infrared),
        "selected_pairs": len(pairs),
        "selection_seed": args.seed if args.select is not None else None,
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()


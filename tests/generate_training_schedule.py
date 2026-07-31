#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


SUFFIXES = {".png", ".jpg", ".bmp"}


def image_files(path):
    return [item for item in sorted(path.glob("*")) if item.suffix in SUFFIXES]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Create a shared sample-order and crop schedule for both frameworks."
    )
    parser.add_argument("--ir-path", type=Path, required=True)
    parser.add_argument("--vi-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    infrared = image_files(args.ir_path)
    visible = image_files(args.vi_path)
    if not infrared or len(infrared) != len(visible):
        raise ValueError(
            f"Invalid training counts: ir={len(infrared)}, vi={len(visible)}"
        )
    if [path.name for path in infrared] != [path.name for path in visible]:
        raise ValueError("Infrared and visible filenames do not match")

    dimensions = []
    for ir_path, vi_path in zip(infrared, visible):
        ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
        vi = cv2.imread(str(vi_path), cv2.IMREAD_GRAYSCALE)
        if ir is None or vi is None or ir.shape != vi.shape:
            raise ValueError(f"Invalid training pair: {ir_path.name}")
        height, width = ir.shape
        if height - 20 < args.patch_size or width - 20 < args.patch_size:
            raise ValueError(
                f"Image is too small for the official crop margin: {ir_path.name}"
            )
        dimensions.append((height, width))

    random_state = np.random.RandomState(args.seed)
    sample_count = len(infrared)
    indices = np.empty((args.epochs, sample_count), dtype=np.int32)
    crop_x = np.empty_like(indices)
    crop_y = np.empty_like(indices)
    for epoch in range(args.epochs):
        order = random_state.permutation(sample_count)
        indices[epoch] = order
        for position, sample_index in enumerate(order):
            height, width = dimensions[int(sample_index)]
            crop_x[epoch, position] = random_state.randint(
                10, height - 10 - args.patch_size + 1
            )
            crop_y[epoch, position] = random_state.randint(
                10, width - 10 - args.patch_size + 1
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        indices=indices,
        crop_x=crop_x,
        crop_y=crop_y,
        filenames=np.asarray([path.name for path in infrared]),
    )
    metadata_path = args.metadata or args.output.with_suffix(".json")
    metadata = {
        "purpose": "Shared data order and random crop coordinates for framework comparison",
        "seed": args.seed,
        "epochs": args.epochs,
        "training_pairs": sample_count,
        "patch_size": args.patch_size,
        "ir_path": str(args.ir_path.resolve()),
        "vi_path": str(args.vi_path.resolve()),
        "schedule": str(args.output.resolve()),
        "schedule_sha256": file_sha256(args.output),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

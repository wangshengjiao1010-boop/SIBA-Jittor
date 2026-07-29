#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def image_files(directory):
    return {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference_files = image_files(args.reference)
    candidate_files = image_files(args.candidate)
    reference_only = sorted(reference_files.keys() - candidate_files.keys())
    candidate_only = sorted(candidate_files.keys() - reference_files.keys())
    shared = sorted(reference_files.keys() & candidate_files.keys())
    if not shared:
        raise RuntimeError("No shared image filenames")

    rows = []
    absolute_sum = 0.0
    value_count = 0
    differing_values = 0
    global_max = 0
    shape_mismatches = []
    for name in shared:
        reference = np.asarray(Image.open(reference_files[name]).convert("RGB"), dtype=np.int16)
        candidate = np.asarray(Image.open(candidate_files[name]).convert("RGB"), dtype=np.int16)
        if reference.shape != candidate.shape:
            shape_mismatches.append(
                {"name": name, "reference_shape": list(reference.shape), "candidate_shape": list(candidate.shape)}
            )
            continue
        difference = np.abs(reference - candidate)
        squared = np.square(reference.astype(np.float64) - candidate.astype(np.float64))
        mse = float(squared.mean())
        psnr = float("inf") if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
        row = {
            "name": name,
            "height": reference.shape[0],
            "width": reference.shape[1],
            "max_abs_uint8": int(difference.max(initial=0)),
            "mean_abs_uint8": float(difference.mean()),
            "differing_values": int(np.count_nonzero(difference)),
            "value_count": int(difference.size),
            "psnr_db": psnr,
        }
        rows.append(row)
        absolute_sum += float(difference.sum())
        value_count += int(difference.size)
        differing_values += int(np.count_nonzero(difference))
        global_max = max(global_max, row["max_abs_uint8"])

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "reference_images": len(reference_files),
        "candidate_images": len(candidate_files),
        "compared_images": len(rows),
        "reference_only": reference_only,
        "candidate_only": candidate_only,
        "shape_mismatches": shape_mismatches,
        "global_max_abs_uint8": global_max,
        "global_mean_abs_uint8": absolute_sum / value_count,
        "different_value_fraction": differing_values / value_count,
        "all_filenames_and_shapes_match": not reference_only and not candidate_only and not shape_mismatches,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()

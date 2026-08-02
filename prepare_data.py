#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "datasets" / "source"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "SIBA"


def image_files(directory):
    files = {
        path.name: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    if not files:
        raise RuntimeError(f"No images found in {directory}")
    return files


def matched_pairs(infrared, visible):
    infrared_files = image_files(infrared)
    visible_files = image_files(visible)
    if set(infrared_files) != set(visible_files):
        raise RuntimeError(
            json.dumps(
                {
                    "missing_visible": sorted(set(infrared_files) - set(visible_files)),
                    "missing_infrared": sorted(
                        set(visible_files) - set(infrared_files)
                    ),
                },
                ensure_ascii=False,
            )
        )
    return [
        (name, infrared_files[name], visible_files[name])
        for name in sorted(infrared_files)
    ]


def link(source, target):
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        target.symlink_to(source.resolve())


def materialize_pairs(pairs, output):
    infrared_output = output / "ir"
    visible_output = output / "vi"
    infrared_output.mkdir(parents=True, exist_ok=True)
    visible_output.mkdir(parents=True, exist_ok=True)
    for name, infrared, visible in pairs:
        link(infrared, infrared_output / name)
        link(visible, visible_output / name)

    expected = {name for name, _, _ in pairs}
    actual_infrared = set(image_files(infrared_output))
    actual_visible = set(image_files(visible_output))
    if actual_infrared != expected or actual_visible != expected:
        raise RuntimeError(f"Unexpected files under {output}")


def prepare_training(args):
    msrs_pairs = matched_pairs(args.msrs_root / "train/ir", args.msrs_root / "train/vi")
    roadscene_pairs = matched_pairs(
        args.roadscene_root / "cropinfrared",
        args.roadscene_root / "crop_LR_visible",
    )
    manifest = json.loads(args.roadscene_manifest.read_text(encoding="utf-8"))
    roadscene_by_name = {
        name: (infrared, visible) for name, infrared, visible in roadscene_pairs
    }
    selected_names = [pair["name"] for pair in manifest["pairs"]]
    selected_roadscene = [
        (name, *roadscene_by_name[name])
        for name in selected_names
        if name in roadscene_by_name
    ]
    if len(selected_roadscene) != 200:
        raise RuntimeError(
            f"Expected 200 RoadScene pairs, found {len(selected_roadscene)}"
        )

    names = [name for name, _, _ in msrs_pairs + selected_roadscene]
    if len(names) != len(set(names)):
        raise RuntimeError("MSRS and RoadScene contain colliding filenames")
    if len(msrs_pairs) != 1083:
        raise RuntimeError(
            f"Expected 1083 MSRS training pairs, found {len(msrs_pairs)}"
        )

    materialize_pairs(msrs_pairs + selected_roadscene, args.output / "train")
    return len(msrs_pairs) + len(selected_roadscene)


def prepare_linked_test(name, infrared, visible, output, expected_count):
    pairs = matched_pairs(infrared, visible)
    if len(pairs) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} {name} pairs, found {len(pairs)}"
        )
    for filename, infrared_path, visible_path in pairs:
        with Image.open(infrared_path) as infrared_image, Image.open(
            visible_path
        ) as visible_image:
            if infrared_image.size != visible_image.size:
                raise RuntimeError(f"Size mismatch for {name}/{filename}")
    materialize_pairs(pairs, output / "test" / name)
    return len(pairs)


def prepare_m3fd(args):
    pairs = matched_pairs(args.m3fd_root / "Ir", args.m3fd_root / "Vis")
    if len(pairs) != 300:
        raise RuntimeError(f"Expected 300 M3FD pairs, found {len(pairs)}")
    output = args.output / "test/M3FD_2x"
    infrared_output = output / "ir"
    visible_output = output / "vi"
    infrared_output.mkdir(parents=True, exist_ok=True)
    visible_output.mkdir(parents=True, exist_ok=True)
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    for name, infrared_path, visible_path in pairs:
        with Image.open(infrared_path) as infrared, Image.open(visible_path) as visible:
            if infrared.size != visible.size:
                raise RuntimeError(f"Size mismatch for M3FD/{name}")
            size = (infrared.size[0] // 2, infrared.size[1] // 2)
            infrared.resize(size, resample=resampling).save(infrared_output / name)
            visible.resize(size, resample=resampling).save(visible_output / name)
    expected = {name for name, _, _ in pairs}
    if (
        set(image_files(infrared_output)) != expected
        or set(image_files(visible_output)) != expected
    ):
        raise RuntimeError(f"Unexpected files under {output}")
    return len(pairs)


def main():
    parser = argparse.ArgumentParser(description="Prepare the complete SIBA datasets")
    parser.add_argument(
        "--msrs-root", type=Path, default=DEFAULT_SOURCE_ROOT / "MSRS"
    )
    parser.add_argument(
        "--roadscene-root", type=Path, default=DEFAULT_SOURCE_ROOT / "RoadScene"
    )
    parser.add_argument(
        "--m3fd-root", type=Path, default=DEFAULT_SOURCE_ROOT / "M3FD_Fusion"
    )
    parser.add_argument(
        "--tno-root", type=Path, default=DEFAULT_SOURCE_ROOT / "TNO"
    )
    parser.add_argument(
        "--roadscene-manifest",
        type=Path,
        default=PROJECT_ROOT / "data/manifests/roadscene_200_seed2025.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()

    report = {
        "train": prepare_training(args),
        "MSRS": prepare_linked_test(
            "MSRS",
            args.msrs_root / "test/ir",
            args.msrs_root / "test/vi",
            args.output,
            361,
        ),
        "M3FD_2x": prepare_m3fd(args),
        "TNO": prepare_linked_test(
            "TNO",
            args.tno_root / "ir",
            args.tno_root / "vi",
            args.output,
            45,
        ),
    }
    (args.output / "dataset_counts.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

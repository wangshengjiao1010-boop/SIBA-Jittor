#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="LABEL=MANIFEST,IR_DIR,VI_DIR",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {}
    for value in args.dataset:
        label, paths = value.split("=", 1)
        manifest_path, infrared_dir, visible_dir = map(Path, paths.split(",", 2))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pairs = {pair["name"]: pair for pair in manifest["pairs"]}
        errors = []
        for name, pair in pairs.items():
            for directory, key in (
                (infrared_dir, "infrared_sha256"),
                (visible_dir, "visible_sha256"),
            ):
                path = directory / name
                if not path.exists():
                    errors.append({"name": name, "error": f"missing {path}"})
                elif sha256(path) != pair[key]:
                    errors.append({"name": name, "error": f"hash mismatch {path}"})
        report[label] = {
            "manifest": str(manifest_path.resolve()),
            "pairs": len(pairs),
            "errors": errors,
            "valid": not errors,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all(item["valid"] for item in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

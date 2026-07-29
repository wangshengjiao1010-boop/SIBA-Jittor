#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_PATTERNS = (
    "official_pytorch/**/*.py",
    "siba_jittor/**/*.py",
    "configs/**/*",
    "data_manifests/*.json",
    "logs/**/*.log",
    "logs/**/*.csv",
    "logs/**/*.json",
    "checkpoints/**/*.pth",
    "checkpoints/**/*.pkl",
    "results/**/*.csv",
    "results/**/*.json",
    "results/**/*.png",
    "results/**/*.pdf",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-sha256", type=Path, required=True)
    args = parser.parse_args()

    project = args.project_root.resolve()
    files = set()
    for pattern in DEFAULT_PATTERNS:
        files.update(path for path in project.glob(pattern) if path.is_file())

    records = []
    for path in sorted(files):
        records.append(
            {
                "path": path.relative_to(project).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    report = {
        "purpose": (
            "Integrity manifest for retained source, logs, checkpoints, metric tables, "
            "plots, and visual comparisons. It detects later file changes but does not "
            "by itself prove how an artifact was produced."
        ),
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "files": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_sha256.parent.mkdir(parents=True, exist_ok=True)
    args.output_sha256.write_text(
        "".join(
            f"{record['sha256']}  {record['path']}\n" for record in records
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "file_count": report["file_count"],
                "total_bytes": report["total_bytes"],
                "output_json": str(args.output_json),
                "output_sha256": str(args.output_sha256),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

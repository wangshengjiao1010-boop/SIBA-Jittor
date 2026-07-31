#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require_equal(label, values):
    if len(set(values)) != 1:
        raise AssertionError(f"{label} differs: {values}")
    return values[0]


def validate_run(label, metadata, expected, schedule_sha256, archive_sha256):
    checks = {
        "epochs": expected["epochs"],
        "batch_size": expected["batch_size"],
        "patch_size": expected["patch_size"],
        "training_pairs": expected["training_pairs"],
    }
    for key, value in checks.items():
        if metadata.get(key) != value:
            raise AssertionError(
                f"{label} {key}: expected {value}, got {metadata.get(key)}"
            )
    if metadata.get("schedule_sha256") != schedule_sha256:
        raise AssertionError(f"{label} schedule hash does not match the shared schedule")
    if metadata.get("initial_weights_sha256") != archive_sha256:
        raise AssertionError(f"{label} initialization archive hash does not match")
    expected_batches = expected["epochs"] * metadata["batches_per_epoch"]
    if metadata.get("logged_batches") != expected_batches:
        raise AssertionError(
            f"{label} logged {metadata.get('logged_batches')} batches; "
            f"expected {expected_batches}"
        )
    checkpoint = Path(metadata["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if file_sha256(checkpoint) != metadata.get("checkpoint_sha256"):
        raise AssertionError(f"{label} checkpoint SHA256 does not match its metadata")
    batch_log = Path(metadata["batch_log"])
    if not batch_log.is_file():
        raise FileNotFoundError(batch_log)
    with batch_log.open("r", encoding="utf-8-sig", newline="") as file:
        rows = sum(1 for _ in csv.DictReader(file))
    if rows != expected_batches:
        raise AssertionError(
            f"{label} batch CSV has {rows} rows; expected {expected_batches}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Verify shared initialization, data schedule, and complete logs."
    )
    parser.add_argument("--initial-metadata", type=Path, required=True)
    parser.add_argument("--schedule-metadata", type=Path, required=True)
    parser.add_argument("--jittor-metadata", type=Path, required=True)
    parser.add_argument("--pytorch-metadata", type=Path)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--training-pairs", type=int, default=1283)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    initial = read_json(args.initial_metadata)
    schedule = read_json(args.schedule_metadata)
    jittor = read_json(args.jittor_metadata)
    pytorch = read_json(args.pytorch_metadata) if args.pytorch_metadata else None

    schedule_path = Path(schedule["schedule"])
    archive_path = Path(initial["framework_neutral_archive"])
    schedule_sha256 = file_sha256(schedule_path)
    archive_sha256 = file_sha256(archive_path)
    require_equal(
        "schedule SHA256",
        [schedule["schedule_sha256"], schedule_sha256, jittor["schedule_sha256"]],
    )
    require_equal(
        "initialization archive SHA256",
        [
            initial["framework_neutral_archive_sha256"],
            archive_sha256,
            jittor["initial_weights_sha256"],
        ],
    )
    state_values = [initial["state_sha256"], jittor["initial_state_sha256"]]
    if pytorch is not None:
        state_values.append(pytorch["initial_state_sha256"])
    shared_state_sha256 = require_equal("initial model state SHA256", state_values)

    expected = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "training_pairs": args.training_pairs,
    }
    validate_run("Jittor", jittor, expected, schedule_sha256, archive_sha256)
    if pytorch is not None:
        validate_run("PyTorch", pytorch, expected, schedule_sha256, archive_sha256)

    report = {
        "passed": True,
        "shared_initial_state_sha256": shared_state_sha256,
        "initialization_archive_sha256": archive_sha256,
        "schedule_sha256": schedule_sha256,
        **expected,
        "frameworks": ["Jittor"] if pytorch is None else ["PyTorch", "Jittor"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

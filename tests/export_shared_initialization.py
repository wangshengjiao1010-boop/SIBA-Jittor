#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Export one PyTorch initialization for both training frameworks."
    )
    parser.add_argument("--pytorch-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--name", default="initial")
    args = parser.parse_args()

    pytorch_root = args.pytorch_root.resolve()
    sys.path.insert(0, str(pytorch_root))
    from models.SIBA import SIBA

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = SIBA()
    state = model.state_dict()
    arrays = {
        name: tensor.detach().cpu().numpy().copy()
        for name, tensor in state.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name
    pytorch_path = args.output_dir / f"{stem}.pth"
    numpy_path = args.output_dir / f"{stem}.npz"
    metadata_path = args.output_dir / f"{stem}.json"
    torch.save({"model": state}, pytorch_path)
    np.savez(numpy_path, **arrays)

    metadata = {
        "purpose": "Shared initialization for complete PyTorch/Jittor training comparison",
        "official_pytorch_root": str(pytorch_root),
        "seed": args.seed,
        "torch_version": torch.__version__,
        "state_tensors": len(state),
        "trainable_parameters": sum(value.numel() for value in model.parameters()),
        "state_sha256": state_sha256(state),
        "pytorch_checkpoint": str(pytorch_path.resolve()),
        "pytorch_checkpoint_sha256": file_sha256(pytorch_path),
        "framework_neutral_archive": str(numpy_path.resolve()),
        "framework_neutral_archive_sha256": file_sha256(numpy_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

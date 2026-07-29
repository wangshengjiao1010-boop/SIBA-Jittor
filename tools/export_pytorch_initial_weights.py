#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Export one deterministic PyTorch SIBA initialization for paired demonstrations."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    source = project_root / "official_pytorch"
    sys.path.insert(0, str(source))
    from models.SIBA import SIBA

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = SIBA()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, args.output)

    metadata_path = args.metadata or args.output.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "purpose": "Shared PyTorch initialization for live PyTorch/Jittor training demonstration only",
        "used_for_final_metrics": False,
        "framework": "PyTorch",
        "torch_version": torch.__version__,
        "seed": args.seed,
        "parameter_tensors": len(model.state_dict()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "output": str(args.output.resolve()),
        "sha256": sha256(args.output),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

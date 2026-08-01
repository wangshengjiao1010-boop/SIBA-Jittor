import argparse
import re
import shlex
from pathlib import Path, PurePosixPath

import yaml


def resolved(project_root, value):
    value = str(value)
    if value.startswith("/"):
        return PurePosixPath(value)
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    with args.config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    experiment_name = str(config["experiment"]["name"])
    if not re.fullmatch(r"[A-Za-z0-9._-]+", experiment_name):
        raise ValueError(
            "experiment.name may contain only letters, digits, '.', '_' and '-'"
        )

    paths = config["paths"]
    protocol = config["protocol"]
    runtime = config["runtime"]
    output = config["output"]
    values = {
        "DATA_ROOT": resolved(project_root, paths["data_root"]),
        "PYTORCH_ROOT": resolved(project_root, paths["pytorch_root"]),
        "JITTOR_PYTHON": resolved(project_root, paths["jittor_python"]),
        "PYTORCH_PYTHON": resolved(project_root, paths["pytorch_python"]),
        "EXPERIMENT_NAME": experiment_name,
        "SEED": int(config["experiment"]["seed"]),
        "EPOCHS": int(protocol["epochs"]),
        "BATCH_SIZE": int(protocol["batch_size"]),
        "PATCH_SIZE": int(protocol["patch_size"]),
        "TRAINING_PAIRS": int(protocol["training_pairs"]),
        "GPU_NUMBER": int(runtime["gpu_number"]),
        "GPU_MONITOR_INTERVAL": int(runtime["gpu_monitor_interval"]),
        "RUN_ROOT": resolved(project_root, output["logs"]) / experiment_name,
        "CHECKPOINT_ROOT": resolved(project_root, output["checkpoints"])
        / experiment_name,
        "RESULT_ROOT": resolved(project_root, output["results"]) / experiment_name,
        "SHARED_ROOT": resolved(project_root, output["shared"]) / experiment_name,
    }
    for name, value in values.items():
        print(f"{name}={shlex.quote(str(value))}")


if __name__ == "__main__":
    main()

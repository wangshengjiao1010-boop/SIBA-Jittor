# Controlled PyTorch/Jittor Comparison

This workflow is an optional migration audit. It is not required to train or test the Jittor model. Ordinary use follows the repository-relative dataset layout and runs `python train.py` or `python test.py` directly.

## Why a shell script is retained

`scripts/run_shared_comparison_screen.sh` coordinates two Conda environments, exports one shared PyTorch initialization, generates one shared 60-epoch sample/crop schedule, trains both frameworks, runs all 706 test pairs, compares fused images, records GPU utilization, and writes completion evidence. A shell script is appropriate here because this is a multi-process experiment rather than a model implementation.

## Output names

`experiment.name` in `configs/comparison.yaml` groups files from one controlled run and prevents a later run from overwriting earlier evidence:

```text
logs/comparisons/<experiment.name>/          console logs and metadata
checkpoint/comparisons/<experiment.name>/   PyTorch and Jittor weights
results/comparisons/<experiment.name>/       curves, fused images and alignment reports
shared/comparisons/<experiment.name>/        shared initialization and crop schedule
```

All four directories are still ordinary directories on the AutoDL disk. The experiment name does not upload data or create another storage service.

The previously suggested expression `recording_$(date +%Y%m%d_%H%M%S)` was only a way to generate a unique name such as `recording_20260801_143025`. The percent codes are Linux `date` formatting: `%Y` year, `%m` month, `%d` day, `%H` hour, `%M` minute and `%S` second. The revised workflow does not require this command; edit `experiment.name` directly in YAML.

## Run the audit

Edit the paths, environment interpreters, protocol and experiment name in `configs/comparison.yaml`, then run:

```bash
bash scripts/run_shared_comparison_screen.sh
screen -r kk
```

The default configuration uses `experiment.name: shared_seed2025`. Successful completion is marked by `logs/comparisons/shared_seed2025/EXPERIMENT_COMPLETE`; no result from this optional workflow should be reported unless that marker and its corresponding artifacts exist.

## Completed reference run

`shared_seed2025` completed on 2026-08-01 using code revision `99a9e2c`, an RTX 3090, 1,283 training pairs, 60 epochs and 19,260 batches per framework. The same initial state and crop schedule are verified by SHA256 in `initial_inputs_verified.json` and `training_inputs_verified.json`.

The raw `code_status.txt` lists `datasets`, `official_pytorch` and the run log directory as untracked because these machine-local inputs and outputs are intentionally excluded from the code revision. The launcher separately checked `git status --porcelain --untracked-files=no`, so tracked source files were clean before training.

Published evidence:

- `logs/comparisons/shared_seed2025/`: completion marker, console logs, metadata and batch CSVs;
- `checkpoint/comparisons/shared_seed2025/`: both 60-epoch checkpoints;
- `results/comparisons/shared_seed2025/`: four loss curves, inference timing, per-image output differences and selected images;
- `shared/comparisons/shared_seed2025/`: framework-neutral initialization and deterministic crop schedule.

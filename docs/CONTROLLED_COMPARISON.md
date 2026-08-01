# Controlled PyTorch/Jittor Comparison

This workflow is an optional migration audit. It is not required to train or test the Jittor model. Ordinary use follows the official SIBA layout: edit paths in `args/args_SIBA.py` and `test.py`, then run `python train.py` or `python test.py`.

## Why a shell script is retained

`scripts/run_shared_comparison_screen.sh` coordinates two Conda environments, exports one shared PyTorch initialization, generates one shared 60-epoch sample/crop schedule, trains both frameworks, runs all 706 test pairs, compares fused images, records GPU utilization, and writes completion evidence. A shell script is appropriate here because this is a multi-process experiment rather than a model implementation.

## Output names

`experiment.name` in `configs/comparison.yaml` groups files from one controlled run and prevents a later run from overwriting earlier evidence:

```text
logs/<experiment.name>/          console logs and metadata
checkpoints/<experiment.name>/   PyTorch and Jittor weights
results/<experiment.name>/       curves, fused images and alignment reports
shared/<experiment.name>/        shared initialization and crop schedule
```

All four directories are still ordinary directories on the AutoDL disk. The experiment name does not upload data or create another storage service.

The previously suggested expression `recording_$(date +%Y%m%d_%H%M%S)` was only a way to generate a unique name such as `recording_20260801_143025`. The percent codes are Linux `date` formatting: `%Y` year, `%m` month, `%d` day, `%H` hour, `%M` minute and `%S` second. The revised workflow does not require this command; edit `experiment.name` directly in YAML.

## Run the audit

Edit the paths, environment interpreters, protocol and experiment name in `configs/comparison.yaml`, then run:

```bash
bash scripts/run_shared_comparison_screen.sh
screen -r kk
```

The default configuration uses `experiment.name: shared_seed2025`. Successful completion is marked by `logs/shared_seed2025/EXPERIMENT_COMPLETE`; no result from this optional workflow should be reported unless that marker and its corresponding artifacts exist.

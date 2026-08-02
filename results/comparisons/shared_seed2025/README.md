# Shared Seed 2025 Result Index

This directory contains the public result evidence from the completed controlled PyTorch/Jittor run. Full 706-image output batches remain on the AutoDL experiment disk; summaries, all per-image difference rows and representative samples are tracked here.

## Training

- `epoch_loss_components.csv`: per-epoch total, Laplacian, intensity and Sobel losses for both frameworks.
- `loss_components.png` and `loss_components.pdf`: four 60-epoch loss curves.
- Batch-level values and console output: `../../../logs/comparisons/shared_seed2025/`.

## Inference

Each `pytorch/<dataset>/` and `jittor/<dataset>/` directory contains synchronized `timing.csv` and `summary.json` for the complete test set.

| Dataset | Pairs | Max abs. uint8 | Mean abs. uint8 |
|---|---:|---:|---:|
| MSRS | 361 | 65 | 0.984653 |
| M3FD_2x | 300 | 108 | 3.086500 |
| TNO | 45 | 46 | 3.064897 |

`alignment/<dataset>/per_image.csv` contains every filename-level comparison. `visual_samples/` retains three high-difference examples per dataset with infrared, visible, PyTorch and Jittor images kept side by side. These samples illustrate real training-trajectory differences; they are not copies of one framework's output.

The complete seven-metric results are included in `../../metrics/metrics_summary.csv`. Per-image values are stored under `../../metrics/ControlledPyTorch/` and `../../metrics/ControlledJittor/`; framework deltas are in `../../metrics/framework_delta.csv`.

## Provenance

- Completion marker: `../../../logs/comparisons/shared_seed2025/EXPERIMENT_COMPLETE`
- Code revision: `99a9e2cfd22fccf8f40b8a5e9a8d836e3879a3b9`
- Initialization archive SHA256: `870e1e10c7a1927d23dc77a8e36ea1974f2ef66325e87342267e30e1f3533983`
- Initial state SHA256: `f58dcdaf3e64504646723d79a25a79bbb341f66e6e48239969bdf818cf6919be`
- Crop schedule SHA256: `41a68929ee9759eaee89eae8f36c8231693d55da7f677f52650b0d8a99ae02fa`

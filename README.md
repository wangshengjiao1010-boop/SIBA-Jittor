# SIBA-Jittor

Unofficial Jittor reproduction of **The Source Image is the Best Attention for Infrared and Visible Image Fusion** (ICCV 2025).

[Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_The_Source_Image_is_the_Best_Attention_for_Infrared_and_ICCV_2025_paper.html) | [Official PyTorch code](https://github.com/Afreshbird/SIBA) | [Jittor](https://github.com/Jittor/jittor)

This repository migrates the official implementation at commit [`880a1dd`](https://github.com/Afreshbird/SIBA/commit/880a1ddf9eaa610c64e5f25f87fbb146448addc9). The SIBA topology, three training losses, 60-epoch protocol and YCbCr inference procedure are retained. Framework-specific Adam and gradient clipping compatibility code is isolated in [`compat/`](compat/).

## Repository Structure

```text
SIBA-Jittor/
|-- args/                   # training configuration
|-- base_blocks/            # Res-SE, CBSM and Restormer blocks
|-- loader/                 # paired training and test loaders
|-- loss/                   # Laplacian, intensity and Sobel losses
|-- models/                 # SIBA network
|-- compat/                 # PyTorch-compatible Adam and clipping
|-- prepare_data.py         # dataset validation and preparation
|-- train.py                # Jittor training entry
|-- test.py                 # Jittor inference entry
|-- tests/                  # module and alignment checks
|-- logs/                   # environment, training and alignment logs
|-- results/                # curves, metrics and visual comparisons
`-- checkpoint/             # published and newly trained checkpoints
```

## Environment

The verified environment is Ubuntu 20.04, Python 3.8.18, CUDA 11.3 and one RTX 3090 24 GB GPU.

```bash
conda create -n JittorDome python=3.8.18 -y
conda activate JittorDome
pip install -r requirements.txt
```

The main dependencies are Jittor 1.3.11.0, NumPy 1.24.4, OpenCV 4.8.1, Pillow 10.0.1 and Matplotlib 3.7.5. Exact package records are available in [`logs/environment/`](logs/environment/) and [`logs/final/environment.txt`](logs/final/environment.txt).

If Jittor reports `Flags has no attribute cuda_archs`, rebuild its local compiled cache:

```bash
mv ~/.cache/jittor ~/.cache/jittor.stale
python -c "import jittor as jt; print(jt.__version__)"
```

## Data Preparation

Dataset images are not included in this repository. Download them from the original sources:

| Usage | Dataset | Pairs | Download |
|---|---|---:|---|
| Train | MSRS | 1,083 | [MSRS](https://github.com/Linfeng-Tang/MSRS) |
| Train | RoadScene | 200 selected pairs | [RoadScene](https://github.com/hanna-xu/RoadScene) |
| Test | MSRS | 361 | [MSRS](https://github.com/Linfeng-Tang/MSRS) |
| Test | M3FD | 300 | [M3FD](https://github.com/JinyuanLiu-CV/TarDAL#download) |
| Test | TNO | 45 | [SIBA data folder](https://drive.google.com/drive/folders/1yURIsV9R9kEYLQovQ-vPogUkXqrIZswA) |

Place the downloaded files under `datasets/source/`:

```text
datasets/source/
|-- MSRS/
|   |-- train/{ir,vi}/
|   `-- test/{ir,vi}/
|-- RoadScene/{cropinfrared,crop_LR_visible}/
|-- M3FD_Fusion/{Ir,Vis}/
`-- TNO/{ir,vi}/
```

Run the preparation script:

```bash
python prepare_data.py
```

[`prepare_data.py`](prepare_data.py) verifies paired filenames and image counts, selects the fixed 200-pair RoadScene subset, resizes all 300 M3FD pairs to half width and height, and creates the model-ready tree:

```text
datasets/SIBA/
|-- train/
|   |-- ir/                    # MSRS 1,083 + RoadScene 200
|   `-- vi/                    # 1,283 matching pairs
`-- test/
    |-- MSRS/{ir,vi}/          # 361 pairs
    |-- M3FD_2x/{ir,vi}/       # 300 pairs
    `-- TNO/{ir,vi}/           # 45 pairs
```

The training directory intentionally combines MSRS and RoadScene because the official SIBA protocol trains one model on both datasets. M3FD and TNO are test sets. With this layout, training and testing use repository-relative paths and require no machine-specific path edits. Download provenance, integrity manifests and dataset citations are documented in [`data/README.md`](data/README.md).

## Training

Run the complete Jittor training protocol:

```bash

python train.py
```

Default settings follow the official implementation: 60 epochs, batch size 4, 128 x 128 random crops, Adam with learning rate `1e-4`, StepLR `25/0.5`, and global L2 gradient clipping at `0.01`.

Each run is written to a new timestamped directory and does not overwrite earlier results:

```text
checkpoint/runs/<YYYYMMDD_HHMMSS>/
|-- SIBA_epoch60.pkl            # trained model
|-- train.log                   # console training log
|-- train_batches.csv           # per-batch total and component losses
|-- epoch_loss_components.csv   # per-epoch losses
|-- loss_components.png         # loss curves
|-- loss_components.pdf
`-- train_metadata.json         # settings, duration and SHA256 values
```

The final verified 60-epoch rerun is indexed in [`results/final_retrain_20260803/`](results/final_retrain_20260803/) and its checkpoint is [`checkpoint/final_retrain_20260803/SIBA_epoch60.pkl`](checkpoint/final_retrain_20260803/SIBA_epoch60.pkl).

## Testing

Test all three datasets:

```bash
python test.py
```

Test one dataset:

```bash
python test.py --dataset TNO
```

`test.py` automatically selects the newest completed checkpoint under `checkpoint/runs/`. If no new run exists, it uses the published final Jittor checkpoint. For each dataset it checks infrared/visible filenames, fuses the visible Y channel, restores Cb/Cr, synchronizes CUDA timing and writes:

```text
results/jittor_test/<dataset>/
|-- *.png                # fused RGB images
|-- timing.csv           # per-image synchronized inference time
`-- summary.json         # dataset, checkpoint hash and mean FPS
```

The optional module check is:

```bash
python tests/test_jittor_modules.py
```

It checks Res-SE, CBSM, self-attention, cross-attention, SIBA and the three losses for valid shapes and finite values. 

## PyTorch Alignment Logs

The repository records three levels of PyTorch/Jittor comparison.

### 1. Module and training-step alignment

[`logs/alignment/alignment_assessment_final_20260728.json`](logs/alignment/alignment_assessment_final_20260728.json) records activation, loss, gradient and one-step update checks. The measured maximum activation error is `2.0206e-4` and the maximum loss error is `2.9802e-6`. Forward and loss checks pass the declared thresholds. Native automatic-differentiation updates are close, but strict element-wise training-step equivalence is not claimed.

### 2. Released-checkpoint inference alignment

Both frameworks load the same author-released PyTorch checkpoint. All 706 test pairs have matching filenames and dimensions; the maximum fused-image difference is one uint8 level (`1/255`). Per-image CSV files and dataset summaries are stored in [`results/alignment/official_checkpoint/`](results/alignment/official_checkpoint/).

### 3. Controlled 60-epoch training

The controlled experiment uses one exported initialization and one shared 60-epoch filename/crop schedule for both frameworks. It contains 19,260 batch records per framework and all 706 test pairs.

- PyTorch training log: [`logs/comparisons/shared_seed2025/pytorch_train.log`](logs/comparisons/shared_seed2025/pytorch_train.log)
- Jittor training log: [`logs/comparisons/shared_seed2025/jittor_train.log`](logs/comparisons/shared_seed2025/jittor_train.log)
- Batch loss CSV files: [`logs/comparisons/shared_seed2025/`](logs/comparisons/shared_seed2025/)
- Loss curves and output comparison: [`results/comparisons/shared_seed2025/`](results/comparisons/shared_seed2025/)
- Experimental protocol: [`docs/CONTROLLED_COMPARISON.md`](docs/CONTROLLED_COMPARISON.md)

Shared inputs reduce initialization and sampling differences, but framework arithmetic and optimizer trajectories are not forced to be identical. The measured output differences are retained in the public logs rather than described as strict equivalence.

## Training and Performance Logs

| Record | PyTorch | Jittor |
|---|---:|---:|
| Independent 60-epoch training time | 2,221 s | 4,079 s |
| Controlled 60-epoch training time | 2,195 s | 4,092 s |

The independent console logs are [`logs/final/pytorch_train_60e.log`](logs/final/pytorch_train_60e.log) and [`logs/final/jittor_train_60e.log`](logs/final/jittor_train_60e.log). The controlled logs, GPU monitor and completion marker are stored in [`logs/comparisons/shared_seed2025/`](logs/comparisons/shared_seed2025/).

Synchronized model-forward speed on the same RTX 3090 is:

| Dataset | Pairs | PyTorch FPS | Jittor FPS |
|---|---:|---:|---:|
| MSRS | 361 | 11.99 | 9.12 |
| M3FD_2x | 300 | 19.90 | 12.66 |
| TNO | 45 | 12.31 | 6.93 |

The complete timing rows are in [`results/inference_timing.csv`](results/inference_timing.csv). GPU utilization and memory records are in [`logs/comparisons/shared_seed2025/gpu_monitor.csv`](logs/comparisons/shared_seed2025/gpu_monitor.csv) and [`gpu_summary.json`](logs/comparisons/shared_seed2025/gpu_summary.json).

## Loss Curves and Visual Results

The controlled experiment records total, Laplacian, intensity and Sobel losses for both frameworks over all 60 epochs:

![PyTorch and Jittor loss components](results/comparisons/shared_seed2025/loss_components.png)

The final Jittor rerun records its complete 19,260-batch training log, four loss curves, seven evaluation metrics, synchronized timing and representative comparisons in [`results/final_retrain_20260803/`](results/final_retrain_20260803/).

![MSRS qualitative comparison](results/final_retrain_20260803/visual_samples/MSRS/00642D/comparison.png)

Red boxes mark local differences between independently trained PyTorch and Jittor models. These models do not share initialization or sample order, so the figure is a qualitative comparison of two complete training runs rather than a strict numerical-equivalence test.

## Migration Notes

The 13 official Python files retain the same relative paths. PyTorch `forward` methods become Jittor `execute` methods, tensor and data APIs are replaced with their Jittor equivalents, and the model topology remains unchanged. SIBA contains infrared and visible modality branches; it does not contain a spatial/frequency dual branch.

Implementation details and the file-by-file mapping are provided in [`MIGRATION.md`](MIGRATION.md) and [`docs/CODE_WALKTHROUGH.md`](docs/CODE_WALKTHROUGH.md).

## Citation

```bibtex
@InProceedings{Wang_2025_ICCV,
    author    = {Wang, Song and Han, Xie and Kuang, Liqun and Wang, Boying and Chen, Zhongyu and Qiao, Zherui and Yang, Fan and Liu, Xiaoxia and Zhang, Bingyu and Wang, Zhixun},
    title     = {The Source Image is the Best Attention for Infrared and Visible Image Fusion},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {13513--13522}
}
```

Please also cite Jittor and the datasets used in your experiments. Dataset citations are listed in [`data/README.md`](data/README.md).

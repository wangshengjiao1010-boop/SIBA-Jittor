# SIBA-Jittor

Jittor reproduction of **The Source Image is the Best Attention for Infrared and Visible Image Fusion**.

- Paper: Song Wang et al., ICCV 2025
- Paper page: <https://openaccess.thecvf.com/content/ICCV2025/html/Wang_The_Source_Image_is_the_Best_Attention_for_Infrared_and_ICCV_2025_paper.html>
- Official PyTorch code: <https://github.com/Afreshbird/SIBA>
- Official source commit: `880a1ddf9eaa610c64e5f25f87fbb146448addc9`
- Jittor: <https://github.com/Jittor/jittor>

This repository changes the implementation framework only. The network topology, losses and loss weights, optimizer, learning-rate schedule, gradient clipping, data loading, random cropping, 60-epoch training protocol, and inference color reconstruction follow the released PyTorch code.

## Repository

```text
SIBA-Jittor/
|-- args/ base_blocks/ loader/ loss/ models/ utils/  # official same-path modules
|-- compat/                 # PyTorch-compatible Adam and gradient clipping
|-- checkpoint/             # Jittor 60-epoch checkpoint
|-- data/                   # dataset protocol and integrity manifests
|-- evaluation/             # wrapper for the metric code linked by SIBA
|-- tests/                  # source, tensor, gradient, and image alignment
|-- logs/final/             # complete 60-epoch training logs
|-- results/                # summaries plus complete per-image experiment outputs
|-- prepare_data.py         # complete train/test data preparation
|-- train.py                # Jittor training entry
`-- test.py                 # Jittor inference entry
```

The 13 official Python files retain the same relative paths. Jittor-specific compatibility code is isolated in `compat/`. The official PyTorch source is not duplicated here; alignment scripts take a separately cloned official repository through `--pytorch-root`.

## Environment

The paper reports a TITAN RTX 24 GB. The reproduction used an RTX 3090 24 GB because the paper GPU was unavailable on AutoDL. Runtime is therefore reported as reproduction-hardware performance, not same-hardware replication.

```bash
conda create -n JittorDome python=3.8.18 -y
conda activate JittorDome
pip install -r requirements.txt
```

Tested versions:

| Component | Version |
|---|---|
| Ubuntu | 20.04 |
| CUDA toolkit | 11.3 |
| Python | 3.8.18 |
| Jittor | 1.3.11.0 |
| NumPy | 1.24.4 |
| OpenCV | 4.8.1.78 |
| Pillow | 10.0.1 |

The captured machine and package record is [logs/final/environment.txt](logs/final/environment.txt).

## Data

The complete protocol uses 1,283 training pairs and 706 test pairs.

| Split | Source | Pairs | Processing |
|---|---|---:|---|
| Train | MSRS | 1,083 | released training set |
| Train | RoadScene | 200 | deterministic public subset, seed 2025 |
| Test | MSRS | 361 | all released test pairs |
| Test | M3FD | 300 | all pairs, half width and height as required by SIBA Sec. 4.1 |
| Test | TNO | 45 | complete SIBA-linked set |

Download the sources linked by the official repository:

- MSRS: <https://github.com/Linfeng-Tang/MSRS>
- RoadScene: <https://github.com/hanna-xu/RoadScene>
- M3FD: <https://github.com/JinyuanLiu-CV/TarDAL>
- TNO: use the Google Drive or Baidu links in the official SIBA README

Prepare all data without reducing any split:

```bash
python prepare_data.py \
  --msrs-root /path/to/MSRS \
  --roadscene-root /path/to/RoadScene \
  --m3fd-root /path/to/M3FD_Fusion \
  --tno-root /path/to/TNO \
  --output datasets
```

Expected output:

```text
datasets/train/{ir,vi}          1,283 pairs
datasets/test/MSRS/{ir,vi}        361 pairs
datasets/test/M3FD_2x/{ir,vi}     300 pairs
datasets/test/TNO/{ir,vi}          45 pairs
```

File lists, dimensions, and SHA256 values are retained in [data/manifests](data/manifests). The authors did not publish their 200 RoadScene filenames or random seed, so this reproduction does not claim that its RoadScene subset is identical to the unpublished author subset. Full provenance is recorded in [data/README.md](data/README.md).

## Train

The command below runs the complete official 60-epoch schedule with batch size 4, patch size 128, Adam learning rate `1e-4`, StepLR step 25/gamma 0.5, and global gradient clipping at 0.01.

```bash
conda activate JittorDome
python train.py \
  --ir-path datasets/train/ir \
  --vi-path datasets/train/vi \
  --output checkpoint \
  --gpu-number 0
```

The checkpoint is saved under a timestamped directory. The completed Jittor checkpoint is provided as [checkpoint/SIBA_epoch60.pkl](checkpoint/SIBA_epoch60.pkl). The unmodified complete training logs are:

- [logs/final/jittor_train_60e.log](logs/final/jittor_train_60e.log)
- [logs/final/pytorch_train_60e.log](logs/final/pytorch_train_60e.log)

The independently trained PyTorch comparison checkpoint is retained as [checkpoint/PyTorch_SIBA_epoch60.pth](checkpoint/PyTorch_SIBA_epoch60.pth).

For a detached AutoDL run, create `screen -S kk`, execute the same command with `python -u`, then use `screen -r kk` to inspect the live loss output.

### Controlled PyTorch/Jittor training comparison

The default command above preserves the released data-loader behavior. For a tightly controlled framework comparison, the formal runner exports one seed-`2025` PyTorch initialization, loads it in both frameworks, applies the same 60-epoch sample and crop schedule, records all four loss terms for every batch, and tests both final checkpoints on all 706 pairs.

```bash
conda activate JittorDome
bash scripts/run_shared_comparison_screen.sh
screen -r kk
```

The runner refuses to overwrite an existing formal run. Successful completion is marked by `logs/shared_seed2025/EXPERIMENT_COMPLETE`; initialization, schedule, checkpoints, batch logs, and datasets are verified by SHA256 or manifests. This shared schedule is an experimental control, not a replacement for the released default shuffle implementation.

The official script prints one loss value every 50 batches. Each retained log contains 420 entries across 60 epochs. Jittor training took 4,079 s; PyTorch training took 2,221 s on the RTX 3090 reproduction machine.

![60-epoch loss curves](results/loss_curve.png)

## Test

Run the Jittor checkpoint on each complete test set:

```bash
python test.py \
  --checkpoint checkpoint/SIBA_epoch60.pkl \
  --data-dir datasets/test/MSRS \
  --output results/fused/MSRS \
  --gpu-number 0

python test.py --checkpoint checkpoint/SIBA_epoch60.pkl --data-dir datasets/test/M3FD_2x --output results/fused/M3FD_2x --gpu-number 0
python test.py --checkpoint checkpoint/SIBA_epoch60.pkl --data-dir datasets/test/TNO --output results/fused/TNO --gpu-number 0
```

`test.py` preserves the official YCbCr decomposition, luminance fusion, RGB reconstruction, clipping, and image saving logic. Only paths and device selection were exposed as command-line arguments.

For the formal PyCharm demonstration, run the complete 45-pair TNO test with synchronized timing:

```bash
python evaluation/run_inference.py \
  --framework jittor \
  --checkpoint checkpoint/SIBA_epoch60.pkl \
  --data-dir datasets/test/TNO \
  --output results/TNO_reproduced \
  --use-cuda --warmup-runs 3 --timing-mode synchronized
```

The retained completed demonstration contains all 45 fused images, not a reduced sample: [results/demo_jittor_tno](results/demo_jittor_tno). PyCharm setup and presentation order are documented in [docs/PYCHARM_REMOTE_GUIDE.md](docs/PYCHARM_REMOTE_GUIDE.md).

## PyTorch Alignment

Clone the frozen official source separately:

```bash
git clone https://github.com/Afreshbird/SIBA.git ../SIBA-official
git -C ../SIBA-official checkout 880a1ddf9eaa610c64e5f25f87fbb146448addc9
```

Create the independent PyTorch reference environment used by the alignment scripts:

```bash
conda create -n PytorchDome python=3.8.18 -y
conda activate PytorchDome
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install kornia==0.7.0 numpy==1.24.4 opencv-python==4.8.1.78 Pillow==10.0.1 tqdm==4.66.1
```

Check file and symbol coverage:

```bash
python tests/audit_source.py \
  --official ../SIBA-official \
  --mirror . \
  --output results/source_audit.json
```

The retained audit has no missing official Python file and no missing official class or function. Jittor-only helper functions are reported separately rather than hidden.

Export the PyTorch reference with the official checkpoint:

```bash
conda activate PytorchDome
python tests/export_pytorch_alignment.py \
  --pytorch-root ../SIBA-official \
  --checkpoint ../SIBA-official/checkpoint/SIBA_epoch60.pth \
  --output alignment/pytorch_reference.npz \
  --device cuda
```

Check the same parameters, inputs, intermediate activations, losses, gradients, clipping, and one Adam update in Jittor:

```bash
conda activate JittorDome
python tests/check_jittor_alignment.py \
  --checkpoint ../SIBA-official/checkpoint/SIBA_epoch60.pth \
  --reference alignment/pytorch_reference.npz \
  --output alignment/jittor_report.json \
  --use-cuda
```

Controlled results:

| Check | Result |
|---|---:|
| Major activation maximum absolute error | `2.0206e-4` |
| Loss maximum absolute error | `2.9802e-6` |
| Native gradient cosine similarity | `0.999945` |
| Native gradient relative L2 error | `1.0508%` |
| One-step update cosine similarity | `0.997738` |
| One-step update relative L2 error | `6.7269%` |
| Adam update error with shared reference gradients | `2.9802e-8` |

Forward/loss alignment and the migrated clipping/Adam implementation pass the declared functional checks. Native framework gradients are close but do not satisfy a `1e-3` strict relative-L2 threshold; strict training-step equality is not claimed. The retained report is [results/alignment/training_step.json](results/alignment/training_step.json).

Released-checkpoint outputs were compared on all 706 images. Filenames and shapes match, and the maximum Jittor/PyTorch pixel difference is one uint8 level on every dataset:

| Dataset | Pairs | Max pixel difference |
|---|---:|---:|
| MSRS | 361 | 1 |
| M3FD_2x | 300 | 1 |
| TNO | 45 | 1 |

Machine-readable summaries are in [results/alignment](results/alignment).

## Quantitative Results

Metrics use the SIBA-linked `Linfeng-Tang/Evaluation-for-Image-Fusion` implementation frozen at commit `f5f055bcadb49c22fb734c3498aef6c56fc71f2a`. The retained wrapper is under [evaluation](evaluation), and the complete summary is [results/metrics_summary.csv](results/metrics_summary.csv).

```bash
git clone https://github.com/Linfeng-Tang/Evaluation-for-Image-Fusion.git ../Evaluation-for-Image-Fusion
git -C ../Evaluation-for-Image-Fusion checkout f5f055bcadb49c22fb734c3498aef6c56fc71f2a

python evaluation/run_matlab_evaluation.py \
  --matlab /path/to/matlab \
  --evaluation-dir ../Evaluation-for-Image-Fusion/Evaluation \
  --data-root datasets/test \
  --experiment Jittor=results/fused \
  --output-dir results/metric_run
```

MATLAB is used only for the paper-linked metric definitions. Jittor training and inference are entirely Python/Jittor.

Complete per-image CSV files, MATLAB logs, summary tables, and metric plots for all four experiment branches and all three datasets are retained under [results/metrics_20260727_siba_official_protocol](results/metrics_20260727_siba_official_protocol). Complete inference timing, released-checkpoint alignment, training analysis, and raw formal run logs are retained alongside the concise top-level summaries. Development smoke tests and 20-step training outputs are intentionally excluded.

### Released checkpoint

| Dataset | Framework | VIF | SCD | MI | Qabf | SSIM | MS-SSIM | FMI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MSRS | PyTorch | 1.060886 | 1.700237 | 5.109932 | 0.715244 | 0.980835 | 0.975079 | 0.932670 |
| MSRS | Jittor | 1.060896 | 1.700205 | 5.110839 | 0.715261 | 0.980834 | 0.975080 | 0.932669 |
| M3FD_2x | PyTorch | 0.759281 | 1.660127 | 3.770172 | 0.653655 | 0.964490 | 0.923059 | 0.882748 |
| M3FD_2x | Jittor | 0.759335 | 1.660086 | 3.770833 | 0.653689 | 0.964495 | 0.923055 | 0.882752 |
| TNO | PyTorch | 0.835492 | 1.724495 | 3.508012 | 0.587577 | 0.931756 | 0.904369 | 0.914383 |
| TNO | Jittor | 0.835515 | 1.724506 | 3.507112 | 0.587575 | 0.931762 | 0.904372 | 0.914384 |

### Independent 60-epoch training

| Dataset | Framework | VIF | SCD | MI | Qabf | SSIM | MS-SSIM | FMI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MSRS | PyTorch | 1.051570 | 1.703795 | 5.043079 | 0.709814 | 0.979919 | 0.974918 | 0.932430 |
| MSRS | Jittor | 1.066289 | 1.690859 | 5.460504 | 0.710271 | 0.973810 | 0.971632 | 0.932556 |
| M3FD_2x | PyTorch | 0.739027 | 1.723960 | 3.546561 | 0.649051 | 0.968319 | 0.934148 | 0.880840 |
| M3FD_2x | Jittor | 0.725593 | 1.670161 | 3.864333 | 0.629573 | 0.962757 | 0.923434 | 0.878365 |
| TNO | PyTorch | 0.799915 | 1.754528 | 3.162455 | 0.577163 | 0.946520 | 0.917269 | 0.911972 |
| TNO | Jittor | 0.802737 | 1.736843 | 3.334378 | 0.567382 | 0.947852 | 0.915246 | 0.911662 |

Independent full training demonstrates convergence and usable outputs. It is not treated as batch-wise numerical equivalence because PyTorch and Jittor use different data-loader shuffle sequences.

## Performance

Synchronized model-forward timing on RTX 3090:

| Dataset | Jittor FPS | PyTorch FPS |
|---|---:|---:|
| MSRS | 9.12 | 11.99 |
| M3FD_2x | 12.66 | 19.90 |
| TNO | 6.93 | 12.31 |

The full timing table, including the official unsynchronized timer, is [results/inference_timing.csv](results/inference_timing.csv). Unsynchronized CUDA timing is retained for protocol fidelity but is not presented as completed inference latency.

## Visual Results

Released-checkpoint comparison on MSRS:

![MSRS released checkpoint comparison](results/visual/MSRS_official_checkpoint_grid.png)

Independent 60-epoch comparison on MSRS:

![MSRS self-trained comparison](results/visual/MSRS_self_trained_grid.png)

M3FD_2x and TNO comparisons are also retained in [results/visual](results/visual).

## Migration Notes

All framework-forced substitutions and debugging records are documented in [MIGRATION.md](MIGRATION.md). The main non-mechanical points are:

- PyTorch `forward` becomes Jittor `execute`.
- Kornia's normalized Laplacian is reproduced with the same kernel, reflect padding, and grouped convolution.
- PyTorch 1.10 Adam epsilon placement is reproduced in `compat/pytorch_adam.py`.
- PyTorch global L2 clipping operation order is reproduced in `compat/pytorch_clip.py`.
- Jittor dataset length and batching use `set_attrs`; sample discovery, normalization, crops, and pairing are unchanged.
- Jittor image output is transposed from CHW to HWC immediately before `ToPILImage`.

## Citation

```bibtex
@InProceedings{Wang_2025_ICCV,
    author    = {Wang, Song and Han, Xie and Kuang, Liqun and Wang, Boying and Chen, Zhongyu and Qiao, Zherui and Yang, Fan and Liu, Xiaoxia and Zhang, Bingyu and Wang, Zhixun},
    title     = {The Source Image is the Best Attention for Infrared and Visible Image Fusion},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2025},
    pages     = {13513--13522}
}
```

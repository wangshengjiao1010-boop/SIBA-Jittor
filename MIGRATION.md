# PyTorch-to-Jittor Migration Record

## Baseline

- Paper: *The Source Image is the Best Attention for Infrared and Visible Image Fusion*, ICCV 2025
- Official repository: <https://github.com/Afreshbird/SIBA>
- Frozen commit: `880a1ddf9eaa610c64e5f25f87fbb146448addc9`
- Official Python files: 13
- Jittor counterparts: 13, with the same relative paths

The public repository contains the Jittor implementation only. The official PyTorch repository is cloned separately for source and numerical alignment.

## Preserved Protocol

The migration preserves:

- model topology, channels, blocks, attention order, and tensor shapes;
- Laplacian, intensity, and Sobel losses and their weights `10`, `0.1`, and `1`;
- Adam settings: learning rate `1e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0`;
- StepLR step size `25`, gamma `0.5`, and learning-rate floor `1e-6`;
- global L2 gradient clipping at `0.01`;
- batch size `4`, patch size `128`, and `60` epochs;
- grayscale training loader, paired random crops, and filename assertions;
- test YCbCr conversion, luminance fusion, RGB reconstruction, and clipping.

The command-line path arguments added to `train.py` and `test.py` do not change these rules.

## File Mapping

| Official PyTorch | Jittor | Change |
|---|---|---|
| `args/args_SIBA.py` | `args/args_SIBA.py` | framework-independent settings retained |
| `base_blocks/cbsm.py` | `base_blocks/cbsm.py` | `torch.nn` to `jittor.nn` |
| `base_blocks/restormer.py` | `base_blocks/restormer.py` | tensor and module APIs migrated |
| `base_blocks/SE.py` | `base_blocks/SE.py` | module API migrated |
| `base_blocks/se_resnet.py` | `base_blocks/se_resnet.py` | module API migrated |
| `loader/train_loader.py` | `loader/train_loader.py` | Jittor `Dataset` and `set_attrs` |
| `loader/test_loader.py` | `loader/test_loader.py` | Jittor transforms and `Dataset` |
| `loss/loss.py` | `loss/loss.py` | Kornia Laplacian reproduced in Jittor |
| `models/SIBA.py` | `models/SIBA.py` | `forward` renamed to `execute` |
| `utils/resize_resolution.py` | `utils/resize_resolution.py` | framework-independent |
| `utils/RGB2YCrBb.py` | `utils/RGB2YCrBb.py` | matrix operations migrated |
| `train.py` | `train.py` | optimizer/backward/data-loader APIs migrated |
| `test.py` | `test.py` | checkpoint loading and image conversion migrated |

Additional framework compatibility code:

- `compat/pytorch_adam.py`
- `compat/pytorch_clip.py`

## Framework-Forced Changes

### Module execution

PyTorch modules implement `forward`; Jittor modules implement `execute`. Class names, constructor arguments, submodule order, residual paths, attention paths, and concatenation order remain unchanged.

### Normalization

The official call `torch.nn.functional.normalize(x, dim=-1)` is implemented as division by `max(L2 norm, 1e-12)` along the same dimension.

### Laplacian loss

Jittor has no Kornia dependency. The Kornia 0.7.0 normalized 3x3 kernel, reflect padding, channel expansion, and grouped convolution are reproduced directly. Alignment is checked against the official Kornia execution.

### Adam

Jittor's built-in Adam and PyTorch 1.10 place epsilon differently in the bias-corrected denominator. `compat/pytorch_adam.py` follows the PyTorch 1.10 update equation. With shared reference gradients, the maximum post-step parameter error is `2.9802e-8`.

### Gradient clipping

`compat/pytorch_clip.py` follows PyTorch 1.10's per-gradient norm, stacked global norm, and `1e-6` denominator order. With shared reference gradients, the maximum post-clip error is `4.8894e-9`.

### Dataset API

Jittor requires dataset length and batching through `set_attrs`. File discovery, filename equality checks, grayscale decoding, `[0,1]` normalization, random crop coordinates, shuffling, and final incomplete batch behavior are retained.

### Image output

Jittor `ToPILImage` expects HWC input in the tested environment. The fused RGB output is transposed from CHW to HWC immediately before saving.

## Alignment Results

A controlled PyTorch export records identical inputs and parameters, major activations, all loss terms, every pre/post-clip gradient, and one Adam update. Jittor checks 706 tensor entries against that export.

| Measurement | Value |
|---|---:|
| activation maximum absolute error | `2.0206e-4` |
| loss maximum absolute error | `2.9802e-6` |
| native gradient cosine similarity | `0.999945` |
| native gradient relative L2 error | `1.0508%` |
| native update cosine similarity | `0.997738` |
| native update relative L2 error | `6.7269%` |

Forward and loss alignment pass. The compatibility implementations pass when supplied with shared gradients. Native gradients and updates are close under the declared engineering tolerances but do not meet strict `1e-3` relative-L2 equality. The repository does not claim bitwise or strict training-step equivalence.

The released checkpoint was then evaluated through both frameworks on all 361 MSRS, 300 M3FD, and 45 TNO pairs. Filenames and dimensions match, and every dataset has a global maximum output difference of one uint8 level.

## Complete Training

- Training pairs: 1,083 MSRS + 200 RoadScene = 1,283
- Epochs: 60 in both frameworks
- Jittor runtime: 4,079 s on RTX 3090
- PyTorch runtime: 2,221 s on RTX 3090
- Logged loss samples: 420 per framework, matching the official one-print-per-50-batches rule
- Test outputs: all 706 pairs for each formal branch

Full logs and result summaries are retained under `logs/final/` and `results/`.

## Official-Code Limitations

- The paper describes RGB-to-YCbCr training, while the released loader reads both modalities as grayscale. This reproduction follows the released code.
- The paper and repository state that 200 RoadScene pairs are selected randomly, but their filenames and seed are not public. A deterministic seed-2025 subset is shared by PyTorch and Jittor; identity with the authors' subset is not claimed.
- M3FD is evaluated at half width and height because SIBA Section 4.1 requires this protocol. All 300 pairs are retained.
- The paper reports TITAN RTX. The reproduction used RTX 3090; timing is not presented as same-hardware replication.

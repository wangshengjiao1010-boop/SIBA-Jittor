# SIBA PyTorch-to-Jittor Migration Log

This document records every framework-forced change from the official SIBA implementation. Algorithmic changes are not permitted.

## Source baseline

- Official repository: <https://github.com/Afreshbird/SIBA>
- Official commit: `880a1ddf9eaa610c64e5f25f87fbb146448addc9`
- Paper: *The Source Image is the Best Attention for Infrared and Visible Image Fusion*, ICCV 2025.
- Official source snapshot: `official_pytorch/`
- Jittor mirror: `siba_jittor/`

## Rules

1. Preserve the official directory and symbol structure.
2. Preserve model topology, tensor shapes, initialization, loss weights, optimizer, scheduler, gradient clipping, preprocessing, and inference color reconstruction.
3. Limit changes to framework API substitutions and additive reproducibility logging.
4. Record every non-mechanical substitution below and verify it numerically against PyTorch before full training.
5. Do not report an experiment until its raw log, configuration, checkpoint hash, and result files exist.

## Migration entries

- All 13 official Python files have a same-path Jittor counterpart.
- PyTorch `forward` methods are renamed to Jittor `execute`; model topology and call order are unchanged.
- `torch.nn`, tensor creation, concatenation, matrix operations, reductions, and activations are replaced with their Jittor equivalents.
- `torch.nn.functional.normalize` is implemented as division by `max(L2 norm, 1e-12)` to preserve PyTorch's default normalization rule.
- `kornia.filters.laplacian` is reproduced from Kornia 0.7.0 with the same normalized 3x3 kernel, reflect padding, and depthwise convolution.
- Fixed Sobel kernels remain non-trainable through Jittor `stop_grad`.
- Jittor datasets retain the official file discovery, filename assertion, grayscale loading, normalization, and random crop coordinates; `set_attrs(total_len=...)` is required by the Jittor dataset API.
- The official PyTorch `DataLoader(batch_size, shuffle=True)` is mapped to Jittor dataset attributes with the same batch size, shuffling, no dropped final batch, and zero worker processes.
- Adam uses the official PyTorch defaults explicitly: betas `(0.9, 0.999)`, epsilon `1e-8`, and zero weight decay.
- Backpropagation and global L2 gradient clipping use Jittor optimizer APIs with the official `max_norm=0.01`.
- The StepLR schedule, 25-epoch step size, 0.5 decay factor, and `1e-6` learning-rate floor are retained.
- Jittor training checkpoints use `.pkl`; the stored dictionary and model state structure remain `{'model': state_dict}`.
- CPU-only AutoDL startup requires `nvcc_path=` before importing Jittor. Without it, Jittor detects the installed CUDA toolkit in no-card mode and attempts to create a CUDA stream despite `use_cuda=0`, producing `cudaErrorInsufficientDriver`. This is an environment launch setting and does not change model code.
- Formal training scripts use unbuffered Python inside `screen -S kk` and write both `screen.log` and `train.log` so progress is visible during detached execution.

## Framework-forced implementation details

### Module execution API

- PyTorch modules implement `forward`; Jittor modules implement `execute`.
- Class names, constructor arguments, submodule order, residual paths, attention paths, and tensor concatenation order are unchanged.

### Normalization

- Official code: `torch.nn.functional.normalize(x, dim=-1)`.
- Jittor mirror: `x / max(sqrt(sum(x * x)), 1e-12)` along the same dimension.
- This is the PyTorch default L2 normalization formula, not a new normalization design.

### Laplacian loss

- The official loss calls `kornia.filters.laplacian(..., kernel_size=3)`.
- Jittor has no Kornia dependency, so the Kornia 0.7.0 normalized 3x3 kernel, reflect padding, channel expansion, and grouped convolution are reproduced directly.
- Loss alignment is checked against the official Kornia execution rather than against a visually similar Laplacian approximation.

### Adam optimizer

- Jittor's built-in Adam and PyTorch 1.10 do not place epsilon identically in the bias-corrected denominator.
- `compat/pytorch_adam.py` implements the official PyTorch update:
  `sqrt(exp_avg_sq) / sqrt(1 - beta2^step) + eps`, followed by first-moment bias correction.
- With identical PyTorch gradients, the migrated clipping plus Adam update has maximum parameter error `2.9802e-8`.

### Gradient clipping

- PyTorch computes each parameter-gradient norm, stacks those norms, and computes a second global norm.
- Jittor's built-in helper reduces a concatenated gradient representation and did not reproduce the same rounding behavior.
- `compat/pytorch_clip.py` follows the PyTorch 1.10 operation order and uses the same `1e-6` coefficient denominator.
- With identical PyTorch gradients, maximum post-clip error is `4.8894e-9`.

### Dataset interface

- Official filename discovery, filename equality assertion, grayscale decoding, `[0,1]` normalization, and identical random crop coordinates for infrared and visible images are retained.
- Jittor requires dataset length and batching through `set_attrs`; no crop or augmentation was added.
- Four controlled real training pairs produce pixel-identical `128 x 128` crops in both loaders.

### Inference conversion

- Jittor `ToTensor` returns NumPy-backed values in this environment, so the clipping helper accepts both Jittor and NumPy values without changing its numerical bounds.
- PyTorch tensor `.mm()` is mapped to `jt.matmul`.
- Jittor `ToPILImage` expects HWC input, so the final fused RGB image is transposed from CHW to HWC immediately before saving.
- The released checkpoint produces a maximum difference of one uint8 level on the controlled full-resolution image.

## Debugging chronology

1. The initial source mirror was checked for official file and symbol coverage before any full training.
2. Checkpoint key count and model parameter count were verified as `137` and `565,941`.
3. A controlled PyTorch export captured inputs, all major activations, individual loss terms, every gradient, clipped gradients, and one-step parameters.
4. The Jittor run compared 706 tensor checks against that export. Activation maximum error was `2.0206e-4`, total-loss maximum error was `2.9802e-6`, full-gradient cosine similarity was `0.999945`, gradient relative L2 error was `1.0508%`, and one-step parameter-update relative L2 error was `6.7269%`. These values support close functional migration but not strict training-step equality.
5. Real training-pair loading and cropping were compared independently and showed zero pixel error.
6. GPU smoke training was completed before the 60-epoch run.
7. Full Jittor training completed in `4,079` seconds and full PyTorch training completed in `2,221` seconds.
8. Jittor and PyTorch self-trained inference completed on all three test sets. The interrupted PyTorch M3FD run was resumed without replacing existing valid outputs.
9. The released checkpoint was evaluated in both frameworks on all 706 test images. Filenames and dimensions match, and the global maximum pixel error is one uint8 level.
10. Synchronized latency, official asynchronous timing, GPU monitoring, loss curves, and visual comparison grids were generated from real runs.
11. All seven linked MATLAB metrics were computed for four framework/checkpoint combinations on MSRS, M3FD_2x, and TNO, producing 2,824 per-image metric rows in total.
12. Paper deltas, Jittor/PyTorch deltas, metric-ratio plots, and formal accelerated-function equivalence results were generated from the retained outputs.
13. A paired PyTorch/Jittor Jupyter demonstration was added. PyTorch exports the shared seed-2025 initialization before the short Jittor run, and the Jittor script loads that checkpoint without changing the official training rule.
14. Four notebooks were executed end to end on the RTX 3090. All code cells completed with zero error outputs; the live run produced a 20-step checkpoint and TNO inference produced 45 images.
15. The first whole-notebook execution exposed missing remote copies of existing audit, metric, and performance files. Those retained real artifacts were synchronized, and all notebooks were rerun successfully. No model or training logic was changed for this correction.

## Evaluation implementation verification

- The SIBA repository links `Linfeng-Tang/Evaluation-for-Image-Fusion` at frozen commit `f5f055bcadb49c22fb734c3498aef6c56fc71f2a`.
- The linked toolkit's active pairwise `ssim` call receives `double` images in `[0,255]`; MATLAB's default double-image dynamic range produces `0.508915` on the complete TNO set and does not reproduce Table 1.
- The alternative `mef_ssim` call retained in the same official function produces `0.931756`, which rounds to the paper's `0.932`. This paper-consistent definition is used for final reporting.
- `mef_ssim_fast.m` is an algebraically equivalent convolution form of the official nested loops. On the controlled full-resolution sample, the absolute score error is `1.221e-15`.
- `analysis_ms_ssim_fast.m` retains the official three scales, weights, symmetric filtering, and downsampling while calling the equivalent MEF-SSIM implementation. The controlled score error is `2.220e-16`.
- `analysis_fmi_fast.m` vectorizes the official 3-by-3 sliding-window calculation without changing the feature extraction or probability equations. The controlled full-resolution sample has zero score error against `analysis_fmi.m`.
- The complete TNO official-checkpoint evaluation reproduces all seven paper values after rounding: VIF `0.836`, SCD `1.724`, MI `3.508`, Qabf `0.588`, SSIM `0.932`, MS-SSIM `0.904`, and FMI `0.914`.
- The formal validation CSV records absolute errors of `1.665e-15` for MEF-SSIM, `2.220e-16` for MS-SSIM, and `0` for FMI on a real full-resolution TNO sample.
- Across all three datasets, released-checkpoint Jittor/PyTorch metric differences are at most `5.42e-5` VIF, `4.08e-5` SCD, `9.07e-4` MI, `3.42e-5` Qabf, `6.52e-6` SSIM, `4.24e-6` MS-SSIM, and `3.31e-6` FMI.
- Self-trained metric differences are retained but are not treated as controlled numerical-alignment errors because shuffled sample orders diverge between framework loaders and the official RoadScene 200-file subset is unpublished.

## AutoDL no-card limitation

- AutoDL no-card mode exposes host memory information to Jittor while the container remains subject to a much smaller effective memory allowance.
- A fresh Jittor cache therefore selected 16 parallel compiler workers and one compiler process was OOM-killed.
- The established single-process CPU cache (`DISABLE_MULTIPROCESSING=1`) remains valid; the current mirror was successfully instantiated from that cache with 137 parameter tensors and 565,941 parameters.
- No-card mode was used for source audit, Python syntax checks, log parsing, MATLAB preparation, and documentation. GPU mode was limited to alignment, smoke testing, full training, full-image inference, and timing, and the instance was shut down after completion.

## Timing protocol

- Official `test.py` records `time.time()` immediately around the CUDA model call without synchronization.
- This asynchronous value is not equivalent to completed GPU latency.
- The reproduction records both the exact official unsynchronized protocol and a synchronized model-forward protocol. The two values are labeled separately and are never mixed in one comparison.

## Known official-code facts retained during reproduction

- Training loads both infrared and visible images with `cv2.IMREAD_GRAYSCALE`, although the paper describes RGB-to-YCbCr preprocessing.
- The official repository specifies MSRS training images plus 200 RoadScene image pairs, but does not publish the selected RoadScene filenames or random seed.
- The reproduction uses a deterministic public 200-of-221 manifest shared by PyTorch and Jittor; it does not claim identity with the authors' unpublished subset.
- RoadScene provides `crop_HR_visible` and `crop_LR_visible`. The official loader applies identical pixel crop coordinates to both modalities, so the size-matched pair `cropinfrared` + `crop_LR_visible` is used. `crop_HR_visible` has different dimensions and would silently produce spatially inconsistent patches under the official loader.
- The rented GPU is RTX 3090 24 GB; the paper reports TITAN RTX 24 GB. Accuracy experiments remain comparable, while runtime is reported as reproduction hardware rather than paper-hardware equivalence.

# Review of Existing Jittor Reproductions

The task workflow was checked against `GrokCV/Jittor-Sprouts` and representative fusion reproductions before implementing SIBA.

## Practices retained

- Preserve both the upstream PyTorch implementation and the Jittor implementation in the repository.
- Include data preparation, training, testing, evaluation, and visualization rather than model inference alone.
- Publish raw logs, loss curves, generated images, quantitative metrics, and runtime/memory measurements.
- Explain framework API substitutions and numerical discrepancies instead of presenting the migration as a simple name replacement.
- Use fixed inputs and shared weights for module-level PyTorch/Jittor alignment.

## Shortcomings not repeated

- A small RoadScene demonstration is not presented as a full reproduction.
- Missing loss terms or partial model branches are not accepted.
- A visual example alone is not treated as PyTorch/Jittor alignment.
- Final metrics are not reported without the corresponding checkpoint, configuration, raw output, and log.

## SIBA acceptance standard

1. All 13 official Python files have a same-path Jittor counterpart.
2. The official pretrained checkpoint runs in both frameworks.
3. Model blocks, full forward output, all three loss terms, gradients, clipping, and one Adam update are numerically checked.
4. MSRS plus the disclosed deterministic 200-pair RoadScene subset is trained for the official 60 epochs.
5. MSRS, half-resolution M3FD, and TNO are evaluated using the seven paper metrics.
6. Environment, data manifests, logs, curves, fusion outputs, speed, memory, and debugging history are retained.

The detailed item-by-item comparison, experiment matrix, and presentation order are documented in `docs/EXPERIMENT_DESIGN_AND_REFERENCE_COMPARISON.md`.

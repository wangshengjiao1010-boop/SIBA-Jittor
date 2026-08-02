# Result Index

- `loss_curve.png`, `epoch_loss.csv`: independent complete 60-epoch total-loss comparison.
- `inference_timing.csv`: synchronized same-GPU inference summary.
- `metrics/metrics_summary.csv`: seven metrics for author weights and independent self-trained weights on all three datasets.
- `metrics/`: per-image metrics, evaluator logs, plots and portable provenance.
- `alignment/official_checkpoint/`: released-checkpoint per-image output differences; near-identical output is expected because both frameworks load the same weights.
- `alignment/self_trained/`: per-image differences between the two independently trained 60-epoch checkpoints.
- `visual/`: qualitative comparison grids for MSRS, M3FD and TNO.
- `jittor_test/TNO/`: complete 45-image Jittor inference output with synchronized timing.
- `comparisons/shared_seed2025/`: completed shared-initialization 60-epoch experiment, including four loss curves, timing, per-image differences and selected qualitative samples.

The full 706-image author-checkpoint and independently trained output batches remain in the dated execution directories as raw experiment evidence. The root-level files above are the recommended entry points for readers.

No synthetic metrics, placeholder fusion images, reduced-epoch training, or reduced test subsets are used as formal results.

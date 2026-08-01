# Result Index

- `loss_curve.png`, `epoch_loss.csv`: independent complete 60-epoch total-loss comparison.
- `inference_timing.csv`: synchronized same-GPU inference summary.
- `metrics/metrics_summary.csv`: seven metrics for author weights and independent self-trained weights on all three datasets.
- `metrics/`: per-image metrics, evaluator logs, plots and portable provenance.
- `alignment/`: released-checkpoint per-image output differences.
- `visual/`: qualitative comparison grids for MSRS, M3FD and TNO.
- `jittor_test/TNO/`: complete 45-image Jittor inference output with synchronized timing.

The full 706-image author-checkpoint and independently trained output batches remain in the dated execution directories as raw experiment evidence. The root-level files above are the recommended entry points for readers.

No synthetic metrics, placeholder fusion images, reduced-epoch training, or reduced test subsets are used as formal results.

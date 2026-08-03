# Checkpoint Provenance

The included weights come from the independent complete 60-epoch runs recorded in `logs/final/`. They are not the author-released PyTorch checkpoint.

| File | Framework | SHA256 |
|---|---|---|
| `SIBA_jittor_self_trained_epoch60.pkl` | Jittor 1.3.11.0 | `7aecde5004cb6304d6fff9b1bdb772f4fcf5876162cda28b75a8b92e8aad45c8` |
| `SIBA_pytorch_self_trained_epoch60.pth` | PyTorch 1.10.0 | `93ac201a9db903af19cb12f63cb3da06617449593a35b258ccaa26c5e42f2313` |
| `final_retrain_20260803/SIBA_epoch60.pkl` | Jittor 1.3.11.0 | `9926f7c5943385e5fc57a90bd6eb2bb3b8a33b6f20de5ee6803ce41a391119d3` |

Protocol: 1,283 training pairs, 60 epochs, batch size 4, patch size 128, Adam `1e-4`, StepLR `25/0.5`, and global L2 gradient clipping at `0.01`.

The author-released checkpoint remains in the separately cloned official repository and is used by the alignment scripts through an explicit path.

The controlled shared-initialization checkpoints are stored separately in `comparisons/shared_seed2025/`. They use the same initial parameter archive and the same 60-epoch crop schedule; their SHA256 values and training metadata are retained in `logs/comparisons/shared_seed2025/`.

The `final_retrain_20260803` checkpoint is the default fallback used by `python test.py` on a fresh clone. It was trained after the final runtime-path cleanup at revision `126400f`, reloaded in a new process, and tested on all 706 image pairs. Its logs and metrics are indexed in `results/final_retrain_20260803/README.md`.

# Official Source Audit

## Frozen baseline

The official repository is frozen at commit `880a1ddf9eaa610c64e5f25f87fbb146448addc9` under `official_pytorch/`.

The snapshot contains 13 Python files:

| File | Lines | Purpose |
|---|---:|---|
| `args/args_SIBA.py` | 20 | Training configuration |
| `base_blocks/cbsm.py` | 17 | Source-image channel boosting and spatial mapping |
| `base_blocks/restormer.py` | 244 | Self-attention, source-image cross-attention, GDFN, layer normalization |
| `base_blocks/SE.py` | 32 | Squeeze-and-excitation block |
| `base_blocks/se_resnet.py` | 44 | SE residual block |
| `loader/test_loader.py` | 35 | Test-pair loading and RGB-to-YCbCr decomposition |
| `loader/train_loader.py` | 41 | Grayscale training-pair loading and random patch extraction |
| `loss/loss.py` | 67 | Laplacian, intensity, and Sobel losses |
| `models/SIBA.py` | 112 | Full SIBA architecture |
| `test.py` | 50 | Checkpoint inference and color reconstruction |
| `train.py` | 97 | Full 60-epoch training procedure |
| `utils/resize_resolution.py` | 30 | Half-resolution M3FD preprocessing |
| `utils/RGB2YCrBb.py` | 43 | RGB/YCbCr conversion and clipping |

Run the machine-readable audit with:

```bash
python tools/audit_source.py \
  --official official_pytorch \
  --mirror siba_jittor \
  --output docs/source_audit.json
```

The comparison fails review if an official Python file or official class/function symbol is absent from the Jittor mirror. Functional equivalence is additionally checked by numerical alignment tests; matching file and symbol inventories alone is not considered sufficient.


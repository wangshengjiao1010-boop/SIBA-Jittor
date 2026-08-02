# PyCharm Remote Reproduction and Demonstration

PyCharm connects to the AutoDL host through SSH. Training and testing still run in the two remote Conda environments; PyCharm is only the editor and process launcher.

## Interpreters

Create the environments with the commands in `README.md`, then add two SSH interpreters in PyCharm:

- `PytorchDome`: the Python executable inside the remote PyTorch environment.
- `JittorDome`: the Python executable inside the remote Jittor environment.

Map the local repository to one remote project directory. Do not place environments inside the Git repository.

## Complete Training Entry

Select `JittorDome`, set the working directory to the repository root, and place the prepared data under `datasets/SIBA` as documented in the README. Run:

```bash
python train.py
```

This is the complete official 60-epoch configuration. A long run should be started from the PyCharm terminal inside `screen -S kk`; closing the PyCharm SSH session must not terminate training.

The completed run is already retained in:

- `logs/final/jittor_train_60e.log`
- `logs/final/pytorch_train_60e.log`
- `results/epoch_loss.csv` and `results/loss_curve.png`
- `checkpoint/SIBA_jittor_self_trained_epoch60.pkl`
- `checkpoint/SIBA_pytorch_self_trained_epoch60.pth`

Do not start another 60-epoch run only for presentation.

## Complete Test Demonstration

Set `model_path`, `testdata_paths`, and `result_save_path` near the top of `test.py`, then run:

```bash
python test.py
```

The command processes all configured MSRS, M3FD and TNO pairs and saves every fused image, `timing.csv`, and `summary.json` under `results/jittor_test/`. The complete 45-image TNO output is retained under `results/jittor_test/TNO/`.

## Module Test Demonstration

Run the migrated Jittor modules before training:

```bash
python tests/test_jittor_modules.py
```

The script checks the main feature block, source-image query module, self-attention, cross-attention, complete SIBA forward path, and all three loss terms on deterministic test fixtures.

## Alignment Demonstration

Use `PytorchDome` to run `tests/export_pytorch_alignment.py` against a separately cloned official SIBA repository. Then use `JittorDome` to run `tests/check_jittor_alignment.py` with the exported `.npz` file. The recorded checks and tolerances are documented in `MIGRATION.md` and `docs/REPRODUCIBILITY_AUDIT.md`.

This demonstration covers the same inputs and parameters, major activations, all loss terms, gradients, clipping, and one Adam update. It does not claim bitwise training-step equality.

## Presentation Order

1. Show the official paper and frozen official commit recorded in `README.md`.
2. Open `models/SIBA.py`, `base_blocks/restormer.py`, `loss/loss.py`, and `train.py`.
3. Show the two complete 60-epoch logs and loss curve.
4. Run the complete 45-image TNO Jittor inference.
5. Open the newly generated images, `summary.json`, and `timing.csv`.
6. Show the 706-image released-checkpoint alignment and the three-dataset metric tables.
7. Open the public GitHub repository and confirm that the displayed commit matches the local repository.

No password, SSH private key, or GitHub token should appear in the recording.

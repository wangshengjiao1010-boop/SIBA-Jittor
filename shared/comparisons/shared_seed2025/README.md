# Shared Training Inputs

- `initial.npz`: framework-neutral parameter archive exported from PyTorch with seed 2025.
- `initial.json`: parameter names, shapes and initial-state hash.
- `schedule.npz`: exact 60-epoch filename order and crop coordinates.
- `schedule.json`: schedule protocol and SHA256.

These files reproduce the controlled initial conditions; they are not trained model checkpoints.

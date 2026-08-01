#!/usr/bin/env python3
"""CPU smoke tests for the migrated Jittor modules.

These deterministic tensors test shapes and finite outputs only. They are not
training data and do not produce paper metrics or experimental results.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import jittor as jt

from base_blocks.cbsm import CBSM
from base_blocks.restormer import TransformerBlock_CA, TransformerBlock_SA
from base_blocks.se_resnet import Res_SE
from loss.loss import Fusionloss, JointGrad
from models.SIBA import SIBA


def check(name, value, shape):
    actual = tuple(value.shape)
    if actual != tuple(shape):
        raise AssertionError(f"{name}: expected {shape}, got {actual}")
    if not np.isfinite(value.numpy()).all():
        raise AssertionError(f"{name}: non-finite output")
    print(f"[PASS] {name}: {actual}")


def check_scalar(name, value):
    array = value.numpy()
    if array.size != 1 or not np.isfinite(array).all():
        raise AssertionError(f"{name}: expected one finite value, got {array}")
    print(f"[PASS] {name}: {float(array.reshape(-1)[0]):.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    jt.flags.use_cuda = 1 if args.cuda else 0
    jt.set_global_seed(2025)

    source = jt.array(
        np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(1, 1, 32, 32)
    )
    feature = Res_SE(1, 48)(source)
    check("Res_SE", feature, (1, 48, 32, 32))
    source_weight = CBSM(48)(source)
    check("CBSM", source_weight, (1, 48, 32, 32))
    self_attention = TransformerBlock_SA(48, 48)(feature)
    check("Self-Attention", self_attention, (1, 48, 32, 32))
    cross_attention = TransformerBlock_CA(48, 48)(feature, source_weight)
    check("Cross-Attention", cross_attention, (1, 48, 32, 32))

    model = SIBA()
    fused = model(source, 1 - source)
    check("SIBA", fused, (1, 1, 32, 32))
    joint = JointGrad()(fused, source, 1 - source)
    intensity, sobel = Fusionloss()(fused, source, 1 - source)
    check_scalar("Laplacian loss", joint)
    check_scalar("Intensity loss", intensity)
    check_scalar("Sobel loss", sobel)
    print("All Jittor module checks passed.")


if __name__ == "__main__":
    main()

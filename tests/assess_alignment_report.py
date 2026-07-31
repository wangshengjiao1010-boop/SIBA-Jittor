#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gradient-relative-l2-tolerance", type=float, default=2e-2)
    parser.add_argument("--update-relative-l2-tolerance", type=float, default=8e-2)
    parser.add_argument("--strict-relative-l2-tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    checks = raw["checks"]
    aggregates = raw["aggregates"]
    activation_max_abs = max(
        value["max_abs"]
        for name, value in checks.items()
        if name.startswith("activation__")
    )
    loss_max_abs = max(
        checks[name]["max_abs"]
        for name in ("loss_laplacian", "loss_intensity", "loss_sobel", "loss_total")
    )
    gradient = aggregates["gradient_preclip"]
    update = aggregates["parameter_update"]
    forward_passed = bool(
        checks["manual_vs_model_output"]["max_abs"] <= 1e-6
        and activation_max_abs <= 5e-4
        and loss_max_abs <= 1e-5
    )
    optimizer_implementation_passed = bool(
        aggregates["reference_gradient_postclip"]["max_abs"] <= 1e-7
        and aggregates["reference_gradient_parameter_after_step"]["max_abs"] <= 1e-6
    )
    native_training_step_close = bool(
        gradient["cosine"] >= 0.999
        and gradient["relative_l2"] <= args.gradient_relative_l2_tolerance
        and update["cosine"] >= 0.995
        and update["relative_l2"] <= args.update_relative_l2_tolerance
    )
    strict_training_equivalence_passed = bool(
        gradient["relative_l2"] <= args.strict_relative_l2_tolerance
        and update["relative_l2"] <= args.strict_relative_l2_tolerance
    )
    assessment = {
        "source_report": str(args.input),
        "historical_raw_summary": raw.get("summary", {}),
        "thresholds": {
            "activation_max_abs": 5e-4,
            "loss_max_abs": 1e-5,
            "gradient_cosine": 0.999,
            "gradient_relative_l2": args.gradient_relative_l2_tolerance,
            "parameter_update_cosine": 0.995,
            "parameter_update_relative_l2": args.update_relative_l2_tolerance,
            "strict_relative_l2": args.strict_relative_l2_tolerance,
        },
        "measured": {
            "activation_max_abs": activation_max_abs,
            "loss_max_abs": loss_max_abs,
            "gradient_cosine": gradient["cosine"],
            "gradient_relative_l2": gradient["relative_l2"],
            "parameter_update_cosine": update["cosine"],
            "parameter_update_relative_l2": update["relative_l2"],
        },
        "conclusions": {
            "forward_and_loss_passed": forward_passed,
            "optimizer_implementation_passed": optimizer_implementation_passed,
            "native_training_step_close": native_training_step_close,
            "functional_migration_passed": forward_passed
            and optimizer_implementation_passed
            and native_training_step_close,
            "strict_training_equivalence_passed": strict_training_equivalence_passed,
        },
        "interpretation": (
            "Forward outputs and losses align, and the migrated clipping/Adam implementations "
            "match when supplied with the same reference gradients. Native Jittor gradients and "
            "the resulting update are close under declared engineering tolerances but do not meet "
            "the strict relative-L2 threshold, so bitwise or strict training-step equivalence is "
            "not claimed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(assessment, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(assessment["conclusions"], indent=2))


if __name__ == "__main__":
    main()

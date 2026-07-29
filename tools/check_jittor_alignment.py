#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

import jittor as jt
import numpy as np


def synchronize(stage):
    print(f"[alignment] {stage}: start", flush=True)
    jt.sync_all()
    print(f"[alignment] {stage}: done", flush=True)


def forward_with_features(model, infrared, visible):
    features = {}
    infrared_raw = infrared
    infrared_raw_invert = 1 - infrared
    visible_raw = visible
    visible_raw_invert = 1 - visible

    infrared = model.ir_conv(infrared)
    visible = model.vi_conv(visible)
    features["ir_conv"] = infrared
    features["vi_conv"] = visible

    infrared_sa = infrared
    visible_sa = visible
    for index, (infrared_layer, visible_layer) in enumerate(zip(model.ir_sa, model.vi_sa)):
        infrared_sa = infrared_layer(infrared_sa)
        visible_sa = visible_layer(visible_sa)
        features[f"ir_sa_{index}"] = infrared_sa
        features[f"vi_sa_{index}"] = visible_sa

    weight_ir = model.weight_ir(infrared_raw)
    weight_ir_invert = model.weight_irI(infrared_raw_invert)
    weight_vi = model.weight_vi(visible_raw)
    weight_vi_invert = model.weight_viI(visible_raw_invert)
    features["weight_ir"] = weight_ir
    features["weight_irI"] = weight_ir_invert
    features["weight_vi"] = weight_vi
    features["weight_viI"] = weight_vi_invert

    infrared_to_visible = visible_sa
    infrared_invert_to_visible = visible_sa
    for index, (infrared_layer, infrared_invert_layer) in enumerate(
        zip(model.ir2vi_ca, model.irI2vi_ca)
    ):
        infrared_to_visible = infrared_layer(infrared_to_visible, weight_ir)
        infrared_invert_to_visible = infrared_invert_layer(
            infrared_invert_to_visible, weight_ir_invert
        )
        features[f"ir2vi_ca_{index}"] = infrared_to_visible
        features[f"irI2vi_ca_{index}"] = infrared_invert_to_visible

    visible_to_infrared = infrared_sa
    visible_invert_to_infrared = infrared_sa
    for index, (visible_layer, visible_invert_layer) in enumerate(
        zip(model.vi2ir_ca, model.viI2ir_ca)
    ):
        visible_to_infrared = visible_layer(visible_to_infrared, weight_vi)
        visible_invert_to_infrared = visible_invert_layer(
            visible_invert_to_infrared, weight_vi_invert
        )
        features[f"vi2ir_ca_{index}"] = visible_to_infrared
        features[f"viI2ir_ca_{index}"] = visible_invert_to_infrared

    mixed = jt.concat(
        [
            infrared_to_visible,
            visible_to_infrared,
            infrared_invert_to_visible,
            visible_invert_to_infrared,
        ],
        dim=1,
    )
    features["mixed"] = mixed
    mixed = model.fuse_conv(mixed)
    features["fuse_conv"] = mixed
    output = model.out_conv(mixed)
    features["output"] = output
    return output, features


def statistics(reference, actual):
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    difference = np.abs(reference - actual)
    denominator = np.maximum(np.abs(reference), 1e-12)
    reference_norm = np.linalg.norm(reference)
    actual_norm = np.linalg.norm(actual)
    cosine = 1.0
    if reference_norm > 0 and actual_norm > 0:
        cosine = float(np.dot(reference, actual) / (reference_norm * actual_norm))
    return {
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean() if difference.size else 0.0),
        "max_rel": float((difference / denominator).max(initial=0.0)),
        "cosine": cosine,
    }


def vector_statistics(reference_values, actual_values):
    reference = np.concatenate(
        [np.asarray(value, dtype=np.float64).reshape(-1) for value in reference_values]
    )
    actual = np.concatenate(
        [np.asarray(value, dtype=np.float64).reshape(-1) for value in actual_values]
    )
    result = statistics(reference, actual)
    reference_norm = float(np.linalg.norm(reference))
    actual_norm = float(np.linalg.norm(actual))
    result.update(
        {
            "reference_l2": reference_norm,
            "actual_l2": actual_norm,
            "relative_l2": float(
                np.linalg.norm(reference - actual) / max(reference_norm, 1e-12)
            ),
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--max-abs-tolerance", type=float, default=5e-4)
    parser.add_argument("--gradient-relative-l2-tolerance", type=float, default=2e-2)
    parser.add_argument("--update-relative-l2-tolerance", type=float, default=8e-2)
    parser.add_argument("--strict-relative-l2-tolerance", type=float, default=1e-3)
    parser.add_argument("--require-strict-training-equivalence", action="store_true")
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    source = args.project_root / "siba_jittor"
    sys.path.insert(0, str(source))
    from compat.pytorch_adam import PyTorchAdam
    from compat.pytorch_clip import clip_grad_norm_pytorch
    from loss.loss import Fusionloss, JointGrad
    from models.SIBA import SIBA

    reference = np.load(args.reference)
    model = SIBA()
    checkpoint = jt.load(str(args.checkpoint))
    checkpoint_names = set(checkpoint["model"].keys())
    model_names = {name for name, _ in model.named_parameters()}
    missing_names = sorted(model_names - checkpoint_names)
    unexpected_names = sorted(checkpoint_names - model_names)
    if missing_names or unexpected_names:
        raise RuntimeError(
            f"Checkpoint key mismatch: missing={missing_names}, unexpected={unexpected_names}"
        )
    model.load_parameters(checkpoint["model"])
    model.train()
    initial_parameters = {
        name: parameter.numpy().copy() for name, parameter in model.named_parameters()
    }

    infrared = jt.array(reference["input_ir"])
    visible = jt.array(reference["input_vi"])
    output, features = forward_with_features(model, infrared, visible)
    direct_output = model(infrared, visible)
    synchronize("forward activations")

    report = {
        "framework": "Jittor",
        "jittor_version": jt.__version__,
        "use_cuda": bool(args.use_cuda),
        "checkpoint_parameter_keys": len(checkpoint_names),
        "model_parameter_keys": len(model_names),
        "checks": {},
        "aggregates": {},
    }
    report["checks"]["manual_vs_model_output"] = statistics(
        output.numpy(), direct_output.numpy()
    )
    for name, parameter in initial_parameters.items():
        report["checks"][f"parameter_initial__{name}"] = statistics(
            reference[f"parameter_initial__{name}"], parameter
        )
    for name, value in features.items():
        report["checks"][f"activation__{name}"] = statistics(
            reference[f"activation__{name}"], value.numpy()
        )

    joint_grad = JointGrad()
    fusion_loss = Fusionloss()
    loss_laplacian = joint_grad(output, infrared, visible)
    loss_intensity, loss_sobel = fusion_loss(output, infrared, visible)
    loss_total = 10 * loss_laplacian + 0.1 * loss_intensity + loss_sobel
    synchronize("losses")
    for name, value in {
        "loss_laplacian": loss_laplacian,
        "loss_intensity": loss_intensity,
        "loss_sobel": loss_sobel,
        "loss_total": loss_total,
    }.items():
        report["checks"][name] = statistics(reference[name], value.numpy())

    optimizer = PyTorchAdam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    optimizer.zero_grad()
    optimizer.backward(loss_total)
    named_parameters = dict(model.named_parameters())
    gradients = optimizer.param_groups[0]["grads"]
    parameter_names = [name for name, _ in model.named_parameters()]
    synchronize("pre-clip gradients")
    actual_preclip_gradients = [gradient.numpy() for gradient in gradients]
    for name, gradient in zip(parameter_names, gradients):
        report["checks"][f"gradient_preclip__{name}"] = statistics(
            reference[f"gradient_preclip__{name}"], gradient.numpy()
        )
    report["aggregates"]["gradient_preclip"] = vector_statistics(
        [reference[f"gradient_preclip__{name}"] for name in parameter_names],
        actual_preclip_gradients,
    )

    parameter_norms = jt.stack(
        [jt.norm(gradient.flatten(), 2) for gradient in gradients]
    )
    total_norm = float(jt.norm(parameter_norms.flatten(), 2).item())
    report["checks"]["clip_total_norm"] = statistics(
        reference["clip_total_norm"], np.asarray(total_norm)
    )
    clip_grad_norm_pytorch(optimizer, max_norm=0.01, norm_type=2)
    synchronize("post-clip gradients")
    actual_postclip_gradients = [gradient.numpy() for gradient in gradients]
    for name, gradient in zip(parameter_names, gradients):
        report["checks"][f"gradient_postclip__{name}"] = statistics(
            reference[f"gradient_postclip__{name}"], gradient.numpy()
        )
    report["aggregates"]["gradient_postclip"] = vector_statistics(
        [reference[f"gradient_postclip__{name}"] for name in parameter_names],
        actual_postclip_gradients,
    )

    optimizer.step()
    synchronize("Adam step")
    actual_parameter_updates = []
    for name, parameter in named_parameters.items():
        parameter_after_step = parameter.numpy()
        actual_parameter_update = parameter_after_step - initial_parameters[name]
        report["checks"][f"parameter_after_step__{name}"] = statistics(
            reference[f"parameter_after_step__{name}"], parameter_after_step
        )
        report["checks"][f"parameter_update__{name}"] = statistics(
            reference[f"parameter_update__{name}"],
            actual_parameter_update,
        )
        actual_parameter_updates.append(actual_parameter_update)
    report["aggregates"]["parameter_update"] = vector_statistics(
        [reference[f"parameter_update__{name}"] for name in parameter_names],
        actual_parameter_updates,
    )

    reference_path_model = SIBA()
    reference_path_model.load_parameters(checkpoint["model"])
    reference_path_optimizer = PyTorchAdam(
        reference_path_model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )
    reference_path_gradients = [
        jt.array(reference[f"gradient_preclip__{name}"]).stop_grad()
        for name in parameter_names
    ]
    reference_path_optimizer.param_groups[0]["grads"] = reference_path_gradients
    reference_path_optimizer._Optimizer__zero_grad = False
    reference_path_optimizer.n_step = 1
    clip_grad_norm_pytorch(reference_path_optimizer, max_norm=0.01, norm_type=2)
    synchronize("reference-gradient clip")
    reference_path_postclip = [gradient.numpy() for gradient in reference_path_gradients]
    report["aggregates"]["reference_gradient_postclip"] = vector_statistics(
        [reference[f"gradient_postclip__{name}"] for name in parameter_names],
        reference_path_postclip,
    )
    reference_path_optimizer.step()
    synchronize("reference-gradient Adam step")
    reference_path_after_step = [
        parameter.numpy() for parameter in reference_path_model.parameters()
    ]
    report["aggregates"]["reference_gradient_parameter_after_step"] = vector_statistics(
        [reference[f"parameter_after_step__{name}"] for name in parameter_names],
        reference_path_after_step,
    )

    activation_max_abs = max(
        value["max_abs"]
        for name, value in report["checks"].items()
        if name.startswith("activation__")
    )
    loss_max_abs = max(
        report["checks"][name]["max_abs"]
        for name in ("loss_laplacian", "loss_intensity", "loss_sobel", "loss_total")
    )
    gradient_relative_l2 = report["aggregates"]["gradient_preclip"]["relative_l2"]
    update_relative_l2 = report["aggregates"]["parameter_update"]["relative_l2"]
    forward_passed = bool(
        report["checks"]["manual_vs_model_output"]["max_abs"] <= 1e-6
        and activation_max_abs <= args.max_abs_tolerance
        and loss_max_abs <= 1e-5
    )
    optimizer_implementation_passed = bool(
        report["aggregates"]["reference_gradient_postclip"]["max_abs"] <= 1e-7
        and report["aggregates"]["reference_gradient_parameter_after_step"]["max_abs"]
        <= 1e-6
    )
    native_training_step_close = bool(
        report["aggregates"]["gradient_preclip"]["cosine"] >= 0.999
        and gradient_relative_l2 <= args.gradient_relative_l2_tolerance
        and report["aggregates"]["parameter_update"]["cosine"] >= 0.995
        and update_relative_l2 <= args.update_relative_l2_tolerance
    )
    strict_training_equivalence_passed = bool(
        gradient_relative_l2 <= args.strict_relative_l2_tolerance
        and update_relative_l2 <= args.strict_relative_l2_tolerance
    )
    report["summary"] = {
        "check_count": len(report["checks"]),
        "activation_max_abs": activation_max_abs,
        "loss_max_abs": loss_max_abs,
        "gradient_preclip_cosine": report["aggregates"]["gradient_preclip"]["cosine"],
        "gradient_preclip_relative_l2": gradient_relative_l2,
        "parameter_update_cosine": report["aggregates"]["parameter_update"]["cosine"],
        "parameter_update_relative_l2": update_relative_l2,
        "reference_gradient_postclip_max_abs": report["aggregates"]["reference_gradient_postclip"]["max_abs"],
        "reference_gradient_parameter_after_step_max_abs": report["aggregates"]["reference_gradient_parameter_after_step"]["max_abs"],
        "forward_and_loss_passed": forward_passed,
        "optimizer_implementation_passed": optimizer_implementation_passed,
        "native_training_step_close": native_training_step_close,
        "strict_training_equivalence_passed": strict_training_equivalence_passed,
    }
    report["summary"]["functional_migration_passed"] = bool(
        forward_passed and optimizer_implementation_passed and native_training_step_close
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    required_pass = (
        strict_training_equivalence_passed
        if args.require_strict_training_equivalence
        else report["summary"]["functional_migration_passed"]
    )
    if not required_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


@contextmanager
def allow_official_loss_on_cpu(device):
    if device.type != "cpu":
        yield
        return

    original_cuda = torch.Tensor.cuda
    torch.Tensor.cuda = lambda tensor, *args, **kwargs: tensor
    try:
        yield
    finally:
        torch.Tensor.cuda = original_cuda


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

    mixed = torch.cat(
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


def numpy(tensor):
    return tensor.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    source = args.project_root / "official_pytorch"
    sys.path.insert(0, str(source))
    from loss.loss import Fusionloss, JointGrad
    from models.SIBA import SIBA

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    model = SIBA().to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.train()

    random = np.random.RandomState(args.seed)
    infrared_numpy = random.rand(args.batch_size, 1, args.height, args.width).astype(np.float32)
    visible_numpy = random.rand(args.batch_size, 1, args.height, args.width).astype(np.float32)
    infrared = torch.from_numpy(infrared_numpy).to(device)
    visible = torch.from_numpy(visible_numpy).to(device)

    output, features = forward_with_features(model, infrared, visible)
    direct_output = model(infrared, visible)
    torch.testing.assert_close(output, direct_output, rtol=0, atol=0)

    with allow_official_loss_on_cpu(device):
        joint_grad = JointGrad().to(device)
        fusion_loss = Fusionloss().to(device)
    loss_laplacian = joint_grad(output, infrared, visible)
    loss_intensity, loss_sobel = fusion_loss(output, infrared, visible)
    loss_total = 10 * loss_laplacian + 0.1 * loss_intensity + loss_sobel

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    optimizer.zero_grad()
    loss_total.backward()

    arrays = {
        "input_ir": infrared_numpy,
        "input_vi": visible_numpy,
        "loss_laplacian": np.asarray(loss_laplacian.item(), dtype=np.float64),
        "loss_intensity": np.asarray(loss_intensity.item(), dtype=np.float64),
        "loss_sobel": np.asarray(loss_sobel.item(), dtype=np.float64),
        "loss_total": np.asarray(loss_total.item(), dtype=np.float64),
    }
    initial_parameters = {}
    for name, parameter in model.named_parameters():
        initial_parameters[name] = numpy(parameter).copy()
        arrays[f"parameter_initial__{name}"] = initial_parameters[name]
    for name, value in features.items():
        arrays[f"activation__{name}"] = numpy(value)
    for name, parameter in model.named_parameters():
        arrays[f"gradient_preclip__{name}"] = numpy(parameter.grad)

    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01, norm_type=2)
    arrays["clip_total_norm"] = np.asarray(total_norm.item(), dtype=np.float64)
    for name, parameter in model.named_parameters():
        arrays[f"gradient_postclip__{name}"] = numpy(parameter.grad)

    optimizer.step()
    for name, parameter in model.named_parameters():
        parameter_after_step = numpy(parameter)
        arrays[f"parameter_after_step__{name}"] = parameter_after_step
        arrays[f"parameter_update__{name}"] = (
            parameter_after_step - initial_parameters[name]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        "framework": "PyTorch",
        "torch_version": torch.__version__,
        "device": str(device),
        "checkpoint": str(args.checkpoint.resolve()),
        "seed": args.seed,
        "shape": [args.batch_size, 1, args.height, args.width],
        "arrays": len(arrays),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

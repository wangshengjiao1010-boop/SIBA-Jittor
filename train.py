import argparse
import csv
import datetime
import hashlib
import json
import os
import pathlib
import random
import sys
import time

import jittor as jt
import numpy as np
from jittor.lr_scheduler import StepLR

sys.path.append("./")

from args.args_SIBA import args
from compat.pytorch_adam import PyTorchAdam
from compat.pytorch_clip import clip_grad_norm_pytorch
from loader.train_loader import TrainLoader
from loss.loss import Fusionloss, JointGrad
from models.SIBA import SIBA


def parse_runtime_args():
    parser = argparse.ArgumentParser(description="Train SIBA with Jittor")
    parser.add_argument("--ir-path", default="datasets/train/ir")
    parser.add_argument("--vi-path", default="datasets/train/vi")
    parser.add_argument("--output", default="checkpoint")
    parser.add_argument("--gpu-number", default="0")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=args.epochs)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--initial-weights", type=pathlib.Path)
    parser.add_argument("--schedule", type=pathlib.Path)
    parser.add_argument("--run-name")
    parser.add_argument("--log-csv", type=pathlib.Path)
    parser.add_argument("--metadata", type=pathlib.Path)
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = np.ascontiguousarray(value.numpy())
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_initial_weights(model, path):
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npz":
        expected = model.state_dict()
        with np.load(str(path), allow_pickle=False) as archive:
            archive_keys = set(archive.files)
            expected_keys = set(expected)
            missing = sorted(expected_keys - archive_keys)
            unexpected = sorted(archive_keys - expected_keys)
            if missing or unexpected:
                raise ValueError("Parameter keys differ")
            parameters = {}
            for name, current in expected.items():
                array = archive[name]
                if tuple(array.shape) != tuple(current.shape):
                    raise ValueError(name)
                parameters[name] = jt.array(array)
        model.load_parameters(parameters)
    else:
        checkpoint = jt.load(str(path))
        parameters = checkpoint.get("model", checkpoint)
        model.load_parameters(parameters)


def main():
    runtime_args = parse_runtime_args()
    args.ir_path = runtime_args.ir_path
    args.vi_path = runtime_args.vi_path
    args.model_save_path = runtime_args.output
    args.use_gpu_number = runtime_args.gpu_number
    args.use_gpu = not runtime_args.cpu
    args.epochs = runtime_args.epochs

    os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu_number
    jt.flags.use_cuda = 1 if args.use_gpu else 0
    if runtime_args.seed is not None:
        random.seed(runtime_args.seed)
        np.random.seed(runtime_args.seed)
        jt.set_global_seed(runtime_args.seed)

    model_save_path = args.model_save_path
    num_epochs = args.epochs
    learning_rate = args.init_lr
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    clip_grad_norm_value = 0.01
    optim_step = args.optim_step
    optim_gamma = args.optim_gamma

    model = SIBA()
    if runtime_args.initial_weights is not None:
        load_initial_weights(model, runtime_args.initial_weights)
    initial_fingerprint = state_fingerprint(model)

    optimizer = PyTorchAdam(
        model.parameters(),
        lr=learning_rate,
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=optim_step, gamma=optim_gamma)
    joint_grad_loss = JointGrad()
    intensity_grad_loss = Fusionloss()

    data = TrainLoader(
        pathlib.Path(args.ir_path),
        pathlib.Path(args.vi_path),
        args.patch_size,
        schedule_path=runtime_args.schedule,
    )
    if data.scheduled and data.schedule_epochs < num_epochs:
        raise ValueError(
            f"Training schedule has {data.schedule_epochs} epochs, "
            f"but {num_epochs} were requested"
        )
    trainloader = data.set_attrs(
        batch_size=batch_size,
        shuffle=not data.scheduled,
        drop_last=False,
        num_workers=0,
    )
    batches_per_epoch = trainloader.__batch_len__()
    if batches_per_epoch <= 0:
        raise ValueError("Training loader contains no batches")

    previous_time = time.time()
    started_at = datetime.datetime.now(datetime.timezone.utc)
    started = time.perf_counter()
    csv_file = None
    csv_writer = None
    if runtime_args.log_csv is not None:
        runtime_args.log_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = runtime_args.log_csv.open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "global_step",
                "epoch",
                "batch",
                "learning_rate",
                "loss_total",
                "loss_joint_grad",
                "loss_intensity",
                "loss_sobel",
                "elapsed_seconds",
            ],
        )
        csv_writer.writeheader()

    try:
        global_step = 0
        for epoch in range(num_epochs):
            data.set_epoch(epoch)
            for batch, (infrared, visible) in enumerate(trainloader):
                model.train()
                optimizer.zero_grad()
                fused = model(infrared, visible)
                loss_joint = joint_grad_loss(fused, infrared, visible)
                loss_intensity, loss_sobel = intensity_grad_loss(
                    fused, infrared, visible
                )
                loss_total = 10 * loss_joint + 0.1 * loss_intensity + loss_sobel
                optimizer.backward(loss_total)
                clip_grad_norm_pytorch(
                    optimizer,
                    max_norm=clip_grad_norm_value,
                    norm_type=2,
                )
                optimizer.step()

                values = {
                    "loss_total": float(loss_total.item()),
                    "loss_joint_grad": float(loss_joint.item()),
                    "loss_intensity": float(loss_intensity.item()),
                    "loss_sobel": float(loss_sobel.item()),
                }
                elapsed_seconds = time.perf_counter() - started
                if csv_writer is not None:
                    csv_writer.writerow(
                        {
                            "global_step": global_step,
                            "epoch": epoch,
                            "batch": batch,
                            "learning_rate": optimizer.lr,
                            **values,
                            "elapsed_seconds": elapsed_seconds,
                        }
                    )

                batches_done = epoch * batches_per_epoch + batch
                batches_left = num_epochs * batches_per_epoch - batches_done
                eta = datetime.timedelta(
                    seconds=batches_left * (time.time() - previous_time)
                )
                previous_time = time.time()
                if batch % 50 == 0:
                    print(
                        "[Epoch {}/{}] [batch {}/{}] [lr {}] "
                        "[total {:.6f}] [joint {:.6f}] [intensity {:.6f}] "
                        "[sobel {:.6f}] ETA: {}".format(
                            epoch,
                            num_epochs,
                            batch,
                            batches_per_epoch,
                            optimizer.lr,
                            values["loss_total"],
                            values["loss_joint_grad"],
                            values["loss_intensity"],
                            values["loss_sobel"],
                            eta,
                        ),
                        flush=True,
                    )
                global_step += 1
            scheduler.step()
            if optimizer.lr < 1e-6:
                optimizer.lr = 1e-6
            if csv_file is not None:
                csv_file.flush()
    finally:
        if csv_file is not None:
            csv_file.close()

    run_name = runtime_args.run_name or datetime.datetime.now().strftime("%m-%d-%H-%M")
    run_directory = pathlib.Path(model_save_path) / run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    save_path = run_directory / f"SIBA_epoch{num_epochs}.pkl"
    jt.save({"model": model.state_dict()}, str(save_path))
    final_fingerprint = state_fingerprint(model)

    if runtime_args.metadata is not None:
        runtime_args.metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "framework": "Jittor",
            "jittor_version": jt.__version__,
            "seed": runtime_args.seed,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "patch_size": args.patch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "scheduler_step": optim_step,
            "scheduler_gamma": optim_gamma,
            "gradient_clip_norm": clip_grad_norm_value,
            "training_pairs": len(data),
            "batches_per_epoch": batches_per_epoch,
            "initial_weights": (
                str(runtime_args.initial_weights.resolve())
                if runtime_args.initial_weights is not None
                else None
            ),
            "initial_weights_sha256": (
                file_sha256(runtime_args.initial_weights)
                if runtime_args.initial_weights is not None
                else None
            ),
            "initial_state_sha256": initial_fingerprint,
            "final_state_sha256": final_fingerprint,
            "schedule": (
                str(runtime_args.schedule.resolve())
                if runtime_args.schedule is not None
                else None
            ),
            "schedule_sha256": (
                file_sha256(runtime_args.schedule)
                if runtime_args.schedule is not None
                else None
            ),
            "started_at_utc": started_at.isoformat(),
            "duration_seconds": time.perf_counter() - started,
            "checkpoint": str(save_path.resolve()),
            "checkpoint_sha256": file_sha256(save_path),
            "logged_batches": global_step,
            "batch_log": (
                str(runtime_args.log_csv.resolve())
                if runtime_args.log_csv is not None
                else None
            ),
        }
        runtime_args.metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print("done")


if __name__ == "__main__":
    main()

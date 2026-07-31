#!/usr/bin/env python3
import argparse
import csv
import datetime
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


SUFFIXES = {".png", ".jpg", ".bmp"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


class ScheduledTrainDataset(Dataset):
    def __init__(self, ir_path, vi_path, patch_size, schedule_path):
        self.ir_list = [
            item for item in sorted(ir_path.glob("*")) if item.suffix in SUFFIXES
        ]
        self.vi_list = [
            item for item in sorted(vi_path.glob("*")) if item.suffix in SUFFIXES
        ]
        if not self.ir_list or len(self.ir_list) != len(self.vi_list):
            raise ValueError(
                f"Invalid training counts: ir={len(self.ir_list)}, "
                f"vi={len(self.vi_list)}"
            )
        if [path.name for path in self.ir_list] != [
            path.name for path in self.vi_list
        ]:
            raise ValueError("Infrared and visible filenames do not match")
        self.patch_size = patch_size
        self.epoch = 0
        with np.load(str(schedule_path), allow_pickle=False) as schedule:
            self.indices = schedule["indices"].astype(np.int64, copy=True)
            self.crop_x = schedule["crop_x"].astype(np.int64, copy=True)
            self.crop_y = schedule["crop_y"].astype(np.int64, copy=True)
            expected = [str(name) for name in schedule["filenames"].tolist()]
        if expected != [path.name for path in self.ir_list]:
            raise ValueError("Training schedule filenames do not match the dataset")
        if not (self.indices.shape == self.crop_x.shape == self.crop_y.shape):
            raise ValueError("Training schedule arrays must have identical shapes")
        if self.indices.shape[1] != len(self.ir_list):
            raise ValueError("Training schedule sample count does not match the dataset")

    def set_epoch(self, epoch):
        if epoch < 0 or epoch >= self.indices.shape[0]:
            raise ValueError(f"Schedule does not contain epoch {epoch}")
        self.epoch = epoch

    def __len__(self):
        return len(self.ir_list)

    def __getitem__(self, position):
        index = int(self.indices[self.epoch, position])
        x = int(self.crop_x[self.epoch, position])
        y = int(self.crop_y[self.epoch, position])
        ir = cv2.imread(str(self.ir_list[index]), cv2.IMREAD_GRAYSCALE)
        vi = cv2.imread(str(self.vi_list[index]), cv2.IMREAD_GRAYSCALE)
        if ir is None or vi is None:
            raise ValueError(f"Invalid training image: {self.ir_list[index].name}")
        ir = np.expand_dims(ir / 255.0, axis=0).astype(np.float32)
        vi = np.expand_dims(vi / 255.0, axis=0).astype(np.float32)
        ir = ir[:, x : x + self.patch_size, y : y + self.patch_size]
        vi = vi[:, x : x + self.patch_size, y : y + self.patch_size]
        if ir.shape[1:] != (self.patch_size, self.patch_size):
            raise ValueError(f"Invalid scheduled crop at position {position}")
        return torch.from_numpy(ir), torch.from_numpy(vi)


def load_numpy_state(model, path):
    expected = model.state_dict()
    with np.load(str(path), allow_pickle=False) as archive:
        missing = sorted(set(expected) - set(archive.files))
        unexpected = sorted(set(archive.files) - set(expected))
        if missing or unexpected:
            raise ValueError(
                f"Initial-weight keys differ: missing={missing}, "
                f"unexpected={unexpected}"
            )
        state = {}
        for name, current in expected.items():
            array = archive[name]
            if tuple(array.shape) != tuple(current.shape):
                raise ValueError(f"Initial-weight shape mismatch for {name}")
            state[name] = torch.from_numpy(array).to(dtype=current.dtype)
    model.load_state_dict(state, strict=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run official SIBA training with controlled comparison inputs."
    )
    parser.add_argument("--pytorch-root", type=Path, required=True)
    parser.add_argument("--ir-path", type=Path, required=True)
    parser.add_argument("--vi-path", type=Path, required=True)
    parser.add_argument("--initial-weights", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-csv", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--gpu-number", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    pytorch_root = args.pytorch_root.resolve()
    sys.path.insert(0, str(pytorch_root))
    from loss.loss import Fusionloss, JointGrad
    from models.SIBA import SIBA

    model = SIBA()
    load_numpy_state(model, args.initial_weights)
    initial_fingerprint = state_fingerprint(model)
    model = model.cuda()

    dataset = ScheduledTrainDataset(
        args.ir_path,
        args.vi_path,
        args.patch_size,
        args.schedule,
    )
    if dataset.indices.shape[0] < args.epochs:
        raise ValueError("Training schedule has fewer epochs than requested")
    trainloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    joint_grad = JointGrad()
    fusion_loss = Fusionloss()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_csv.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.datetime.now(datetime.timezone.utc)
    started = time.perf_counter()
    previous = time.time()
    batches_per_epoch = len(trainloader)
    global_step = 0
    with args.log_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
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
        writer.writeheader()
        for epoch in range(args.epochs):
            dataset.set_epoch(epoch)
            for batch, (infrared, visible) in enumerate(trainloader):
                infrared = infrared.cuda(non_blocking=True)
                visible = visible.cuda(non_blocking=True)
                model.train()
                optimizer.zero_grad()
                fused = model(infrared, visible)
                loss_joint = joint_grad(fused, infrared, visible)
                loss_intensity, loss_sobel = fusion_loss(fused, infrared, visible)
                loss_total = 10 * loss_joint + 0.1 * loss_intensity + loss_sobel
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=0.01, norm_type=2
                )
                optimizer.step()

                values = {
                    "loss_total": float(loss_total.item()),
                    "loss_joint_grad": float(loss_joint.item()),
                    "loss_intensity": float(loss_intensity.item()),
                    "loss_sobel": float(loss_sobel.item()),
                }
                elapsed = time.perf_counter() - started
                writer.writerow(
                    {
                        "global_step": global_step,
                        "epoch": epoch,
                        "batch": batch,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        **values,
                        "elapsed_seconds": elapsed,
                    }
                )
                batches_done = epoch * batches_per_epoch + batch
                batches_left = args.epochs * batches_per_epoch - batches_done
                eta = datetime.timedelta(
                    seconds=batches_left * (time.time() - previous)
                )
                previous = time.time()
                if batch % 50 == 0:
                    print(
                        "[Epoch {}/{}] [batch {}/{}] [lr {}] "
                        "[total {:.6f}] [joint {:.6f}] [intensity {:.6f}] "
                        "[sobel {:.6f}] ETA: {}".format(
                            epoch,
                            args.epochs,
                            batch,
                            batches_per_epoch,
                            optimizer.param_groups[0]["lr"],
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
            if optimizer.param_groups[0]["lr"] < 1e-6:
                optimizer.param_groups[0]["lr"] = 1e-6
            csv_file.flush()

    torch.save({"model": model.state_dict()}, args.output)
    final_fingerprint = state_fingerprint(model)
    metadata = {
        "framework": "PyTorch",
        "torch_version": torch.__version__,
        "official_pytorch_root": str(pytorch_root),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "learning_rate": 1e-4,
        "weight_decay": 0,
        "scheduler_step": 25,
        "scheduler_gamma": 0.5,
        "gradient_clip_norm": 0.01,
        "training_pairs": len(dataset),
        "batches_per_epoch": batches_per_epoch,
        "initial_weights": str(args.initial_weights.resolve()),
        "initial_weights_sha256": file_sha256(args.initial_weights),
        "initial_state_sha256": initial_fingerprint,
        "final_state_sha256": final_fingerprint,
        "schedule": str(args.schedule.resolve()),
        "schedule_sha256": file_sha256(args.schedule),
        "started_at_utc": started_at.isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "checkpoint": str(args.output.resolve()),
        "checkpoint_sha256": file_sha256(args.output),
        "logged_batches": global_step,
        "batch_log": str(args.log_csv.resolve()),
    }
    args.metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("done")


if __name__ == "__main__":
    main()

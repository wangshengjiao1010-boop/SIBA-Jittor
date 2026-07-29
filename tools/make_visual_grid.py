#!/usr/bin/env python3
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def load_rgb(path):
    return Image.open(path).convert("RGB")


def fit(image, size):
    return ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)


def label_font(size):
    for path in (
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", type=Path, required=True)
    parser.add_argument("--vi-dir", type=Path, required=True)
    parser.add_argument("--jittor-dir", type=Path, required=True)
    parser.add_argument("--pytorch-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jittor-label", default="Jittor 60 epochs")
    parser.add_argument("--pytorch-label", default="PyTorch 60 epochs")
    parser.add_argument("--name", action="append", default=[])
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--cell-width", type=int, default=480)
    parser.add_argument("--cell-height", type=int, default=320)
    args = parser.parse_args()

    available = sorted(
        path.name
        for path in args.jittor_dir.iterdir()
        if path.is_file() and (args.pytorch_dir / path.name).exists()
    )
    names = args.name or available[: args.count]
    labels = ("Infrared", "Visible", args.jittor_label, args.pytorch_label)
    directories = (args.ir_dir, args.vi_dir, args.jittor_dir, args.pytorch_dir)
    margin = 20
    label_height = 42
    row_height = args.cell_height + label_height + margin
    canvas = Image.new(
        "RGB",
        (margin + 4 * (args.cell_width + margin), margin + len(names) * row_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = label_font(24)
    for row_index, name in enumerate(names):
        for column_index, (label, directory) in enumerate(zip(labels, directories)):
            image = fit(load_rgb(directory / name), (args.cell_width, args.cell_height))
            x = margin + column_index * (args.cell_width + margin)
            y = margin + row_index * row_height
            image_x = x + (args.cell_width - image.width) // 2
            image_y = y + (args.cell_height - image.height) // 2
            canvas.paste(image, (image_x, image_y))
            draw.text((x, y + args.cell_height + 7), f"{label} | {name}", fill="#222222", font=font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=95)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

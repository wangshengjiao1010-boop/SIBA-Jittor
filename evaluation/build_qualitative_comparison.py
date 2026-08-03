import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an infrared/visible/PyTorch/Jittor comparison grid."
    )
    parser.add_argument("--infrared", required=True)
    parser.add_argument("--visible", required=True)
    parser.add_argument("--pytorch", required=True)
    parser.add_argument("--jittor", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=8)
    return parser.parse_args()


def load_rgb(path):
    return Image.open(path).convert("RGB")


def best_difference_crop(reference, candidate, crop_size, stride):
    reference_array = np.asarray(reference, dtype=np.float32)
    candidate_array = np.asarray(candidate, dtype=np.float32)
    if reference_array.shape != candidate_array.shape:
        raise ValueError("PyTorch and Jittor images must have the same dimensions")

    difference = np.abs(reference_array - candidate_array).mean(axis=2)
    height, width = difference.shape
    crop_size = min(crop_size, height, width)
    best_score = -1.0
    best_xy = (0, 0)

    y_positions = list(range(0, height - crop_size + 1, stride))
    x_positions = list(range(0, width - crop_size + 1, stride))
    if y_positions[-1] != height - crop_size:
        y_positions.append(height - crop_size)
    if x_positions[-1] != width - crop_size:
        x_positions.append(width - crop_size)

    for y in y_positions:
        for x in x_positions:
            score = float(difference[y : y + crop_size, x : x + crop_size].mean())
            if score > best_score:
                best_score = score
                best_xy = (x, y)

    return (*best_xy, crop_size, crop_size), best_score, float(difference.mean())


def load_font(size):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def fit_image(image, target_width, target_height):
    scale = min(target_width / image.width, target_height / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def paste_centered(canvas, image, left, top, width, height):
    fitted = fit_image(image, width, height)
    x = left + (width - fitted.width) // 2
    y = top + (height - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return x, y, fitted.width, fitted.height


def main():
    args = parse_args()
    paths = {
        "Infrared": Path(args.infrared),
        "Visible": Path(args.visible),
        "PyTorch": Path(args.pytorch),
        "Jittor": Path(args.jittor),
    }
    images = {label: load_rgb(path) for label, path in paths.items()}
    bbox, local_difference, mean_difference = best_difference_crop(
        images["PyTorch"], images["Jittor"], args.crop_size, args.stride
    )

    full_images = {label: image.copy() for label, image in images.items()}
    x, y, width, height = bbox
    rectangle = (x, y, x + width, y + height)
    for label in ("PyTorch", "Jittor"):
        ImageDraw.Draw(full_images[label]).rectangle(
            rectangle, outline="#D7191C", width=5
        )

    crops = {
        label: image.crop((x, y, x + width, y + height))
        for label, image in images.items()
    }

    columns = 4
    cell_width = 430
    full_height = 300
    crop_height = 230
    label_height = 48
    margin = 28
    gap = 18
    canvas_width = margin * 2 + columns * cell_width + (columns - 1) * gap
    canvas_height = margin * 2 + label_height * 2 + full_height + crop_height + gap
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = load_font(28)
    note_font = load_font(22)

    for index, label in enumerate(paths):
        left = margin + index * (cell_width + gap)
        draw.text(
            (left + cell_width / 2, margin + label_height / 2),
            label,
            fill="#111111",
            font=label_font,
            anchor="mm",
        )
        paste_centered(
            canvas,
            full_images[label],
            left,
            margin + label_height,
            cell_width,
            full_height,
        )
        crop_top = margin + label_height + full_height + gap + label_height
        crop_box = paste_centered(
            canvas, crops[label], left, crop_top, cell_width, crop_height
        )
        draw.rectangle(
            (
                crop_box[0] - 2,
                crop_box[1] - 2,
                crop_box[0] + crop_box[2] + 2,
                crop_box[1] + crop_box[3] + 2,
            ),
            outline="#D7191C",
            width=4,
        )

    crop_label_y = margin + label_height + full_height + gap + label_height / 2
    draw.text(
        (margin, crop_label_y),
        "Local enlargement",
        fill="#111111",
        font=note_font,
        anchor="lm",
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    metadata = {
        "inputs": {label: str(path) for label, path in paths.items()},
        "crop_xywh": [x, y, width, height],
        "local_mean_abs_uint8": local_difference,
        "full_mean_abs_uint8": mean_difference,
        "output": str(output_path),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

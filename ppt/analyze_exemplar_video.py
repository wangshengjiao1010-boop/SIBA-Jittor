#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SEGMENTS = [
    (0.0, 31.0, "封面与目录"),
    (31.0, 79.0, "创新点动画"),
    (79.0, 217.0, "总体架构与模块图"),
    (217.0, 1776.0, "完整代码讲解"),
    (1776.0, 1908.0, "训练演示"),
    (1908.0, 1992.0, "迁移测试与W&B"),
    (1992.0, 2064.0, "推理演示"),
    (2064.0, 2172.0, "曲线与性能对比"),
    (2172.0, 2268.0, "定量与定性对比"),
    (2268.0, float("inf"), "结束页"),
]


def segment_name(seconds):
    for start, end, name in SEGMENTS:
        if start <= seconds < end:
            return name
    return "未分类"


def descriptor(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)


def mean_abs(left, right):
    return float(np.abs(left - right).mean())


def scan_video(video_path, sample_stride):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rows = []
    samples = []
    previous = None
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        current = descriptor(frame)
        previous_delta = 0.0 if previous is None else mean_abs(current, previous)
        rows.append((frame_index, frame_index / fps, previous_delta))
        if frame_index % sample_stride == 0:
            samples.append(
                {
                    "frame_index": frame_index,
                    "time_seconds": frame_index / fps,
                    "previous_frame_delta": previous_delta,
                    "descriptor": current,
                }
            )
        previous = current
        frame_index += 1
    capture.release()
    return {
        "fps": fps,
        "declared_frame_count": frame_count,
        "decoded_frame_count": frame_index,
        "width": width,
        "height": height,
        "duration_seconds": frame_index / fps,
        "rows": rows,
        "samples": samples,
    }


def select_keyframes(samples, slide_end, change_threshold, stable_threshold):
    selected = []
    last_descriptor = None
    last_time = -1e9
    previous_sample_delta = 0.0
    for index, sample in enumerate(samples):
        current = sample["descriptor"]
        seconds = sample["time_seconds"]
        if last_descriptor is None:
            reason = "first_frame"
        else:
            delta_from_last = mean_abs(current, last_descriptor)
            sample_delta = 0.0
            if index:
                sample_delta = mean_abs(current, samples[index - 1]["descriptor"])
            stable_after_change = (
                previous_sample_delta >= stable_threshold
                and sample_delta < stable_threshold * 0.45
                and delta_from_last >= change_threshold * 0.55
                and seconds - last_time >= 0.45
            )
            changed = (
                delta_from_last >= change_threshold
                and seconds - last_time >= 0.45
            )
            maximum_interval = 8.0 if seconds >= slide_end else 20.0
            interval_sample = seconds - last_time >= maximum_interval
            if changed:
                reason = "visual_change"
            elif stable_after_change:
                reason = "stable_state"
            elif interval_sample:
                reason = "interval_guard"
            else:
                previous_sample_delta = sample_delta
                continue
        selected.append(
            {
                "frame_index": sample["frame_index"],
                "time_seconds": seconds,
                "reason": reason,
                "segment": segment_name(seconds),
            }
        )
        last_descriptor = current
        last_time = seconds
        previous_sample_delta = 0.0
    return selected


def save_keyframes(video_path, selected, output_dir):
    frame_dir = output_dir / "fullres_significant_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    selected_by_frame = {item["frame_index"]: item for item in selected}
    capture = cv2.VideoCapture(str(video_path))
    written = []
    frame_index = 0
    while selected_by_frame:
        ok, frame = capture.read()
        if not ok:
            break
        item = selected_by_frame.pop(frame_index, None)
        if item is not None:
            seconds = item["time_seconds"]
            minutes = int(seconds // 60)
            remainder = seconds - minutes * 60
            name = (
                f"{len(written) + 1:04d}_{minutes:02d}m{remainder:05.2f}s_"
                f"{item['reason']}.jpg"
            )
            path = frame_dir / name
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
            item = dict(item)
            item["file"] = str(path.relative_to(output_dir)).replace("\\", "/")
            written.append(item)
        frame_index += 1
    capture.release()
    return written


def load_font(size, bold=False):
    candidates = [
        Path("C:/Windows/Fonts/Noto Sans SC Bold (TrueType).otf") if bold else Path("C:/Windows/Fonts/Noto Sans SC (TrueType).otf"),
        Path("C:/Windows/Fonts/msyhbd.ttc") if bold else Path("C:/Windows/Fonts/msyh.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def build_contact_sheets(output_dir, written, columns=3, rows=3):
    sheet_dir = output_dir / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    cell_w, cell_h, header_h = 640, 360, 34
    per_sheet = columns * rows
    font = load_font(21)
    sheets = []
    for sheet_index in range(math.ceil(len(written) / per_sheet)):
        chunk = written[sheet_index * per_sheet : (sheet_index + 1) * per_sheet]
        canvas = Image.new(
            "RGB",
            (cell_w * columns, (cell_h + header_h) * rows),
            (24, 24, 24),
        )
        draw = ImageDraw.Draw(canvas)
        for index, item in enumerate(chunk):
            image_path = output_dir / item["file"]
            screenshot = Image.open(image_path).convert("RGB")
            screenshot.thumbnail((cell_w, cell_h))
            column = index % columns
            row = index // columns
            x = column * cell_w
            y = row * (cell_h + header_h)
            canvas.paste(screenshot, (x, y + header_h))
            label = (
                f"{sheet_index * per_sheet + index + 1:04d}  "
                f"{item['time_seconds']:07.2f}s  {item['segment']}  {item['reason']}"
            )
            draw.text((x + 8, y + 5), label, fill=(245, 245, 245), font=font)
        path = sheet_dir / f"contact_{sheet_index + 1:03d}.jpg"
        canvas.save(path, quality=92)
        sheets.append(str(path.relative_to(output_dir)).replace("\\", "/"))
    return sheets


def write_outputs(output_dir, scan, written, sheets):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "all_frame_diff.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "time_seconds", "delta_from_previous_frame"])
        writer.writerows(scan["rows"])
    public_scan = {
        key: value
        for key, value in scan.items()
        if key not in {"rows", "samples"}
    }
    public_scan["sample_interval_seconds"] = 0.5
    public_scan["selected_keyframe_count"] = len(written)
    public_scan["segments"] = [
        {"start": start, "end": None if math.isinf(end) else end, "name": name}
        for start, end, name in SEGMENTS
    ]
    (output_dir / "video_scan_summary.json").write_text(
        json.dumps(public_scan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "significant_frames.json").write_text(
        json.dumps(written, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# 优秀范例视频全帧复查结果",
                "",
                f"- 解码并检查全部 {scan['decoded_frame_count']:,} 帧。",
                f"- 视频分辨率：{scan['width']}×{scan['height']}，帧率：{scan['fps']:.2f} FPS。",
                f"- 视频时长：{scan['duration_seconds'] / 60:.2f} 分钟。",
                f"- 显著状态截图：{len(written)} 张。",
                f"- 联系表：{len(sheets)} 张。",
                "- `all_frame_diff.csv` 保存每一帧与前一帧的视觉差异，证明检查范围覆盖完整视频。",
                "- `significant_frames.json` 保存每张截图的原始帧号、时间、章节与选择原因。",
                "- `fullres_significant_frames/` 保存 1920×1080 截图。",
                "- `contact_sheets/` 用于快速逐页检查静态页、动画状态和现场演示过程。",
                "",
                "截图只用于分析优秀范例的讲解顺序和排版，不作为SIBA实验结果。",
            ]
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-seconds", type=float, default=0.5)
    parser.add_argument("--change-threshold", type=float, default=0.92)
    parser.add_argument("--stable-threshold", type=float, default=0.38)
    parser.add_argument("--slide-end", type=float, default=1776.0)
    args = parser.parse_args()

    probe = cv2.VideoCapture(str(args.video))
    if not probe.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = float(probe.get(cv2.CAP_PROP_FPS))
    probe.release()
    sample_stride = max(1, int(round(fps * args.sample_seconds)))
    scan = scan_video(args.video, sample_stride)
    selected = select_keyframes(
        scan["samples"],
        args.slide_end,
        args.change_threshold,
        args.stable_threshold,
    )
    written = save_keyframes(args.video, selected, args.output)
    sheets = build_contact_sheets(args.output, written)
    write_outputs(args.output, scan, written, sheets)
    print(json.dumps({
        "decoded_frames": scan["decoded_frame_count"],
        "selected_frames": len(written),
        "contact_sheets": len(sheets),
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

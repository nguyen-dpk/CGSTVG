import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


GREEN = (60, 220, 60)
RED = (40, 40, 230)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clamp_box_xyxy(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(0, min(int(round(x2)), width - 1))
    y2 = max(0, min(int(round(y2)), height - 1))
    return [x1, y1, x2, y2]


def xywh_to_xyxy(box, width, height):
    x1, y1, w, h = box
    return clamp_box_xyxy([x1, y1, x1 + w, y1 + h], width, height)


def raw_to_xyxy(box, width, height):
    x1, y1, x2, y2 = box
    return clamp_box_xyxy([x1, y1, x2, y2], width, height)


def draw_labeled_box(frame, box, color, label):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    )
    top = max(0, y1 - text_h - baseline - 6)
    right = min(frame.shape[1] - 1, x1 + text_w + 8)
    cv2.rectangle(frame, (x1, top), (right, y1), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 4, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        BLACK,
        2,
        cv2.LINE_AA,
    )


def add_footer(frame, footer_lines):
    line_height = 24
    footer_h = 14 + line_height * len(footer_lines)
    canvas = np.zeros((frame.shape[0] + footer_h, frame.shape[1], 3), dtype=np.uint8)
    canvas[: frame.shape[0]] = frame
    canvas[frame.shape[0] :] = 25

    y = frame.shape[0] + 24
    for line in footer_lines:
        cv2.putText(
            canvas,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            WHITE,
            2,
            cv2.LINE_AA,
        )
        y += line_height
    return canvas


def read_frame(video_path: Path, frame_index: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def make_panel(frame, raw_box, width, height, frame_id, sample_name):
    panel = frame.copy()
    xywh_box = xywh_to_xyxy(raw_box, width, height)
    xyxy_box = raw_to_xyxy(raw_box, width, height)

    draw_labeled_box(panel, xywh_box, GREEN, "as xywh")
    draw_labeled_box(panel, xyxy_box, RED, "as xyxy")

    footer_lines = [
        f"video={sample_name} frame={frame_id}",
        f"raw bbox={raw_box}",
        f"green: xywh -> {xywh_box} | red: xyxy -> {xyxy_box}",
    ]
    return add_footer(panel, footer_lines)


def stack_panels_horizontally(panels):
    max_h = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] == max_h:
            padded.append(panel)
            continue
        canvas = np.zeros((max_h, panel.shape[1], 3), dtype=np.uint8)
        canvas[:] = 20
        canvas[: panel.shape[0], : panel.shape[1]] = panel
        padded.append(canvas)
    return np.concatenate(padded, axis=1)


def choose_frame_offsets(num_boxes):
    if num_boxes == 1:
        return [0]
    if num_boxes == 2:
        return [0, 1]
    return [0, num_boxes // 2, num_boxes - 1]


def resolve_video_path(video_root: Path, video_name: str):
    path = video_root / video_name
    if path.exists():
        return path
    raise FileNotFoundError(f"Video file not found: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Draw a few HC-STVG2 samples to inspect whether raw bbox values are xywh or xyxy."
    )
    parser.add_argument(
        "--anno",
        default="data/hc-stvg2/annos/hcstvg_v2/test.json",
        help="Path to the converted HC-STVG2 test.json file",
    )
    parser.add_argument(
        "--video-root",
        default="data/hc-stvg2/v2_video",
        help="Directory containing the HC-STVG2 video files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/hc-stvg2/sample_visualizations",
        help="Directory to save rendered sample images",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="How many samples to render",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start offset into the annotation list",
    )
    args = parser.parse_args()

    anno_path = Path(args.anno)
    video_root = Path(args.video_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    anno_data = load_json(anno_path)
    items = list(anno_data.items())
    selected = items[args.start_index : args.start_index + args.num_samples]

    if not selected:
        raise ValueError("No samples selected. Check --start-index and --num-samples.")

    for sample_idx, (video_name, record) in enumerate(selected, start=1):
        width = int(record["width"])
        height = int(record["height"])
        st_frame = int(record["st_frame"])
        bbox_seq = record["bbox"]
        video_path = resolve_video_path(video_root, video_name)

        panels = []
        for offset in choose_frame_offsets(len(bbox_seq)):
            frame_id = st_frame + offset
            frame = read_frame(video_path, frame_id)
            raw_box = bbox_seq[offset]
            panels.append(make_panel(frame, raw_box, width, height, frame_id, video_name))

        collage = stack_panels_horizontally(panels)
        title = np.zeros((48, collage.shape[1], 3), dtype=np.uint8)
        title[:] = 15
        cv2.putText(
            title,
            f"Sample {sample_idx}: {video_name}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            WHITE,
            2,
            cv2.LINE_AA,
        )
        collage = np.concatenate([title, collage], axis=0)

        out_path = output_dir / f"sample_{sample_idx:02d}_{Path(video_name).stem}.jpg"
        cv2.imwrite(str(out_path), collage)
        print(f"Saved {out_path}")

    print()
    print("Interpretation guide:")
    print("  Green box  = interpret raw bbox as [x1, y1, w, h]")
    print("  Red box    = interpret raw bbox as [x1, y1, x2, y2]")
    print("If green consistently fits the object and red does not, the source format is xywh.")


if __name__ == "__main__":
    main()

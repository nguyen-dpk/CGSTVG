import argparse
import json
from pathlib import Path

import cv2


GREEN = (60, 220, 60)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clamp_xyxy(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), max(width - 1, 0)))
    y1 = max(0, min(int(round(y1)), max(height - 1, 0)))
    x2 = max(0, min(int(round(x2)), max(width - 1, 0)))
    y2 = max(0, min(int(round(y2)), max(height - 1, 0)))
    return [x1, y1, x2, y2]


def xywh_to_xyxy(box, width, height):
    x1, y1, w, h = box
    return clamp_xyxy([x1, y1, x1 + w, y1 + h], width, height)


def draw_text_block(frame, lines):
    line_height = 24
    pad = 10
    box_height = pad * 2 + line_height * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (frame.shape[1] - 8, 8 + box_height), (20, 20, 20), -1)
    frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    y = 8 + pad + 18
    for line in lines:
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            WHITE,
            2,
            cv2.LINE_AA,
        )
        y += line_height


def draw_bbox(frame, box):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
    label = "bbox"
    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    )
    top = max(0, y1 - text_h - baseline - 6)
    right = min(frame.shape[1] - 1, x1 + text_w + 8)
    cv2.rectangle(frame, (x1, top), (right, y1), GREEN, -1)
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


def resolve_annotation(video_name, anno_paths):
    for anno_path in anno_paths:
        if not anno_path.exists():
            continue
        data = load_json(anno_path)
        if video_name in data:
            return anno_path, data[video_name]
    searched = ", ".join(str(path) for path in anno_paths)
    raise KeyError(f"{video_name} not found in annotations: {searched}")


def find_video_path(video_root, video_name):
    video_path = video_root / video_name
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    return video_path


def wrap_caption(text, width=60):
    words = text.split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines[:3]


def main():
    parser = argparse.ArgumentParser(
        description="Render a HC-STVG/HC-STVG2 video with annotation bounding boxes."
    )
    parser.add_argument("video_name", help="Video filename, e.g. 50_TM5MPJIq1Is.mkv")
    parser.add_argument(
        "--video-root",
        default="data/hc-stvg2/v2_video",
        help="Directory containing the video files",
    )
    parser.add_argument(
        "--anno",
        default=None,
        help="Optional single annotation JSON to use directly",
    )
    parser.add_argument(
        "--anno-paths",
        nargs="*",
        default=[
            "data/hc-stvg2/annos/hcstvg_v2/train.json",
            "data/hc-stvg2/annos/hcstvg_v2/test.json",
            "data/hc-stvg2/annos/train_v2.json",
            "data/hc-stvg2/annos/val_v2.json",
        ],
        help="Annotation files to search when --anno is not provided",
    )
    parser.add_argument(
        "--output-dir",
        default="data/hc-stvg2/visualized_videos",
        help="Directory to save the rendered video",
    )
    parser.add_argument(
        "--annotated-only",
        action="store_true",
        help="Only render the annotated moment instead of the full source video",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override output FPS. Defaults to source video FPS.",
    )
    args = parser.parse_args()

    video_root = Path(args.video_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    anno_paths = [Path(args.anno)] if args.anno else [Path(path) for path in args.anno_paths]
    anno_path, record = resolve_annotation(args.video_name, anno_paths)
    video_path = find_video_path(video_root, args.video_name)

    width = int(record["width"] if "width" in record else record["img_size"][1])
    height = int(record["height"] if "height" in record else record["img_size"][0])
    st_frame = int(record["st_frame"])
    bbox_seq = record["bbox"]
    end_frame = st_frame + len(bbox_seq) - 1
    caption = record.get("caption") or record.get("English") or record.get("Chinese") or ""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = args.fps if args.fps is not None else (source_fps if source_fps and source_fps > 0 else 25.0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = st_frame if args.annotated_only else 0
    stop_frame = end_frame if args.annotated_only else total_frames - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out_path = output_dir / f"{Path(args.video_name).stem}_vis.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {out_path}")

    caption_lines = wrap_caption(caption, width=62)
    frame_id = start_frame
    while frame_id <= stop_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))

        if st_frame <= frame_id <= end_frame:
            bbox_idx = frame_id - st_frame
            xyxy = xywh_to_xyxy(bbox_seq[bbox_idx], width, height)
            draw_bbox(frame, xyxy)

        overlay_lines = [
            f"video={args.video_name}",
            f"annotation={anno_path.name} frame={frame_id} annotated_range=[{st_frame}, {end_frame}]",
        ] + caption_lines
        draw_text_block(frame, overlay_lines)

        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

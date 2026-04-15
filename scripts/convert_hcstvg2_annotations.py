import argparse
import json
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


def dump_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def dump_text(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")

def clamp_xyxy(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), max(width - 1, 0)))
    y1 = max(0, min(int(round(y1)), max(height - 1, 0)))
    x2 = max(0, min(int(round(x2)), max(width - 1, 0)))
    y2 = max(0, min(int(round(y2)), max(height - 1, 0)))
    return [x1, y1, x2, y2]


def make_canvas(record):
    img_size = record.get("img_size") or [720, 1280, 3]
    height = int(img_size[0])
    width = int(img_size[1])
    return np.zeros((height, width, 3), dtype=np.uint8) + 20


def load_video_frame(video_path: Path, frame_id: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


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


def draw_negative_bbox_sample(video_name, box, record, box_idx, debug_dir, video_root):
    debug_dir.mkdir(parents=True, exist_ok=True)

    frame_id = int(record["st_frame"]) + box_idx
    video_path = video_root / video_name
    frame = load_video_frame(video_path, frame_id) if video_path.exists() else None
    if frame is None:
        frame = make_canvas(record)

    height, width = frame.shape[:2]
    x1, y1, w, h = box
    raw_box = [x1, y1, w, h]
    drawn_box = clamp_xyxy([x1, y1, x1 + w, y1 + h], width, height)
    fixed_box = clamp_xyxy([min(x1, x1 + w), min(y1, y1 + h), max(x1, x1 + w), max(y1, y1 + h)], width, height)

    panel = frame.copy()
    draw_labeled_box(panel, drawn_box, RED, "raw xywh")
    draw_labeled_box(panel, fixed_box, GREEN, "sorted corners")
    panel = add_footer(
        panel,
        [
            f"video={video_name} frame={frame_id} bbox_idx={box_idx}",
            f"raw bbox={raw_box}",
            "red: raw xywh | green: sorted corners for inspection",
        ],
    )

    out_path = debug_dir / f"{Path(video_name).stem}_bbox_{box_idx:04d}.jpg"
    cv2.imwrite(str(out_path), panel)
    return out_path


def validate_xywh(video_name, box, record, box_idx, debug_dir, video_root):
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"{video_name}: invalid bbox entry {box}")

    x1, y1, w, h = box
    # if w < 0 or h < 0:
    #     debug_path = draw_negative_bbox_sample(
    #         video_name, box, record, box_idx, debug_dir, video_root
    #     )
    #     raise ValueError(
    #         f"{video_name}: bbox has negative size {box}. Debug image saved to {debug_path}"
    #     )
    return [x1, y1, w, h]


def normalize_record(video_name, record, query_map, debug_dir, video_root):
    img_size = record.get("img_size")
    if not isinstance(img_size, list) or len(img_size) < 2:
        raise ValueError(f"{video_name}: missing or invalid img_size")

    height = int(img_size[0])
    width = int(img_size[1])

    caption = record.get("English") or query_map.get(video_name) or record.get("Chinese")
    if not caption:
        raise ValueError(f"{video_name}: missing caption text")

    bbox = record.get("bbox")
    if not isinstance(bbox, list) or not bbox:
        raise ValueError(f"{video_name}: missing bbox list")
    bbox = [
        validate_xywh(video_name, box, record, box_idx, debug_dir, video_root)
        for box_idx, box in enumerate(bbox)
    ]

    img_num = int(record["img_num"])
    st_frame = int(record["st_frame"])

    return {
        "width": width,
        "height": height,
        "img_num": img_num,
        "st_frame": st_frame,
        "st_time": float(record["st_time"]),
        "ed_time": float(record["ed_time"]),
        "caption": caption,
        "bbox": bbox,
    }


def convert_split(src_path: Path, dst_path: Path, query_map, debug_dir, video_root, split_name, report_dir):
    src_data = load_json(src_path)
    converted = {}
    bad_videos = []

    for video_name, record in src_data.items():
        try:
            converted[video_name] = normalize_record(
                video_name, record, query_map, debug_dir, video_root
            )
        except Exception as exc:
            bad_videos.append(
                {
                    "video_name": video_name,
                    "split": split_name,
                    "error": str(exc),
                }
            )

    dump_json(dst_path, converted)
    dump_json(report_dir / f"bad_{split_name}_videos.json", bad_videos)
    dump_text(
        report_dir / f"bad_{split_name}_videos.txt",
        [f"{item['video_name']}	{item['error']}" for item in bad_videos],
    )
    return len(converted), bad_videos


def main():
    parser = argparse.ArgumentParser(
        description="Convert HC-STVG2 source annotations from data/hc-stvg2/annos to the CG-STVG hcstvg_v2 format."
    )
    parser.add_argument(
        "--data-root",
        default="data/hc-stvg2",
        help="Root directory containing annos/train_v2.json, annos/val_v2.json, and annos/query_v2.json",
    )
    parser.add_argument(
        "--train-src",
        default=None,
        help="Optional override for the source training annotation file",
    )
    parser.add_argument(
        "--test-src",
        default=None,
        help="Optional override for the source test/validation annotation file",
    )
    parser.add_argument(
        "--query-src",
        default=None,
        help="Optional override for the query caption file",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional override for the destination annotation directory",
    )
    parser.add_argument(
        "--video-root",
        default=None,
        help="Optional override for the source video directory used for debug visualizations",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Optional override for negative-bbox debug images",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Optional override for bad-video report files",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_src = Path(args.train_src) if args.train_src else data_root / "annos" / "train_v2.json"
    test_src = Path(args.test_src) if args.test_src else data_root / "annos" / "val_v2.json"
    query_src = Path(args.query_src) if args.query_src else data_root / "annos" / "query_v2.json"
    out_dir = Path(args.out_dir) if args.out_dir else data_root / "annos" / "hcstvg_v2"
    video_root = Path(args.video_root) if args.video_root else data_root / "v2_video"
    debug_dir = Path(args.debug_dir) if args.debug_dir else data_root / "bbox_debug"
    report_dir = Path(args.report_dir) if args.report_dir else data_root / "conversion_reports"

    query_map = load_json(query_src) if query_src.exists() else {}

    train_count, train_bad = convert_split(
        train_src, out_dir / "train.json", query_map, debug_dir, video_root, "train", report_dir
    )
    test_count, test_bad = convert_split(
        test_src, out_dir / "test.json", query_map, debug_dir, video_root, "test", report_dir
    )
    all_bad = train_bad + test_bad
    dump_json(report_dir / "bad_all_videos.json", all_bad)
    dump_text(
        report_dir / "bad_all_videos.txt",
        [f"{item['split']}	{item['video_name']}	{item['error']}" for item in all_bad],
    )

    print(f"Converted {train_count} training records -> {out_dir / 'train.json'}")
    print(f"Converted {test_count} test records -> {out_dir / 'test.json'}")
    print(f"Skipped {len(train_bad)} bad training videos -> {report_dir / 'bad_train_videos.json'}")
    print(f"Skipped {len(test_bad)} bad test videos -> {report_dir / 'bad_test_videos.json'}")
    print(f"Wrote combined bad-video report -> {report_dir / 'bad_all_videos.json'}")
    print("Note: annos/val_v2.json was written to test.json because this repo evaluates the HC-STVG test split.")


if __name__ == "__main__":
    main()

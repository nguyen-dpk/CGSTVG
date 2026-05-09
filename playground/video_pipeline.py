"""
Build a model input batch from an arbitrary video path, mirroring HC-STVG test preprocessing.
"""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from torchvision.transforms import Resize

from copy import deepcopy

from datasets.build import build_transforms
from datasets.data_utils import make_hcstvg_input_clip
from utils.bounding_box import BoxList
from utils.misc import NestedTensor


def _heatmaps_for_action(actioness: np.ndarray, epsilon: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    action_idx = np.where(actioness)[0]
    start_idx, end_idx = int(action_idx[0]), int(action_idx[-1])
    start_heatmap = np.ones(actioness.shape) * epsilon
    pesudo_prob = (1 - (start_heatmap.shape[0] - 3) * epsilon - 0.5) / 2
    start_heatmap[start_idx] = 0.5
    if start_idx > 0:
        start_heatmap[start_idx - 1] = pesudo_prob
    if start_idx < actioness.shape[0] - 1:
        start_heatmap[start_idx + 1] = pesudo_prob
    end_heatmap = np.ones(actioness.shape) * epsilon
    end_heatmap[end_idx] = 0.5
    if end_idx > 0:
        end_heatmap[end_idx - 1] = pesudo_prob
    if end_idx < actioness.shape[0] - 1:
        end_heatmap[end_idx + 1] = pesudo_prob
    return start_heatmap, end_heatmap


def decode_video_rgb(video_path: str) -> Tuple[np.ndarray, int, int, float]:
    """Decode full video to RGB uint8 [T, H, W, 3] via OpenCV (no ffprobe/ffmpeg CLI on PATH)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps < 1e-3:
        fps = 30.0
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from: {video_path}")
    out = np.stack(frames, axis=0)
    h, w = int(out.shape[1]), int(out.shape[2])
    return out, h, w, fps


def sted_frame_ids_to_seconds(
    sted: list,
    orig_timeline_index: np.ndarray,
    fps: float,
) -> Tuple[float, float]:
    """
    Map predicted [start_id, end_id_exclusive] (postprocessor convention) to wall-clock seconds.
    orig_timeline_index[i] = source frame index in the full OpenCV decode (before raw decimation);
    keeps times correct when frames are subsampled.
    """
    o = orig_timeline_index.astype(np.float64)
    n = len(o)
    if n == 0:
        return 0.0, 0.0
    s = int(np.clip(sted[0], 0, n - 1))
    e_excl = int(sted[1])
    start_sec = float(o[s]) / fps
    if e_excl >= n:
        end_sec = float(o[-1] + 1.0) / fps
    else:
        end_sec = float(o[e_excl]) / fps
    return start_sec, end_sec


def synthetic_hcstvg_record(
    frame_count: int,
    height: int,
    width: int,
    query: str,
    item_id: int = 0,
) -> Dict:
    """
    Minimal HC-STVG-like dict for make_hcstvg_input_clip / dataset __getitem__ logic.
    Uses a centered dummy tube and whole-clip action (pseudo ground truth for model internals).
    """
    epsilon = 1e-10
    # Match datasets/hcstvg.py load_data: frame_ids are 0 .. frame_nums-2
    if frame_count < 2:
        frame_count = 2
    frame_ids = list(range(0, frame_count - 1))
    temp_gt_begin = frame_ids[0]
    temp_gt_end = frame_ids[-1]
    actioness = np.array([int(temp_gt_begin <= fid <= temp_gt_end) for fid in frame_ids], dtype=np.int64)
    start_heatmap, end_heatmap = _heatmaps_for_action(actioness, epsilon)

    n_tube = temp_gt_end - temp_gt_begin + 1
    cx, cy = width * 0.5, height * 0.5
    bw, bh = width * 0.4, height * 0.4
    x1, y1 = cx - bw / 2, cy - bh / 2
    x2, y2 = cx + bw / 2, cy + bh / 2
    bbox_array = np.stack([[x1, y1, x2, y2]] * n_tube, axis=0)

    return {
        "item_id": item_id,
        "vid": "playground.mp4",
        "frame_ids": frame_ids,
        "width": width,
        "height": height,
        "start_heatmap": start_heatmap,
        "end_heatmap": end_heatmap,
        "actioness": actioness,
        "bboxs": bbox_array,
        "gt_temp_bound": [temp_gt_begin, temp_gt_end],
        "description": query,
        "object": "person",
        "frame_count": frame_count,
    }


def tensorize_frames_like_hcstvg(cfg, frames_rgb: np.ndarray) -> torch.Tensor:
    """Resize short side to INPUT.RESOLUTION like HCSTVGDataset.load_frames."""
    t, h, w, _ = frames_rgb.shape
    rate = w / h
    max_rate = 1.4
    h_r = cfg.INPUT.RESOLUTION
    w_r = min(int(cfg.INPUT.RESOLUTION * rate), int(cfg.INPUT.RESOLUTION * max_rate))
    resize = Resize((h_r, w_r), antialias=True)
    out = []
    for i in range(t):
        img = torch.from_numpy(frames_rgb[i]).permute(2, 0, 1).float() / 255.0
        out.append(resize(img))
    return torch.stack(out, dim=0)


def build_playground_sample(cfg, video_path: str, query: str):
    """
    Returns (NestedTensor batch dict components, sentence, targets list) compatible with collate_fn output.
    """
    frames_rgb, h, w, fps = decode_video_rgb(video_path)
    # Index into full decode timeline for each kept frame (for seconds output).
    orig_timeline_index = np.arange(frames_rgb.shape[0], dtype=np.float64)
    # Long videos: thin raw frames so sampled clip length stays near INPUT.MAX_VIDEO_LEN (model time embed).
    max_raw = max(400, cfg.INPUT.MAX_VIDEO_LEN * 4)
    if frames_rgb.shape[0] > max_raw:
        idx = np.linspace(0, frames_rgb.shape[0] - 1, num=max_raw, dtype=np.float64)
        idx = np.unique(np.round(idx).astype(np.int64))
        frames_rgb = frames_rgb[idx]
        orig_timeline_index = idx.astype(np.float64)
    frame_count = int(frames_rgb.shape[0])
    video_data = synthetic_hcstvg_record(frame_count, h, w, query.lower())

    video_data = make_hcstvg_input_clip(cfg, "test", deepcopy(video_data))
    frame_ids_clip = video_data["frame_ids"]
    frames_clip = frames_rgb[frame_ids_clip]
    max_t = cfg.INPUT.MAX_VIDEO_LEN
    if len(frame_ids_clip) > max_t:
        pick = np.linspace(0, len(frame_ids_clip) - 1, max_t, dtype=np.float64)
        pick = np.unique(np.round(pick).astype(np.int64))
        frame_ids = [frame_ids_clip[i] for i in pick]
        frames_rgb = frames_clip[pick]
        n = len(frame_ids)
        tb0, tb1 = frame_ids[0], frame_ids[-1]
        video_data["frame_ids"] = frame_ids
        video_data["actioness"] = np.ones(n, dtype=np.int64)
        video_data["start_heatmap"], video_data["end_heatmap"] = _heatmaps_for_action(video_data["actioness"])
        video_data["gt_temp_bound"] = [tb0, tb1]
        n_tube = tb1 - tb0 + 1
        x1, y1 = w * 0.2, h * 0.2
        x2, y2 = w * 0.8, h * 0.8
        video_data["bboxs"] = np.stack([[x1, y1, x2, y2]] * n_tube)
    else:
        frame_ids = frame_ids_clip
        frames_rgb = frames_clip

    frames = tensorize_frames_like_hcstvg(cfg, frames_rgb)

    temp_gt = video_data["gt_temp_bound"]
    action_idx = np.where(video_data["actioness"])[0]
    start_idx, end_idx = int(action_idx[0]), int(action_idx[-1])
    bbox_idx = [video_data["frame_ids"][idx] - temp_gt[0] for idx in range(start_idx, end_idx + 1)]
    bboxs = torch.from_numpy(video_data["bboxs"][bbox_idx]).reshape(-1, 4)
    bboxs = BoxList(bboxs, (w, h), "xyxy")

    input_dict = {
        "frames": frames,
        "boxs": bboxs,
        "text": video_data["description"],
        "actioness": video_data["actioness"],
    }
    transforms = build_transforms(cfg, is_train=False)
    input_dict = transforms(input_dict)

    targets = {
        "item_id": video_data["item_id"],
        "frame_ids": video_data["frame_ids"],
        "actioness": torch.from_numpy(video_data["actioness"]),
        "start_heatmap": torch.from_numpy(video_data["start_heatmap"]),
        "end_heatmap": torch.from_numpy(video_data["end_heatmap"]),
        "boxs": input_dict["boxs"],
        "img_size": input_dict["frames"].shape[2:],
        "ori_size": (h, w),
    }

    videos_list = [input_dict["frames"]]
    batch = {
        "videos": NestedTensor.from_tensor_list(videos_list),
        "texts": [input_dict["text"]],
        "targets": [targets],
        "video_fps": fps,
        "orig_timeline_index": orig_timeline_index,
    }
    return batch

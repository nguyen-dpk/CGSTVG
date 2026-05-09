#!/usr/bin/env python3
"""
Run CGSTVG / HC-STVG checkpoint on one video and one text query (playground / demo).

Equivalent inference pattern to distributed test (engine.evaluate.do_eval): two subsampled
streams merged for boxes (linear_interp) and temporal bounds.

Config overrides (KEY VALUE ...) can appear before or after --video / --query.

Example:
  python playground/run_video_query.py \\
    --config-file experiments/hcstvg.yaml \\
    INPUT.RESOLUTION 420 MODEL.WEIGHT model_zoo/hcstvg.pth \\
    --video /path/to/video.mp4 --query "the person who is cooking"
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from config import cfg
from engine.evaluate import linear_interp
from models import build_model, build_postprocessors
from playground.confidence import merge_dual_stream_confidence, single_forward_with_confidence
from utils.checkpoint import VSTGCheckpointer
from utils.logger import setup_logger
from utils.misc import to_device

from playground.video_pipeline import build_playground_sample, sted_frame_ids_to_seconds


def parse_args():
    p = argparse.ArgumentParser(description="Single-video STVG inference (playground)")
    p.add_argument("--config-file", default="experiments/hcstvg.yaml", metavar="FILE")
    p.add_argument("--video", required=True, type=str, help="Path to a video file (e.g. mp4)")
    p.add_argument("--query", required=True, type=str, help="Natural language query (lower-cased like training)")
    p.add_argument("--device", default=None, help="cuda or cpu (default: cfg MODEL.DEVICE)")
    p.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write predictions as JSON (bbox per frame id, sted)",
    )
    # parse_known_args: config overrides (KEY VALUE pairs) must not use REMAINDER,
    # or they swallow --video/--query when listed first.
    args, unknown = p.parse_known_args()
    return args, unknown


@torch.no_grad()
def run_dual_stream_like_eval(cfg, model, postprocessor, device, batch_dict):
    videos = batch_dict["videos"].to(device)
    texts = batch_dict["texts"]
    targets = to_device(batch_dict["targets"], device)
    for i in range(len(targets)):
        if "qtype" not in targets[i]:
            targets[i]["qtype"] = "none"

    videos1 = videos.subsample(2, start_idx=0)
    targets1 = [
        {
            "item_id": target["item_id"],
            "ori_size": target["ori_size"],
            "qtype": target["qtype"],
            "frame_ids": target["frame_ids"][0::2],
            "boxs": target["boxs"].bbox.clone(),
            "actioness": target["actioness"][0::2],
            "eval": True,
        }
        for target in targets
    ]

    videos2 = videos.subsample(2, start_idx=1)
    targets2 = [
        {
            "item_id": target["item_id"],
            "ori_size": target["ori_size"],
            "qtype": target["qtype"],
            "frame_ids": target["frame_ids"][1::2],
            "boxs": target["boxs"].bbox.clone(),
            "actioness": target["actioness"][1::2],
            "eval": True,
        }
        for target in targets
    ]

    if torch.where(targets[0]["actioness"])[0][0] % 2 == 0:
        targets1[0]["boxs"] = targets1[0]["boxs"][0::2]
        targets2[0]["boxs"] = targets2[0]["boxs"][1::2]
    else:
        targets1[0]["boxs"] = targets1[0]["boxs"][1::2]
        targets2[0]["boxs"] = targets2[0]["boxs"][0::2]

    bbox_pred1, temp_pred1, conf1 = single_forward_with_confidence(
        cfg, model, videos1, texts, targets1, device, postprocessor
    )
    bbox_pred2, temp_pred2, conf2 = single_forward_with_confidence(
        cfg, model, videos2, texts, targets2, device, postprocessor
    )

    bbox_out, temp_out = {}, {}
    for vid in bbox_pred1:
        bbox_pred1[vid].update(bbox_pred2[vid])
        bbox_out[vid] = linear_interp(bbox_pred1[vid])
        temp_out[vid] = {
            "sted": [
                min(temp_pred1[vid]["sted"][0], temp_pred2[vid]["sted"][0]),
                max(temp_pred1[vid]["sted"][1], temp_pred2[vid]["sted"][1]),
            ]
        }
        if "qtype" in temp_pred1[vid]:
            temp_out[vid]["qtype"] = temp_pred1[vid]["qtype"]

    conf_merged = merge_dual_stream_confidence(conf1, conf2)
    return bbox_out, temp_out, conf_merged


def main():
    args, unknown = parse_args()
    cfg.merge_from_file(args.config_file)
    if unknown:
        cfg.merge_from_list(unknown)
    cfg.freeze()

    device = torch.device(args.device or cfg.MODEL.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device = torch.device("cpu")

    logger = setup_logger("playground", "", 0)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model, _, _ = build_model(cfg)
    model.to(device)
    model.eval()

    checkpointer = VSTGCheckpointer(cfg, model, logger=logger, is_train=False)
    _ = checkpointer.load(cfg.MODEL.WEIGHT, with_optim=False)

    postprocessor = build_postprocessors()
    batch_dict = build_playground_sample(cfg, args.video, args.query)

    bbox_pred, temp_pred, confidence = run_dual_stream_like_eval(cfg, model, postprocessor, device, batch_dict)

    item_id = batch_dict["targets"][0]["item_id"]
    print("item_id:", item_id)
    sted = temp_pred[item_id]["sted"]
    print("temporal (frame ids, end exclusive):", sted)
    fps = float(batch_dict["video_fps"])
    t0, t1 = sted_frame_ids_to_seconds(sted, batch_dict["orig_timeline_index"], fps)
    print(f"temporal (seconds, end exclusive): [{t0:.3f}, {t1:.3f}]  (video_fps={fps:g})")
    print("confidence:")
    tp = confidence.get("temporal_prob_mean")
    sp = confidence.get("spatial_box_conf_mean_in_sted_mean")
    if tp is not None:
        print(f"  temporal P(winning segment, MAP): {tp:.6f}  (streams: {confidence.get('temporal_prob_stream1')}, {confidence.get('temporal_prob_stream2')})")
    if sp is not None:
        print(f"  spatial mean box conf in predicted segment: {sp:.6f}  (streams: {confidence.get('spatial_box_conf_mean_in_sted_stream1')}, {confidence.get('spatial_box_conf_mean_in_sted_stream2')})")
    ap = confidence.get("actioness_prob_mean_in_sted_mean")
    if ap is not None:
        print(f"  actioness sigmoid mean in predicted segment: {ap:.6f}")
    n_boxes = len(bbox_pred[item_id])
    print("num frames with box predictions:", n_boxes)

    if args.output_json:
        out = {
            "video": os.path.abspath(args.video),
            "query": args.query,
            "item_id": item_id,
            "sted_frame_ids": sted,
            "sted_seconds_end_exclusive": [t0, t1],
            "video_fps": fps,
            "confidence": confidence,
            "boxes_xyxy_by_frame": {str(k): v for k, v in sorted(bbox_pred[item_id].items())},
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("wrote", args.output_json)


if __name__ == "__main__":
    main()

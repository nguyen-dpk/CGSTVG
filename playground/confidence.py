"""Confidence metrics from raw model outputs (temporal MAP prob, spatial box conf, optional actioness)."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

@torch.no_grad()
def _temporal_winning_stats(outputs: Dict, durations: List[int], i_b: int = 0) -> Tuple[float, float]:
    """Same log-prob map as models.post_processor.PostProcess; return (prob, log_prob) at argmax."""
    out_sted = outputs["pred_sted"]
    b, t, _ = out_sted.shape
    device = out_sted.device
    inf = -1e32
    temp_prob_map = torch.zeros(b, t, t, device=device, dtype=out_sted.dtype)
    for j in range(len(durations)):
        duration = durations[j]
        sted_prob = (torch.ones(t, t, device=device, dtype=out_sted.dtype) * inf).tril(0)
        sted_prob[duration:, :] = inf
        sted_prob[:, duration:] = inf
        temp_prob_map[j, :, :] = sted_prob
    temp_prob_map = temp_prob_map + F.log_softmax(out_sted[:, :, 0], dim=1).unsqueeze(2) + F.log_softmax(
        out_sted[:, :, 1], dim=1
    ).unsqueeze(1)
    prob_seq = temp_prob_map[i_b].flatten(0)
    max_logp = prob_seq.max()
    return max_logp.exp().item(), max_logp.item()


@torch.no_grad()
def _mean_spatial_conf_in_sted(
    outputs: Dict,
    frames_id: List,
    sted: List[int],
    duration: int,
    b: int,
    t: int,
    i_b: int = 0,
) -> float:
    """Mean boxes_conf on rows whose frame id lies in [sted[0], sted[1]) (end exclusive)."""
    bc = outputs["boxes_conf"]
    if bc.dim() > 1:
        bc = bc.squeeze(-1)
    bc = bc.view(b, t)
    vals = []
    for idx in range(duration):
        fid = frames_id[idx]
        if sted[0] <= fid < sted[1]:
            vals.append(bc[i_b, idx].item())
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


@torch.no_grad()
def _mean_actioness_prob_in_sted(
    outputs: Dict,
    frames_id: List,
    sted: List[int],
    duration: int,
    b: int,
    t: int,
    i_b: int = 0,
) -> Optional[float]:
    if "pred_actioness" not in outputs:
        return None
    pa = outputs["pred_actioness"].squeeze(-1)
    pa = torch.sigmoid(pa).view(b, t)
    vals = []
    for idx in range(duration):
        fid = frames_id[idx]
        if sted[0] <= fid < sted[1]:
            vals.append(pa[i_b, idx].item())
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


@torch.no_grad()
def single_forward_with_confidence(cfg, model, videos, texts, targets, device, postprocessor):
    """
    Like engine.evaluate.single_forward but adds a confidence dict for batch index 0.
    """
    durations = videos.durations
    targets[0]["durations"] = durations
    outputs = model(videos, texts, targets)

    b = len(durations)
    t = max(durations)
    batch_img_size = [list(target["ori_size"]) for target in targets]
    orig_target_sizes = [img_size for img_size in batch_img_size for _ in range(t)]
    orig_target_sizes = torch.tensor(orig_target_sizes, device=device)
    frames_ids = [target["frame_ids"] for target in targets]
    pred_boxs, pred_steds = postprocessor(outputs, orig_target_sizes, frames_ids, durations)
    pred_boxs = pred_boxs.view(b, t, 4)

    vids = [target["item_id"] for target in targets]
    bbox_pred, temp_pred = {}, {}

    for i_b in range(b):
        frames_id = frames_ids[i_b]
        bbox_pred[vids[i_b]] = {}
        assert durations[i_b] == len(frames_id)
        for idx in range(durations[i_b]):
            bbox_pred[vids[i_b]][frames_id[idx]] = [pred_boxs[i_b][idx].detach().cpu().tolist()]

    if cfg.DATASET.NAME == "VidSTG":
        qtypes = [target["qtype"] for target in targets]
        assert len(pred_steds) == len(qtypes)
        for i_b in range(b):
            temp_pred[vids[i_b]] = {"sted": pred_steds[i_b], "qtype": qtypes[i_b]}
    else:
        for i_b in range(b):
            temp_pred[vids[i_b]] = {"sted": pred_steds[i_b]}

    i_b = 0
    sted = pred_steds[i_b]
    tp, tlog = _temporal_winning_stats(outputs, durations, i_b)
    sp = _mean_spatial_conf_in_sted(outputs, frames_ids[i_b], sted, durations[i_b], b, t, i_b)
    ap = _mean_actioness_prob_in_sted(outputs, frames_ids[i_b], sted, durations[i_b], b, t, i_b)

    confidence: Dict[str, Any] = {
        "temporal_prob": tp,
        "temporal_log_prob": tlog,
        "spatial_box_conf_mean_in_sted": sp,
    }
    if ap is not None:
        confidence["actioness_prob_mean_in_sted"] = ap

    return bbox_pred, temp_pred, confidence


def merge_dual_stream_confidence(c1: Dict[str, Any], c2: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k in ("temporal_prob", "temporal_log_prob", "spatial_box_conf_mean_in_sted"):
        v1, v2 = c1.get(k), c2.get(k)
        if v1 is None or v2 is None:
            continue
        if isinstance(v1, float) and (math.isnan(v1) or math.isnan(v2)):
            continue
        out[f"{k}_stream1"] = v1
        out[f"{k}_stream2"] = v2
        out[f"{k}_mean"] = (v1 + v2) / 2.0
    a1, a2 = c1.get("actioness_prob_mean_in_sted"), c2.get("actioness_prob_mean_in_sted")
    if a1 is not None and a2 is not None and not math.isnan(a1) and not math.isnan(a2):
        out["actioness_prob_mean_in_sted_stream1"] = a1
        out["actioness_prob_mean_in_sted_stream2"] = a2
        out["actioness_prob_mean_in_sted_mean"] = (a1 + a2) / 2.0
    return out

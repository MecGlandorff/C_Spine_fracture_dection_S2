# Program: Cervical fracture detection of C1-C7 (stage 2 inference)
# Author: Mec Glandorff
# Version: 3.2-proto
# Description:
#   Load model checkpoint and score a patient over his/her vertebrae(C1..C7).
#
# Output:
#   - per vertebra: probability + prediction
#   - patient-level: max(prob) + prediction

"""
Note (prototype):
This module currently still expects the UNPACKED per-timestep .npy files:
  {uid}_{cid}_{t}.npy with shape (H, W, 6)

Training uses PACKED per-vertebra .npy files:
  {uid}_{cid}.npy with shape (T, H, W, 6)

Action: update load_vertebra_sequence() to read the packed format for consistency.
"""

from __future__ import annotations

import os
from typing import Dict, Any

import numpy as np
import torch

from .common import get_device, load_checkpoint
from .model import build_model
from .data import normalize_intensity_and_mask


def _assert_npy_ok(arr: np.ndarray, fp: str, expect_ch: int = 6) -> None:
    """Validate a single unpacked timestep array"""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Loaded object is not np.ndarray: {fp}")
    if arr.ndim != 3:
        raise ValueError(f"Expected (H,W,C), got {arr.shape} in {fp}")
    if arr.shape[-1] != expect_ch:
        raise ValueError(f"Expected {expect_ch} channels, got {arr.shape[-1]} in {fp}")


def load_vertebra_sequence(uid: str, cid: int, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Loads UNPACKED per-timestep npy files for one vertebra:
      {uid}_{cid}_{t}.npy  -> (H, W, 6)
    Returns:
      seq: (T, 6, H, W) float32
    """
    data_dir = str(cfg["data_dir"])
    T = int(cfg.get("n_slice_per_c", 15))

    frames = []
    for t in range(T):
        fp = os.path.join(data_dir, f"{uid}_{cid}_{t}.npy")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing npy: {fp}")

        arr = np.load(fp, allow_pickle=False)
        _assert_npy_ok(arr, fp, expect_ch=6)

        arr = normalize_intensity_and_mask(arr, cfg)  # (H,W,6) float
        arr = arr.transpose(2, 0, 1)                  # (6,H,W)
        frames.append(arr)

    return np.stack(frames, axis=0).astype(np.float32, copy=False)  # (T,6,H,W)


@torch.no_grad()
def score_patient(uid: str, checkpoint_path: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    device = get_device(cfg)
    ckpt = load_checkpoint(checkpoint_path, device)

    # Merge cfg: checkpoint cfg as base, runtime cfg overrides (for calibration experiments).
    merged = dict(ckpt.get("cfg", {}) or {})
    merged.update(cfg)

    model = build_model(merged).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    # Threshold policy: runtime cfg override > checkpoint.
    th_override = merged.get("threshold", None)
    th = float(ckpt.get("threshold", 0.5)) if th_override is None else float(th_override)

    cids = list(range(1, 8))
    xs = [load_vertebra_sequence(uid, c, merged) for c in cids]

    x = np.stack(xs, axis=0).astype(np.float32, copy=False)  # (7, T, 6, H, W)
    x_t = torch.from_numpy(x).to(device)

    logits = model(x_t)  # (7,)
    probs = torch.sigmoid(logits).detach().cpu().numpy()

    out: Dict[str, Any] = {
        "StudyInstanceUID": uid,
        "checkpoint": checkpoint_path,
        "threshold": th,
        "per_vertebra": {},
        "patient_score": None,
        "patient_pred": None,
    }

    for i, c in enumerate(cids):
        p = float(probs[i])
        out["per_vertebra"][f"C{c}"] = {"prob": p, "pred": int(p >= th)}

    # Patient rule: OR-aggregation via max(prob). So it is sensitive by design.
    out["patient_score"] = float(np.max(probs))
    out["patient_pred"] = int(out["patient_score"] >= th)

    return out

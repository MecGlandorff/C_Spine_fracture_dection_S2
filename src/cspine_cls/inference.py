# Program: Cervical fracture detection of C1-C7 (stage 2 inference)
# Author: Mec Glandorff
# Version: 3.2-proto
# Description:
#   Load model checkpoint and score a patient over his/her vertebrae(C1..C7).
#
# Output:
#   - per vertebra: probability + prediction
#   - patient-level: max(prob) + prediction

from __future__ import annotations

import os
from typing import Dict, Any

import numpy as np
import torch

from .common import get_device, load_checkpoint
from .data import build_transforms, normalize_intensity_and_mask


def _assert_packed_ok(arr: np.ndarray, fp: str, expect_t: int, expect_ch: int = 6) -> None:
    """Validate one packed vertebra sequence."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Loaded object is not np.ndarray: {fp}")
    if arr.ndim != 4:
        raise ValueError(f"Expected packed shape (T,H,W,C), got {arr.shape} in {fp}")
    if arr.shape[0] != expect_t:
        raise ValueError(f"Expected T={expect_t}, got T={arr.shape[0]} in {fp}")
    if arr.shape[-1] != expect_ch:
        raise ValueError(f"Expected {expect_ch} channels, got {arr.shape[-1]} in {fp}")


def load_vertebra_sequence(uid: str, cid: int, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Loads one packed per-vertebra npy file:
      {uid}_{cid}.npy -> (T, H, W, 6)
    Returns:
      seq: (T, 6, H, W) float32
    """
    data_dir = str(cfg["data_dir"])
    T = int(cfg.get("n_slice_per_c", 15))
    fp = os.path.join(data_dir, f"{uid}_{cid}.npy")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Missing packed npy: {fp}")

    seq = np.load(fp, allow_pickle=False)
    _assert_packed_ok(seq, fp, expect_t=T, expect_ch=6)

    geo_tf, _ = build_transforms(cfg, is_train=False)
    frames = []
    for t in range(T):
        arr = seq[t]
        intensity = arr[..., :5]
        mask = arr[..., 5]

        if geo_tf is not None:
            res = geo_tf(image=intensity, mask=mask)
            intensity = res["image"]
            mask = res["mask"]

        if mask.ndim == 2:
            mask = mask[..., None]

        arr = np.concatenate([intensity, mask], axis=-1)
        arr = normalize_intensity_and_mask(arr, cfg)  # (H,W,6) float
        arr = arr.transpose(2, 0, 1)                  # (6,H,W)
        frames.append(arr)

    return np.stack(frames, axis=0).astype(np.float32, copy=False)  # (T,6,H,W)


@torch.no_grad()
def score_patient(uid: str, checkpoint_path: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    from .model import build_model

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

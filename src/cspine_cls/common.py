# Program: Cervical fracture detection of C1-C7 (stage 2)
# Author: Mec Glandorff
# Version: 3.2-proto
# Description:
#   Common utils:
#     - seeds
#     - device selection
#     - checkpoint IO
#     - small logging helpers
#
# Notes for myself:
#   - scripts/ reads yaml. src/ expects a dictionary!
#   - keep failure modes obvious.

from __future__ import annotations

import os
import json
import time
import random
from typing import Dict, Any, Optional

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(cfg: Dict[str, Any]) -> torch.device:
    # Activate GPU
    if bool(cfg.get("force_cpu", False)):
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str):
    if path == "":
        return
    os.makedirs(path, exist_ok=True)


def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def save_json(obj: Dict[str, Any], path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def log_line(path: str, line: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "a") as f:
        f.write(line.rstrip() + "\n")


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)

    # Check if dict stays to preferred format. Preferred: {"state_dict":... , "threshold": ...., "cfg":..., "meta": ...}
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt

    # Backward compatibility, raw state_dict only
    if isinstance(ckpt, dict):
        return {"state_dict": ckpt, "threshold": 0.5, "cfg": {}, "meta": {}}

    raise ValueError("Checkpoint format not recognized.")


def save_checkpoint(
    path: str,
    model_state: Dict[str, Any],
    cfg: Dict[str, Any],
    threshold: float,
    meta: Optional[Dict[str, Any]] = None,
):
    ensure_dir(os.path.dirname(path))
    payload = {
        "state_dict": model_state,
        "threshold": float(threshold),
        "cfg": cfg,
        "meta": meta or {},
    }
    torch.save(payload, path)

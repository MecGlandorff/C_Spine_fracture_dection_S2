# Program: Cervical fracture detection of C1-C7 (stage 2)
# Author: Mec Glandorff
# Version: 3.2-proto
# Description:
#   Metrics + threshold sweep.
#
# Notes:
#   - threshold=0.5 is a default, not an objective. We can tweak this around
#   - We pick a threshold on validation based on cfg objective (recall or f1). Most likely recall is most important since we want to reduce false negatives.

from __future__ import annotations

from typing import Dict, Any
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    accuracy_score,
    precision_score,
)


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, th: float) -> Dict[str, float]:
    pred = (y_prob >= th).astype(int)
    return {
        "auc": safe_auc(y_true, y_prob),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "acc": float(accuracy_score(y_true, pred)),
    }


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, float]:
    ths = cfg.get("thresholds", None)
    if ths is None:
        ths = np.linspace(0.05, 0.95, 19)

    objective = str(cfg.get("threshold_objective", "recall")).lower()  # recall or f1?
    precision_floor = float(cfg.get("precision_floor", 0.0))

    best = {"th": 0.5, "recall": -1.0, "f1": -1.0, "precision": -1.0, "acc": -1.0}

    for th in ths:
        pred = (y_prob >= th).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        prec = precision_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        acc = accuracy_score(y_true, pred)

        if prec < precision_floor:
            continue

        score = f1 if objective == "f1" else rec
        best_score = best["f1"] if objective == "f1" else best["recall"]

        if score > best_score:
            best = {
                "th": float(th),
                "recall": float(rec),
                "f1": float(f1),
                "precision": float(prec),
                "acc": float(acc),
            }

    return best

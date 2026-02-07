# src/cspine_cls/train.py
# Program: Cervical fracture detection of C1-C7 (stage 2 training)
# Author: Mec Glandorff
# Version: 3.2-proto
# Description:
#   Train/valid loops + early stopping.
#
# Notes:
#   - This file is a MODULE. It is not meant to be run directly.
#   - Import via scripts/train.py so the package context is correct.
#   - Focal alpha must be in [0,1].
#   - IMPORTANT: validation split is patient-level (StudyInstanceUID) to avoid leakage.

from __future__ import annotations

import os
import json
import time
import gc
from typing import Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from .common import set_seed, get_device, ensure_dir, save_checkpoint, log_line
from .data import make_loaders
from .model import build_model
from .metrics import threshold_sweep, compute_binary_metrics, safe_auc


# If someone runs this file directly, fail with a helpful message.
if __name__ == "__main__":
    raise SystemExit(
        "Do not run src/cspine_cls/train.py directly.\n"
        "Run: python scripts/train.py --config configs/train.yaml\n"
    )


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Keep targets strictly binary to avoid alpha_t ambiguities
        targets = (targets > 0.5).to(logits.dtype)

        # BCE per sample
        bce = self.bce(logits, targets)

        # pt = p if y=1 else 1-p
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        # alpha_t = alpha if y=1 else 1-alpha
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # focal factor
        focal = alpha_t * (1 - pt) ** self.gamma
        return focal * bce


def _binarize_labels(expanded: pd.DataFrame) -> pd.DataFrame:
    expanded["label"] = pd.to_numeric(expanded["label"], errors="coerce").fillna(0.0)
    expanded["label"] = (expanded["label"] > 0.5).astype(np.float32)

    u = np.unique(expanded["label"].values)
    print("Unique labels:", u)
    if not set(u.tolist()).issubset({0.0, 1.0}):
        raise ValueError(f"Labels not binary after binarize: {u}")

    return expanded


def _expand_patient_df(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Expand patient-level rows to (StudyInstanceUID, c, label) rows
    Assumes df_in has columns: StudyInstanceUID, C1..C7
    """
    study_list, c_list, label_list = [], [], []
    for _, row in df_in.iterrows():
        uid = row.StudyInstanceUID
        for c in range(1, 8):
            study_list.append(uid)
            c_list.append(c)
            label_list.append(row[f"C{c}"])

    expanded = pd.DataFrame({"StudyInstanceUID": study_list, "c": c_list, "label": label_list})
    expanded.drop_duplicates(inplace=True)
    expanded.reset_index(drop=True, inplace=True)
    expanded = _binarize_labels(expanded)
    return expanded


def _focal_alpha_from_class_weights(labels01: np.ndarray) -> float:
    # sklearn gives "balanced" weights per class. Those are not focal alpha!
    # We map the weights to alpha in (0,1) so alpha_t for negatives (1-alpha) stays positive.
    labels01 = labels01.astype(int)
    classes = np.unique(labels01)

    cw = compute_class_weight(class_weight="balanced", classes=classes, y=labels01)
    class_w = {int(classes[i]): float(cw[i]) for i in range(len(classes))}

    w0 = float(class_w.get(0, 1.0))
    w1 = float(class_w.get(1, 1.0))

    alpha_pos = w1 / (w0 + w1 + 1e-12)
    alpha_pos = float(np.clip(alpha_pos, 1e-4, 1.0 - 1e-4))

    print(f"Class weights: w0={w0:.3f} w1={w1:.3f} -> focal alpha_pos={alpha_pos:.4f}")
    return alpha_pos


def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device, cfg: Dict[str, Any]) -> Dict[str, float]:
    model.train()

    acc_steps = int(cfg.get("accumulation_steps", 1))
    grad_clip = float(cfg.get("grad_clip", 1.0))

    # AMP only makes sense on CUDA, so only apply when device type is the GPU.
    use_amp = bool(cfg.get("use_amp", True)) and (device.type == "cuda")

    losses = []
    all_probs = []
    all_targets = []

    optimizer.zero_grad(set_to_none=True)
    bar = tqdm(loader, desc="Training", leave=True)

    for step, (x, y, cid, w, uid) in enumerate(bar):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)

        # Loss must not be negative for BCE/focal if y in [0,1] and alpha in [0,1].
        if (y.min() < 0) or (y.max() > 1):
            raise RuntimeError(f"Targets out of range: min={float(y.min())} max={float(y.max())}")

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)                # (bs,)
            per_sample = loss_fn(logits, y)  # (bs,)
            per_sample = per_sample * w      # per vertebra weights
            loss = per_sample.mean()
            loss = loss / acc_steps

        scaler.scale(loss).backward()

        if (step + 1) % acc_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item() * acc_steps)

        probs = torch.sigmoid(logits).detach().float().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_targets.extend(y.detach().float().cpu().numpy().tolist())

        bar.set_postfix(loss=float(np.mean(losses)))

    # leftover grads
    if (step + 1) % acc_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    y_true = np.array(all_targets, dtype=np.float32)
    y_prob = np.array(all_probs, dtype=np.float32)

    m05 = compute_binary_metrics(y_true, y_prob, th=0.5)

    return {
        "loss": float(np.mean(losses)),
        "auc": safe_auc(y_true, y_prob),
        "recall@0.5": float(m05["recall"]),
        "f1@0.5": float(m05["f1"]),
        "precision@0.5": float(m05["precision"]),
        "acc@0.5": float(m05["acc"]),
    }


@torch.no_grad()
def validate(model, loader, loss_fn, device, cfg: Dict[str, Any]) -> Dict[str, Any]:
    model.eval()

    losses = []
    all_probs = []
    all_targets = []
    all_cids = []

    bar = tqdm(loader, desc="Validating", leave=True)
    for x, y, cid, w, uid in bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)

        if (y.min() < 0) or (y.max() > 1):
            raise RuntimeError(f"Targets out of range: min={float(y.min())} max={float(y.max())}")

        logits = model(x)
        per_sample = loss_fn(logits, y)
        per_sample = per_sample * w
        loss = per_sample.mean()

        losses.append(loss.item())

        probs = torch.sigmoid(logits).detach().float().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_targets.extend(y.detach().float().cpu().numpy().tolist())
        all_cids.extend(cid.detach().cpu().numpy().tolist())

        bar.set_postfix(loss=float(np.mean(losses)))

    y_true = np.array(all_targets, dtype=np.float32)
    y_prob = np.array(all_probs, dtype=np.float32)

    sweep = threshold_sweep(y_true, y_prob, cfg)
    th = float(sweep["th"])
    metrics = compute_binary_metrics(y_true, y_prob, th=th)

    # per vertebra metrics at the selected threshold
    per_c = {}
    all_cids = np.array(all_cids, dtype=int)
    for c in np.unique(all_cids):
        idx = np.where(all_cids == c)[0]
        yt = y_true[idx]
        yp = y_prob[idx]
        per_c[int(c)] = compute_binary_metrics(yt, yp, th=th)

    return {
        "loss": float(np.mean(losses)),
        "th": th,
        **metrics,
        "per_c": per_c,
    }


def run_train(cfg: Dict[str, Any]) -> str:
    device = get_device(cfg)
    print(f"Device: {device}")

    set_seed(int(cfg.get("seed", 42)))

    df = pd.read_csv(cfg["csv_path"])

    # Patient-level stratification target: positive if any vertebra fractured
    c_cols = [f"C{c}" for c in range(1, 8)]
    y_patient = (df[c_cols].fillna(0).astype(float).max(axis=1) > 0.5).astype(int)

    # Patient-level train/validation split to avoid leakage
    val_split = float(cfg.get("val_split", 0.2))
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_split,
        random_state=int(cfg.get("seed", 42)),
    )
    tr_pat_idx, va_pat_idx = next(sss.split(df, y_patient))

    df_tr = df.iloc[tr_pat_idx].reset_index(drop=True)
    df_va = df.iloc[va_pat_idx].reset_index(drop=True)

    # Expand after splitting (no UID leakage)
    train_df = _expand_patient_df(df_tr)
    val_df = _expand_patient_df(df_va)

    # focal alpha fix: alpha must be in [0,1]
    alpha_pos = _focal_alpha_from_class_weights(train_df["label"].values)

    gamma = float(cfg.get("focal_gamma", 2.0))
    loss_fn = FocalLoss(alpha=alpha_pos, gamma=gamma).to(device)

    loader_tr, loader_va = make_loaders(train_df, val_df, cfg)

    model = build_model(cfg).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=float(cfg.get("init_lr", 1e-4)))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(cfg.get("lr_factor", 0.5)),
        patience=int(cfg.get("lr_patience", 3)),
        verbose=True,
        min_lr=float(cfg.get("eta_min", 1e-6)),
    )

    # AMP only on CUDA
    use_amp = bool(cfg.get("use_amp", True)) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_name = str(cfg.get("run_name", "run_" + time.strftime("%Y%m%d_%H%M%S")))
    run_dir = os.path.join(str(cfg.get("out_dir", "./outputs")), run_name)
    ensure_dir(run_dir)

    # save cfg snapshot for reproducibility
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    log_path = os.path.join(run_dir, "train.log")
    best_path = os.path.join(run_dir, "best_model.pth")

    best_recall = -1.0
    best_th = 0.5
    patience = int(cfg.get("early_stopping_patience", 5))
    no_improve = 0

    n_epochs = int(cfg.get("n_epochs", 25))
    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch [{epoch}/{n_epochs}]")

        tr = train_one_epoch(model, loader_tr, optimizer, scaler, loss_fn, device, cfg)
        va = validate(model, loader_va, loss_fn, device, cfg)

        line = (
            f"{time.ctime()} | epoch={epoch} "
            f"train_loss={tr['loss']:.4f} train_auc={tr['auc']:.4f} "
            f"val_loss={va['loss']:.4f} val_auc={va['auc']:.4f} "
            f"val_recall={va['recall']:.4f} val_f1={va['f1']:.4f} "
            f"val_prec={va['precision']:.4f} th={va['th']:.2f}"
        )
        print(line)
        log_line(log_path, line)

        if bool(cfg.get("print_per_c", True)):
            for c_id in sorted(va["per_c"].keys()):
                m = va["per_c"][c_id]
                print(f"  C{c_id}: rec={m['recall']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f}")

        scheduler.step(va["recall"])

        # early stopping: maximize recall
        if va["recall"] > best_recall:
            best_recall = float(va["recall"])
            best_th = float(va["th"])
            no_improve = 0

            save_checkpoint(
                best_path,
                model.state_dict(),
                cfg=cfg,
                threshold=best_th,
                meta={"epoch": epoch, "best_recall": best_recall},
            )
            print(f"  [*] best updated: recall={best_recall:.4f} th={best_th:.2f}")
        else:
            no_improve += 1
            print(f"  no recall improvement: {no_improve}/{patience}")

        if no_improve >= patience:
            print("Early stopping triggered.")
            break

    print("\nTraining complete.")
    print(f"Best checkpoint: {best_path}")
    print(f"Best threshold: {best_th:.2f}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return run_dir

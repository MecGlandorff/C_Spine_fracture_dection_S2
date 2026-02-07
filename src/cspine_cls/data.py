# src/cspine_cls/data.py
# Program: Cervical fracture detection of C1-C7 (stage 2)
# Author: Mec Glandorff
# Version: 3.2-proto
# Description:
#   Dataset + transforms for vertebra-level fracture classification.
#
#   IMPORTANT:
#   Packed-only mode!!!
#   Expects one npy per (StudyInstanceUID, vertebra) with:
#       filename: {uid}_{cid}.npy
#       shape:    (T, H, W, 6)
#       channels: 0..4 intensity, 5 mask
#

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A


# -------------------------
# Tiny utils
# -------------------------
def _as_float01(x: np.ndarray) -> np.ndarray:
    # convert image data
    # uint8 [0..255] to float32 [0..1]
    if x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0
    return x.astype(np.float32)


def _assert_packed_ok(seq: np.ndarray, fp: str, T: int) -> None:
    if not isinstance(seq, np.ndarray):
        raise TypeError(f"Loaded object is not np.ndarray:{fp}")
    if seq.ndim != 4:
        raise ValueError(f"Expected packed shape (T,H,w,6), got {seq.shape} in {fp}")
    if seq.shape[-1] != 6:
        raise ValueError(f"Expected 6 channels last dim, but got {seq.shape} in {fp}")
    if seq.shape[0] != T:
        raise ValueError(f"Expected T={T}, got T={seq.shape[0]} in {fp}")


def normalize_intensity_and_mask(arr_hw6: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """
    arr_hw6: (H,W,6)
    0..4 are intensity channels, 5 is mask channel.

    We do:
      - intensities: scale to [0,1] if uint8 then normalize it by mean/std
      - mask: binarize -> float {0,1}
    """
    if arr_hw6.ndim != 3 or arr_hw6.shape[-1] != 6:
        raise ValueError(f"Expected (H,W,6), got {arr_hw6.shape}")

    intensity_ch = int(cfg.get("intensity_chans", 5))
    if intensity_ch != 5:
        # Hard assumption of this project: 5 intensity + 1 mask
        raise ValueError(f"intensity_chans must be 5, got {intensity_ch}")

    mean = float(cfg.get("intensity_mean", 0.5))
    std = float(cfg.get("intensity_std", 0.25))
    eps = 1e-6

    x = arr_hw6.astype(np.float32, copy=False)

    intens = _as_float01(x[..., :5])  # (H,W,5)
    mask = x[..., 5:6]                # (H,W,1)

    # mask might be uint8 {0,255} or float {0,1}
    if mask.dtype == np.uint8:
        mask = (mask > 127).astype(np.float32)
    else:
        mask = (mask > 0.5).astype(np.float32)

    intens = (intens - mean) / (std + eps)

    return np.concatenate([intens, mask], axis=-1)


# -------------------------
# Transforms
# -------------------------
def build_transforms(cfg: Dict[str, Any], is_train: bool) -> Tuple[Optional[A.Compose], Optional[A.Compose]]:
    """
    Returns:
      geo_tf: geometry transforms for intensity+mask together
      photo_tf: intensity-only transforms
    """
    image_size = int(cfg.get("image_size", 224))

    if is_train:
        geo_tf = A.Compose(
            [
                A.Resize(image_size, image_size, p=1.0),
                A.HorizontalFlip(p=0.5), # Note for self: Should we flip vertabrae horizontally? Or is this medically speaking not that smart?
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=0,
                    p=0.5,
                ),
            ],
            additional_targets={"mask": "mask"},
        )

        # Cutout is deprecated apperently-> CoarseDropout
        photo_tf = A.Compose(
            [
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.GaussNoise(var_limit=(2.0, 7.0), p=0.3),
                A.CoarseDropout(
                    max_holes=2,
                    max_height=int(image_size * 0.2),
                    max_width=int(image_size * 0.2),
                    min_holes=1,
                    fill_value=0,
                    p=0.3,
                ),
            ]
        )
    else:
        geo_tf = A.Compose(
            [A.Resize(image_size, image_size, p=1.0)],
            additional_targets={"mask": "mask"},
        )
        photo_tf = None

    return geo_tf, photo_tf


# -------------------------
# Dataset
# -------------------------
class CLSDataset(Dataset):
    """
    Per item:
      x:   (T, 6, H, W) float32
      y:  () float32
      cid: () int64
      w:   () float32
      uid: str
    """

    def __init__(self, df, cfg: Dict[str, Any], mode: str):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.mode = mode

        self.data_dir = str(cfg["data_dir"])
        self.T = int(cfg.get("n_slice_per_c", 15))

        self.geo_tf, self.photo_tf = build_transforms(cfg, is_train=(mode == "train"))

        # Default weights if cfg doesn't specify, the weights are adjusted since some vertebrae (3 for example) fracture way less
        # and are therefore underrepresented in training data.
        self.vertebra_weights = cfg.get(
            "vertebra_weights",
            {1: 2.69, 2: 1.38, 3: 5.38, 4: 3.64, 5: 2.43, 6: 1.42, 7: 1.0},
        )

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"data_dir not found: {self.data_dir}")

        for col in ["StudyInstanceUID", "c", "label"]:
            if col not in self.df.columns:
                raise ValueError(f"Dataset df missing column: {col}")

    def __len__(self) -> int:
        return len(self.df)

    def _load_packed(self, uid: str, cid: int) -> np.ndarray:
        fp = os.path.join(self.data_dir, f"{uid}_{cid}.npy")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing packed npy: {fp}")

        seq = np.load(fp, allow_pickle=False)
        _assert_packed_ok(seq, fp, self.T)
        return seq

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        uid = str(row.StudyInstanceUID)
        cid = int(row.c)

        seq = self._load_packed(uid, cid)  # (T,H,W,6)

        frames = []
        for t in range(self.T):
            arr = seq[t]  # (H,W,6)

            intensity = arr[..., :5]
            mask = arr[..., 5]

            if self.geo_tf is not None:
                res = self.geo_tf(image=intensity, mask=mask)
                intensity = res["image"]
                mask = res["mask"]

            if self.photo_tf is not None:
                res2 = self.photo_tf(image=intensity)
                intensity = res2["image"]

            if mask.ndim == 2:
                mask = mask[..., None]

            arr2 = np.concatenate([intensity, mask], axis=-1)  # (H,W,6)
            arr2 = normalize_intensity_and_mask(arr2, self.cfg)
            arr2 = arr2.transpose(2, 0, 1)  # (6,H,W)

            frames.append(arr2)

        x = np.stack(frames, axis=0).astype(np.float32)  # (T,6,H,W)

        y = float(row.label)
        w = float(self.vertebra_weights.get(cid, 1.0))

        return (
            torch.from_numpy(x).float(),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(cid, dtype=torch.long),
            torch.tensor(w, dtype=torch.float32),
            uid,
        )


# -------------------------
# Loader factory (what train.py expects)
# -------------------------
def make_loaders(train_df, val_df, cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Note FOR SELF:Ran into issues in the past within train.py, this is quite a delicate function somehow.
    Keep it small: create datasets and dataloaders.
    """
    bs = int(cfg.get("batch_size", 4))
    nw = int(cfg.get("num_workers", 0))

    ds_tr = CLSDataset(train_df, cfg=cfg, mode="train")
    ds_va = CLSDataset(val_df, cfg=cfg, mode="valid")

    loader_tr = DataLoader(
        ds_tr,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
    )
    loader_va = DataLoader(
        ds_va,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        drop_last=False,
    )

    return loader_tr, loader_va

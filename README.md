
# Cervical Spine Fracture Classification (Stage 2)

This repository contains the **stage 2 prototype** for cervical spine (C1–C7) fracture classification.
The focus is on **vertebra-level classification** with **patient-level aggregation**.

The code is written as a **research prototype and is intended to be run locally**.

---

## Project Overview

### Goal

Detect cervical spine fractures (C1–C7) from preprocessed 2D slice sequences derived from CT scans.

The data format and labeling follow the RSNA 2022 Cervical Spine Fracture Detection challenge:
[https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection)

This repository does not include raw data or preprocessing steps for DICOM conversion. 

---

### Approach

* 2D CNN feature extraction per slice (EfficientNetV2 via `timm`)
* Small CNN encoder for vertebra mask (region-of-interest signal)
* Temporal modeling with a bidirectional LSTM
* Attention-based temporal pooling
* Binary classification per vertebra (fracture / no fracture)
* Patient-level prediction via max aggregation over vertebra probabilities

---

### Design Choice

This model intentionally avoids a full 3D CNN. To keep it lightweight and efficient.

Slice-wise feature extraction combined with temporal modeling:

* reduces memory requirements
* simplifies training and debugging
* keeps the architecture lightweight and interpretable

---

## Repository Structure

cspine-fracture-cls/
├─ README.md
├─ requirements.txt
├─ configs/
│  ├─ train.yaml        # training configuration
│  └─ infer.yaml        # inference configuration
├─ src/cspine_cls/
│  ├─ common.py         # seeds, device selection, logging, checkpoint IO
│  ├─ data.py           # dataset + transforms
│  ├─ model.py          # CNN + bidirectional LSTM + attention
│  ├─ train.py          # training & validation loops
│  ├─ metrics.py        # recall / F1 helpers + threshold sweep
│  └─ inference.py     # patient-level inference
└─ scripts/
├─ train.py          # training entrypoint
└─ infer.py          # inference entrypoint

---

## Data Assumptions:

### Input Format (Packed)

Training and inference both expect **packed `.npy` files per vertebra**:

{StudyInstanceUID}_{c}.npy

Where:

* `c` ∈ {1,…,7} corresponds to vertebra C1–C7

**Shape**
(T, H, W, 6)

**Channels**

* 0..4 : intensity slices (normalized)
* 5    : vertebra mask (binary, 0/1)

`T` (number of slices per vertebra) is configured via `n_slice_per_c` in the config.

---

## Training

Training is launched via the script entrypoint:

python scripts/train.py --config configs/train.yaml

### Key Characteristics of project

* Patient-level train/validation split
  (no overlap of StudyInstanceUID between sets)
* Focal loss with an automatically derived alpha
* Vertebra-specific sample weighting to compensate for more commonly fractured vertebrae
* Threshold selection on validation data
* Configuration snapshot is saved per run

Outputs are written to:

outputs/<run_name>/

Each run directory contains:

* model checkpoints
* training logs
* the exact configuration used

---

## Inference

Inference for a single patient:

python scripts/infer.py --config configs/infer.yaml --uid <StudyInstanceUID>

### Output

* Per-vertebra probability and binary prediction
* Patient-level probability (max over vertebrae)
* Patient-level binary prediction

The decision threshold is loaded from the checkpoint unless explicitly overridden in the config.

---

## Metrics & Thresholding

* recall, precision, F1, and accuracy are computed
* Thresholds are selected on the validation set via a configurable sweep
* Recall is treated as the primary optimization objective, given that reducing concluding that there isnt a fracture
* while there is, is medically important. 

Metrics are intended for **model comparison and development**

---

## Notes & Limitations

* This is a **research prototype meant to run locally**, not a production-ready system
* No DICOM handling or preprocessing is included in this project

---

## Dependencies

See `requirements.txt`.

Tested with:

* Python 3.9+
* PyTorch
* timm
* albumentations
* scikit-learn

---


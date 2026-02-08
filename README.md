
# Cervical Spine Fracture Classification (Stage 2)

This repository contains the **stage 2 prototype** for cervical spine (C1–C7) fracture classification.
The focus is on **vertebra-level classification** with patient-level aggregation.

The code is written as a research prototype and is intended to be run locally.

---

## Project Overview

### Goal

Detect cervical spine fractures (C1–C7) from preprocessed 2D slice sequences derived from CT scans.
Stage 1 converted the 2d slice sequences to numpy arrays for stage 2 
 
This repository does not include raw data or preprocessing steps for DICOM conversion. 

---

### Data and Results

The data format and labeling follow the RSNA 2022 Cervical Spine Fracture Detection challenge:
[https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection)
So a special thanks to Kaggle and the RSNA.\

Each study corresponds to one CT scan and contains binary fracture labels for vertebrae C1–C7.

For this stage-2 prototype, patient-level annotations are expanded to vertebra-level samples:

One StudyInstanceUID → up to 7 vertebra samples (C1–C7)

Each vertebra is treated as a separate binary classification task for the model

A patient is considered positive if any vertebra is fractured (note: multiple vertebra can also be fractured at the same time)

**Class distribution**

The dataset is imbalanced, especially at the vertebra level.

Mean positive rate per label (= prevalence of fracture):

Patient overall: 47.6%

C1: 7.2%

C2: 14.1%

C3: 3.6%

C4: 5.3%

C5: 8.0%

C6: 13.7%

C7: 19.5%

This imbalance requires thinking about:

* recall-focused optimization
* per-vertebra weighting
* careful threshold selection

Below is an example of the first 3 patients in the data. Where only patient 2 has no fractures, patient 1 has c1 and c2 fractured 
and patient 3 has only c5 fractured.

```text
| StudyInstanceUID | patient_overall | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
| ---------------- | --------------- | -- | -- | -- | -- | -- | -- | -- |
| 1.2.826.0.1.…    | 1               | 1  | 1  | 0  | 0  | 0  | 0  | 0  |
| 1.2.826.0.1.…    | 0               | 0  | 0  | 0  | 0  | 0  | 0  | 0  |
| 1.2.826.0.1.…    | 1               | 0  | 0  | 0  | 0  | 1  | 0  | 0  |
```
Note: Expansion is performed after patient-level train/validation splitting to avoid data leakage.

**The following results correspond to a stage-2 prototype trained with:**

* patient-level train/validation split
* recall-focused objective
* threshold selected on validation data

**Validation performance (overall)**
* Recall: 0.8885 (missed 11.5% of actual fractures)
* Precision: 0.2440 (only 24.4% labeled fractured is an actual fracture)
* F1-score: 0.3828
* Selected threshold: 0.45

High recall is prioritized to minimize false negatives, which is desirable for clinical screening or triage scenarios. Lower precision is expected given the class imbalance. This reflects the same issues that are reported in English hospitals where 19% of centres identified missed cervical spine injuries after they passed the ct-scans clearance protocols (https://boneandjoint.org.uk/Article/10.1302/0301-620X.98B6.37435). Detected spinal fractures seems to be quite a complex task, given the complexitity of the bone structures.

### Approach

* 2D CNN feature extraction per slice (EfficientNetV2)
* Small CNN encoder for vertebra mask (region-of-interest signal)
* Temporal modeling with a bidirectional LSTM
* Attention-based temporal pooling
* Binary classification per vertebra (fracture / no fracture)
---

### Design Choice

This model intentionally avoids a full 3D CNN or vision transformer. This is to keep it lightweight and efficient.

Slice-wise feature extraction combined with temporal modeling:

* reduces memory requirements of local computer
* simplifies the training and debugging
* keeps the architecture lightweight and interpretable

---

## Repository Structure

```text
cspine-fracture-cls/
├── README.md
├── requirements.txt
├── configs/
│   ├── train.yaml        # training configuration
│   └── infer.yaml        # inference configuration
├── src/cspine_cls/
│   ├── common.py         # seeds, device selection, logging, checkpoint IO
│   ├── data.py           # dataset + transforms
│   ├── model.py          # CNN + bidirectional LSTM + attention
│   ├── train.py          # training & validation loops
│   ├── metrics.py        # recall / F1 helpers + threshold sweep
│   └── inference.py      # patient-level inference
└── scripts/
    ├── train.py          # training entrypoint
    └── infer.py          # inference entrypoint
```
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

So to run, type in terminal: "python scripts/train.py --config configs/train.yaml"

### Key Characteristics of project

* Patient-level train/validation split
  (no overlap of StudyInstanceUID between sets)
* Focal loss with an automatically derived alpha
* Different weighting per vertabreato compensate for more commonly or less commonly fractured vertebrae
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

Note: The decision threshold is loaded from the checkpoint unless explicitly overridden in the config.

---

## Metrics & Thresholding

* recall, precision, F1, and accuracy are computed
* Thresholds are selected on the validation set via a configurable sweep
* Recall is treated as the primary optimization objective, given that reducing concluding that there isnt a fracture
* while there is, is medically important. 

Note: Metrics are intended for model comparison and development

---

## Notes & Limitations

* This is a research prototype meant to run locally, not a full system ready for production.
* No DICOM handling or preprocessing is included in this project yet. 

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


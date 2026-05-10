import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cspine_cls.data import normalize_intensity_and_mask
from cspine_cls.inference import load_vertebra_sequence
from cspine_cls.metrics import threshold_sweep
from scripts.pack_sequences import pack_sequences


class CoreBehaviorTests(unittest.TestCase):
    def test_uint8_intensity_is_scaled_before_normalization(self):
        arr = np.zeros((2, 2, 6), dtype=np.uint8)
        arr[..., :5] = 128
        arr[..., 5] = 255

        out = normalize_intensity_and_mask(arr, {"intensity_mean": 0.0, "intensity_std": 1.0})

        self.assertAlmostEqual(float(out[..., 0].mean()), 128.0 / 255.0, places=6)
        self.assertEqual(float(out[..., 5].min()), 1.0)
        self.assertEqual(float(out[..., 5].max()), 1.0)

    def test_threshold_sweep_reports_unmet_precision_floor(self):
        y_true = np.array([0, 0, 1, 1], dtype=np.float32)
        y_prob = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)

        result = threshold_sweep(
            y_true,
            y_prob,
            {"threshold_objective": "recall", "precision_floor": 0.99, "thresholds": [0.5, 0.7]},
        )

        self.assertFalse(result["precision_floor_met"])
        self.assertIn(result["th"], {0.5, 0.7})

    def test_pack_sequences_writes_packed_vertebra_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            dst = os.path.join(tmp, "dst")
            os.makedirs(src)

            for t in range(2):
                arr = np.full((3, 4, 6), t, dtype=np.uint8)
                np.save(os.path.join(src, f"study.with.dots_1_{t}.npy"), arr)

            stats = pack_sequences(src, dst, n_slices=2)
            packed = np.load(os.path.join(dst, "study.with.dots_1.npy"), allow_pickle=False)

            self.assertEqual(stats["written"], 1)
            self.assertEqual(packed.shape, (2, 3, 4, 6))
            self.assertEqual(int(packed[1, 0, 0, 0]), 1)

    def test_inference_loader_reads_packed_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.zeros((2, 3, 4, 6), dtype=np.uint8)
            arr[..., :5] = 255
            arr[..., 5] = 255
            np.save(os.path.join(tmp, "study_2.npy"), arr)

            cfg = {
                "data_dir": tmp,
                "n_slice_per_c": 2,
                "intensity_mean": 0.0,
                "intensity_std": 1.0,
            }

            with patch("cspine_cls.inference.build_transforms", return_value=(None, None)):
                seq = load_vertebra_sequence("study", 2, cfg)

            self.assertEqual(seq.shape, (2, 6, 3, 4))
            self.assertAlmostEqual(float(seq[:, :5].min()), 1.0, places=5)
            self.assertEqual(float(seq[:, 5].min()), 1.0)


if __name__ == "__main__":
    unittest.main()

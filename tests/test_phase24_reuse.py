"""Guard reuse of historical outer predictions against row and fold mismatches."""

import unittest

import numpy as np

from src.numpy_boosting import HistogramBinner
from tools.phase24_original_restricted_imputation import collapsed_binary_flags, verify_rows


class TestOriginalImputationReuse(unittest.TestCase):
    def setUp(self):
        self.development = np.array([4, 1, 3, 0], dtype=np.int64)
        self.labels = np.array([0, 1, 0, 1, 0], dtype=np.int8)
        self.outer = [(np.array([1, 3]), np.array([0, 2])),
                      (np.array([0, 2]), np.array([1, 3]))]
        self.arrays = {"development_indices": self.development.copy(),
                       "labels": self.labels[self.development].copy(),
                       "fold_ids": np.array([1, 2, 1, 2], dtype=np.int8),
                       "probabilities": np.array([0.1, 0.4, 0.7, 0.2])}

    def test_accepts_exact_order_and_partition(self):
        verify_rows(self.arrays, self.development, self.labels, self.outer)

    def test_rejects_reordered_rows(self):
        self.arrays["development_indices"] = self.development[::-1]
        with self.assertRaises(AssertionError):
            verify_rows(self.arrays, self.development, self.labels, self.outer)

    def test_rejects_changed_folds(self):
        self.arrays["fold_ids"][0] = 2
        with self.assertRaises(AssertionError):
            verify_rows(self.arrays, self.development, self.labels, self.outer)

    def test_rejects_changed_labels(self):
        self.arrays["labels"][0] = 1
        with self.assertRaises(AssertionError):
            verify_rows(self.arrays, self.development, self.labels, self.outer)

    def test_rejects_incomplete_or_invalid_scores(self):
        for scores in (np.array([0.1]), np.array([0.1, np.nan, 0.3, 0.4]),
                       np.array([0.1, 0.2, 0.3, 1.1])):
            with self.subTest(scores=scores):
                self.arrays["probabilities"] = scores
                with self.assertRaises(ValueError):
                    verify_rows(self.arrays, self.development, self.labels, self.outer)

    def test_unused_allocated_bin_does_not_preserve_rare_flag(self):
        matrix = np.zeros((100, 303), dtype=np.float32)
        matrix[-1, 295] = 1
        matrix[50:, 296:] = 1
        names = [f"source_{i}" for i in range(295)] + ["rare"] + [f"common_{i}" for i in range(7)]
        binner = HistogramBinner(64).fit(matrix)
        self.assertEqual(binner.bin_counts_[295], 2)
        self.assertEqual(collapsed_binary_flags(names, binner), ["rare"])


if __name__ == "__main__":
    unittest.main()

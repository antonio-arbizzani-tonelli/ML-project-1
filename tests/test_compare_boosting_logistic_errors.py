"""Small hand-checkable paired-error and alignment tests."""

import unittest

import numpy as np

from src.compare_boosting_logistic_errors import (
    align_to_boosting,
    margin_counts,
    paired_error_margins,
    paired_outcomes,
)


class TestPairedErrorComparison(unittest.TestCase):
    def test_original_row_alignment_preserves_labels_and_folds(self):
        boosting = {
            "indices": np.array([30, 10, 20]),
            "labels": np.array([1, 0, 1]),
            "folds": np.array([2, 1, 3]),
        }
        logistic = {
            "indices": np.array([20, 30, 10]),
            "labels": np.array([1, 1, 0]),
            "folds": np.array([3, 2, 1]),
            "probabilities": np.array([0.2, 0.3, 0.1]),
            "thresholds": np.array([0.2, 0.2, 0.2]),
            "predictions": np.array([1, 1, 0]),
        }
        aligned = align_to_boosting(boosting, logistic)
        np.testing.assert_array_equal(aligned["indices"], boosting["indices"])
        np.testing.assert_array_equal(aligned["probabilities"], [0.3, 0.1, 0.2])
        logistic["folds"][1] = 1
        with self.assertRaisesRegex(ValueError, "folds disagree"):
            align_to_boosting(boosting, logistic)

    def test_margin_bands_and_paired_errors(self):
        labels = np.array([1, 1, 0, 0], dtype=np.int8)
        boosting = {
            "probabilities": np.array([0.205, 0.05, 0.205, 0.40]),
            "thresholds": np.full(4, 0.20),
            "predictions": np.array([1, 0, 1, 1], dtype=np.int8),
        }
        logistic = {"predictions": np.array([0, 1, 0, 1], dtype=np.int8)}
        joint = paired_outcomes(
            labels, boosting["predictions"], logistic["predictions"]
        )
        self.assertEqual(joint["actual_positive"]["boosting_only_correct"], 1)
        self.assertEqual(joint["actual_positive"]["logistic_only_correct"], 1)
        self.assertEqual(joint["actual_negative"]["logistic_only_correct"], 1)
        self.assertEqual(joint["actual_negative"]["both_wrong"], 1)
        paired = paired_error_margins(labels, boosting, logistic)
        self.assertEqual(paired["false_negative"]["logistic_correct"][">=0.10"], 1)
        self.assertEqual(paired["false_positive"]["logistic_correct"]["<0.01"], 1)
        self.assertEqual(paired["false_positive"]["both_wrong"][">=0.10"], 1)
        self.assertEqual(
            sum(
                margin_counts(
                    np.ones(4, dtype=bool),
                    boosting["probabilities"],
                    boosting["thresholds"],
                ).values()
            ),
            4,
        )


if __name__ == "__main__":
    unittest.main()

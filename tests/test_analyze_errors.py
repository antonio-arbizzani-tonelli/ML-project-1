"""Checks for the nested out-of-fold error-analysis helpers."""

import unittest

import numpy as np

from src.analyze_errors import (
    _apply_zero_codes,
    _cross_group_rows,
    _continuous_groups,
    _error_names,
    _group_rows,
    _thresholds_for_rows,
)


class TestErrorAnalysis(unittest.TestCase):
    def test_thresholds_map_rows_to_their_own_outer_fold(self):
        folds = np.array([2, 1, 2, 3, 1])
        nested = {
            "folds": [
                {"outer_fold": 1, "inner_threshold": 0.2},
                {"outer_fold": 2, "inner_threshold": 0.3},
                {"outer_fold": 3, "inner_threshold": 0.4},
            ]
        }
        np.testing.assert_allclose(
            _thresholds_for_rows(folds, nested), np.array([0.3, 0.2, 0.3, 0.4, 0.2])
        )

    def test_zero_day_codes_are_normalized_before_binning(self):
        values = _apply_zero_codes(np.array([0.0, 1.0, 88.0, np.nan]), [88])
        groups = _continuous_groups(
            values, [-1.0, 1.0, 14.0, 30.0, 100.0], ["0", "1-13", "14-29", "30"]
        )
        np.testing.assert_array_equal(groups, np.array(["0", "1-13", "0", "missing"]))

    def test_group_rates_are_conditional_on_actual_class(self):
        labels = np.array([1, 1, 0, 0, 1, 0], dtype=np.int8)
        predictions = np.array([0, 1, 1, 0, 0, 0], dtype=np.int8)
        errors = _error_names(labels, predictions)
        groups = np.array(["A", "A", "A", "B", "B", "B"])
        rows = _group_rows(
            "feature", labels, predictions, errors, groups, ["A", "B"], 1
        )

        self.assertEqual(rows[0]["false_negative"], 1)
        self.assertAlmostEqual(rows[0]["false_negative_rate"], 0.5)
        self.assertAlmostEqual(rows[0]["false_positive_rate"], 1.0)
        self.assertAlmostEqual(rows[1]["false_negative_rate"], 1.0)
        self.assertAlmostEqual(rows[1]["false_positive_rate"], 0.0)

    def test_cross_group_rows_retain_the_outer_fold_breakdown(self):
        labels = np.array([1, 0, 1, 0], dtype=np.int8)
        predictions = np.array([0, 1, 1, 0], dtype=np.int8)
        errors = _error_names(labels, predictions)
        rows = _cross_group_rows(
            ("age", "health"),
            np.array(["young", "young", "old", "old"]),
            np.array(["good", "good", "poor", "poor"]),
            labels,
            predictions,
            errors,
            np.array([1, 1, 2, 2]),
            1,
        )

        self.assertEqual(len(rows), 2)
        young = next(row for row in rows if row["group"] == "young | good")
        self.assertAlmostEqual(young["false_negative_rate"], 1.0)
        self.assertAlmostEqual(young["false_positive_rate"], 1.0)
        self.assertEqual(young["folds"][0]["fold"], 1)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the reproducible statistical dataset analysis."""

import unittest

import numpy as np

from src.analyze_dataset import (
    allowed_numeric_mask,
    numeric_code_intervals,
    stratified_split,
    total_variation,
    validation_intervals,
)


class TestAnalyzeDataset(unittest.TestCase):
    """Check core statistics and codebook interpretation on small fixtures."""

    def test_total_variation(self):
        self.assertAlmostEqual(
            total_variation({"a": 50, "b": 50}, {"a": 20, "b": 80}),
            0.3,
        )
        self.assertEqual(total_variation({}, {"a": 1}), 0.0)

    def test_stratified_split_on_twenty_alternating_labels(self):
        # Ten respondents per class: a 20% validation split must take two
        # positives and two negatives, regardless of their row positions.
        labels = np.asarray([-1, 1] * 10, dtype=np.int8)
        development, validation = stratified_split(labels, 0.2, seed=7)
        development_again, validation_again = stratified_split(labels, 0.2, seed=7)

        np.testing.assert_array_equal(development, development_again)
        np.testing.assert_array_equal(validation, validation_again)
        self.assertEqual(len(development), 16)
        self.assertEqual(len(validation), 4)
        np.testing.assert_array_equal(
            np.sort(np.concatenate((development, validation))),
            np.arange(len(labels)),
        )
        self.assertEqual(int((labels[validation] == 1).sum()), 2)
        self.assertEqual(int((labels[validation] == -1).sum()), 2)
        self.assertEqual(int((labels[development] == 1).sum()), 8)
        self.assertEqual(int((labels[development] == -1).sum()), 8)

    def test_stratified_split_rejects_invalid_fraction(self):
        labels = np.asarray([-1, 1] * 10, dtype=np.int8)
        for fraction in (0.0, 1.0, -0.1, 1.1):
            with self.subTest(fraction=fraction):
                with self.assertRaises(ValueError):
                    stratified_split(labels, fraction, seed=7)

    def test_codebook_intervals_apply_documented_implied_decimals(self):
        documented = [
            {
                "code": "91 - 244",
                "label": "Height in meters [2 implied decimal places]",
            }
        ]
        intervals = numeric_code_intervals(documented)
        self.assertEqual(intervals, [(0.91, 2.44)])
        observed = np.asarray([0.90, 0.91, 1.75, 2.44, 2.45])
        np.testing.assert_array_equal(
            allowed_numeric_mask(observed, intervals),
            np.asarray([False, True, True, True, False]),
        )

    def test_implied_decimal_scale_applies_to_special_codes(self):
        documented = [
            {"code": "0 - 501", "label": "Measurement (two implied decimal places)"},
            {"code": "99900", "label": "Missing"},
        ]
        self.assertEqual(
            numeric_code_intervals(documented), [(0.0, 5.01), (999.0, 999.0)]
        )

    def test_codebook_intervals_include_exact_special_codes(self):
        documented = [
            {"code": "1 - 5", "label": "Substantive response"},
            {"code": "7", "label": "Don't know"},
            {"code": "9", "label": "Refused"},
            {"code": "BLANK", "label": "Not asked"},
        ]
        intervals = numeric_code_intervals(documented)
        observed = np.asarray([1.0, 5.0, 6.0, 7.0, 9.0])
        np.testing.assert_array_equal(
            allowed_numeric_mask(observed, intervals),
            np.asarray([True, True, False, True, True]),
        )

    def test_validation_skips_domains_with_only_exception_codes(self):
        entry = {
            "documented_values": [
                {"code": "777777", "label": "Don't know"},
                {"code": "999999", "label": "Refused"},
            ],
            "special_values": [
                {"code": "777777", "label": "Don't know"},
                {"code": "999999", "label": "Refused"},
            ],
        }
        self.assertEqual(validation_intervals(entry), [])


if __name__ == "__main__":
    unittest.main()

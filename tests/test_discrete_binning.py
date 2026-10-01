"""Regression checks for discrete states and pure missing-value splits."""

import unittest
from pathlib import Path

import numpy as np

from src.numpy_boosting import HistogramBinner, HistogramGradientBoostingClassifier


class TestDiscreteBinning(unittest.TestCase):
    def test_rare_binary_state_survives(self):
        values = np.r_[np.zeros(99), 1.0].reshape(-1, 1)
        old = HistogramBinner(64).fit_transform(values)
        exact = HistogramBinner(64, "exact_low_cardinality")
        new = exact.fit_transform(values)
        self.assertEqual(np.unique(old).size, 1)
        self.assertEqual(np.unique(new).size, 2)
        self.assertTrue(exact.exact_columns_[0])

    def test_exact_limit_and_quantile_fallback(self):
        values = np.column_stack((np.arange(256) % 64, np.arange(256)))
        old = HistogramBinner(64).fit(values)
        new = HistogramBinner(64, "exact_low_cardinality").fit(values)
        np.testing.assert_array_equal(new.exact_columns_, [True, False])
        self.assertEqual(np.unique(new.transform(values)[:, 0]).size, 64)
        np.testing.assert_array_equal(old.edges_[1], new.edges_[1])
        np.testing.assert_array_equal(old.transform(values)[:, 1], new.transform(values)[:, 1])

    def test_unseen_values_endpoints_and_missing(self):
        new = HistogramBinner(64, "exact_low_cardinality").fit(np.array([[2], [4], [6]]))
        actual = new.transform(np.array([[1], [2], [3], [4], [5], [6], [7], [np.nan]]))[:, 0]
        np.testing.assert_array_equal(actual, [0, 0, 0, 1, 1, 2, 2, 255])

    def test_maximum_bin_count_keeps_missing_distinct(self):
        binner = HistogramBinner(255, "exact_low_cardinality").fit(np.arange(255).reshape(-1, 1))
        result = binner.transform(np.r_[np.arange(255), np.nan].reshape(-1, 1))[:, 0]
        self.assertEqual(np.unique(result[:-1]).size, 255)
        self.assertEqual(result[-1], 255)
        self.assertNotIn(255, result[:-1])

    def test_constant_and_empty_columns(self):
        values = np.array([[7, np.nan], [7, np.nan], [np.nan, np.nan]])
        binner = HistogramBinner(64, "exact_low_cardinality").fit(values)
        np.testing.assert_array_equal(binner.bin_counts_, [1, 1])
        np.testing.assert_array_equal(binner.transform(values) == 255, np.isnan(values))

    def test_float32_adjacent_states_do_not_need_midpoints(self):
        lower = np.float32(1.0)
        upper = np.nextafter(lower, np.float32(2.0))
        values = np.array([[lower], [upper]], dtype=np.float32)
        result = HistogramBinner(64, "exact_low_cardinality").fit_transform(values)
        self.assertEqual(np.unique(result).size, 2)

    def test_pure_missing_split_with_multiple_observed_states(self):
        values = np.array([[0], [0], [1], [1], [np.nan], [np.nan]])
        labels = np.array([0, 0, 0, 0, 1, 1])
        for strategy in ("recursive", "levelwise"):
            with self.subTest(strategy=strategy):
                model = HistogramGradientBoostingClassifier(n_estimators=1, max_depth=1,
                    min_samples_leaf=2, binning_strategy="exact_low_cardinality",
                    histogram_strategy=strategy).fit(values, labels)
                scores = model.decision_function(values)
                np.testing.assert_array_equal(scores[:4], np.repeat(scores[0], 4))
                self.assertGreater(scores[4], scores[0])
                self.assertEqual(model.trees_[0].root_.split_bin, 1)
                self.assertFalse(model.trees_[0].root_.missing_go_left)

    def test_exact_checkpoint_resume_and_legacy_defaults(self):
        values = np.array([[0], [0], [1], [1], [np.nan], [np.nan]])
        labels = np.array([0, 0, 1, 1, 1, 1])
        settings = dict(max_depth=2, min_samples_leaf=1, random_seed=20)
        path = Path("results/test_exact_binning_checkpoint.pkl")
        try:
            complete = HistogramGradientBoostingClassifier(n_estimators=4,
                binning_strategy="exact_low_cardinality", **settings).fit(values, labels)
            partial = HistogramGradientBoostingClassifier(n_estimators=2,
                binning_strategy="exact_low_cardinality", **settings).fit(values, labels)
            partial.save_checkpoint(path)
            resumed, _ = HistogramGradientBoostingClassifier.load_checkpoint(path)
            resumed.continue_fit(values, labels, 2)
            np.testing.assert_array_equal(complete.decision_function(values), resumed.decision_function(values))
            legacy = HistogramGradientBoostingClassifier(n_estimators=2, **settings).fit(values, labels)
            del legacy.binning_strategy
            del legacy.binner_.binning_strategy
            del legacy.binner_.exact_columns_
            legacy.save_checkpoint(path)
            restored, _ = HistogramGradientBoostingClassifier.load_checkpoint(path)
            restored.continue_fit(values, labels, 2)
            original = HistogramGradientBoostingClassifier(n_estimators=4, **settings).fit(values, labels)
            np.testing.assert_array_equal(original.decision_function(values), restored.decision_function(values))
        finally:
            path.unlink(missing_ok=True)

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            HistogramBinner(64, "unknown")
        with self.assertRaises(ValueError):
            HistogramGradientBoostingClassifier(binning_strategy="unknown")


if __name__ == "__main__":
    unittest.main()

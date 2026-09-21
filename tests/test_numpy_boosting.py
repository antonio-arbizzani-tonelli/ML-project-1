"""Hand-checkable tests for the NumPy histogram boosting baseline."""

import unittest
from pathlib import Path

import numpy as np

from src.numpy_boosting import HistogramBinner, HistogramGradientBoostingClassifier


class TestHistogramBinner(unittest.TestCase):
    def test_binner_reserves_a_distinct_missing_bin(self):
        binner = HistogramBinner(n_bins=2).fit(np.array([[0.0], [1.0], [2.0], [3.0]]))

        transformed = binner.transform(np.array([[0.0], [3.0], [np.nan]]))

        self.assertEqual(transformed.dtype, np.uint8)
        self.assertEqual(int(transformed[2, 0]), 255)
        self.assertEqual(int(binner.bin_counts_[0]), 2)
        self.assertLess(int(transformed[0, 0]), int(binner.bin_counts_[0]))
        self.assertLess(int(transformed[1, 0]), int(binner.bin_counts_[0]))

    def test_binner_requires_fit_before_transform(self):
        with self.assertRaises(RuntimeError):
            HistogramBinner().transform(np.array([[1.0]]))


class TestHistogramGradientBoosting(unittest.TestCase):
    def test_learns_missingness_as_a_split_direction(self):
        features = np.array([[np.nan], [np.nan], [np.nan], [0.0], [0.0], [0.0]])
        labels = np.array([1, 1, 1, 0, 0, 0], dtype=np.int8)
        model = HistogramGradientBoostingClassifier(
            n_estimators=20,
            learning_rate=0.2,
            max_depth=1,
            min_samples_leaf=2,
            n_bins=2,
            random_seed=3,
        ).fit(features, labels)

        probabilities = model.predict_proba(np.array([[np.nan], [0.0]]))[:, 1]

        self.assertGreater(probabilities[0], 0.8)
        self.assertLess(probabilities[1], 0.2)

    def test_depth_and_feature_sampling_are_configurable_and_reproducible(self):
        features = np.array(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
        )
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
        settings = dict(
            n_estimators=12,
            learning_rate=0.1,
            max_depth=2,
            min_samples_leaf=1,
            n_bins=2,
            max_features=1,
            random_seed=17,
        )

        first = HistogramGradientBoostingClassifier(**settings).fit(features, labels)
        second = HistogramGradientBoostingClassifier(**settings).fit(features, labels)

        np.testing.assert_allclose(
            first.decision_function(features), second.decision_function(features)
        )
        self.assertTrue(all(tree.max_depth == 2 for tree in first.trees_))

    def test_rejects_invalid_feature_sampling_fraction(self):
        with self.assertRaises(ValueError):
            HistogramGradientBoostingClassifier(max_features=1.5)

    def test_rejects_unknown_histogram_strategy(self):
        with self.assertRaises(ValueError):
            HistogramGradientBoostingClassifier(histogram_strategy="unknown")

    def test_staged_predictions_match_the_complete_fitted_model(self):
        features = np.array([[0.0], [0.0], [1.0], [1.0], [np.nan], [np.nan]])
        labels = np.array([0, 0, 1, 1, 1, 1], dtype=np.int8)
        model = HistogramGradientBoostingClassifier(
            n_estimators=6,
            learning_rate=0.1,
            max_depth=2,
            min_samples_leaf=1,
            n_bins=2,
            random_seed=5,
        ).fit(features, labels)

        staged = dict(model.staged_predict_proba(features, [2, 6]))

        self.assertEqual(len(model.tree_fit_seconds_), 6)
        self.assertTrue(all(seconds >= 0.0 for seconds in model.tree_fit_seconds_))
        np.testing.assert_allclose(staged[6], model.predict_proba(features)[:, 1])
        with self.assertRaises(ValueError):
            list(model.staged_predict_proba(features, [6, 2]))

    def test_saved_checkpoint_resumes_to_the_same_model(self):
        features = np.array([[0.0], [0.0], [1.0], [1.0], [np.nan], [np.nan]])
        labels = np.array([0, 0, 1, 1, 1, 1], dtype=np.int8)
        settings = dict(
            learning_rate=0.1,
            max_depth=2,
            min_samples_leaf=1,
            n_bins=2,
            random_seed=29,
        )
        complete = HistogramGradientBoostingClassifier(n_estimators=8, **settings).fit(
            features, labels
        )
        path = Path("results") / "test_numpy_boosting_checkpoint.pkl"
        path.unlink(missing_ok=True)
        try:

            def save_at_four(tree_count, model):
                if tree_count == 4:
                    model.save_checkpoint(path, {"training_rows": [0, 1, 2, 3, 4, 5]})

            HistogramGradientBoostingClassifier(n_estimators=4, **settings).fit(
                features, labels, [4], save_at_four
            )
            resumed, metadata = HistogramGradientBoostingClassifier.load_checkpoint(
                path
            )
            resumed.continue_fit(features, labels, 4)
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(metadata["training_rows"], [0, 1, 2, 3, 4, 5])
        np.testing.assert_allclose(
            resumed.decision_function(features), complete.decision_function(features)
        )


if __name__ == "__main__":
    unittest.main()

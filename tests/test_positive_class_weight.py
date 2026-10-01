"""Check the weighted logistic objective and compatibility with old models."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.numpy_boosting import HistogramGradientBoostingClassifier, _NewtonTree


class TestPositiveClassWeight(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[0.0], [0.0], [1.0], [1.0], [np.nan], [np.nan]])
        self.y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
        self.settings = dict(learning_rate=0.1, max_depth=2, min_samples_leaf=1,
                             n_bins=2, random_seed=29, positive_class_weight=2.0)

    def test_constant_model_starts_at_weighted_prevalence(self):
        model = HistogramGradientBoostingClassifier(n_estimators=1, min_samples_leaf=1,
                                                    positive_class_weight=2).fit(
            np.zeros((4, 1)), np.array([0, 0, 0, 1]))
        self.assertAlmostEqual(model.base_score_, np.log(2 / 3))
        np.testing.assert_allclose(model.predict_proba(np.zeros((4, 1)))[:, 1], 0.4)

    def test_gradient_and_hessian_match_weighted_loss_derivatives(self):
        seen = []
        original = _NewtonTree.fit

        def capture(tree, features, bins, gradients, hessians):
            seen.append((gradients.copy(), hessians.copy()))
            return original(tree, features, bins, gradients, hessians)

        with patch.object(_NewtonTree, "fit", capture):
            model = HistogramGradientBoostingClassifier(n_estimators=1, **self.settings).fit(self.x, self.y)
        scores = np.full(self.y.size, model.base_score_)
        weights = np.where(self.y == 1, 2.0, 1.0)

        def objective(values):
            return weights * (np.logaddexp(0.0, values) - self.y * values)

        epsilon = 1e-4
        first = (objective(scores + epsilon) - objective(scores - epsilon)) / (2 * epsilon)
        second = (objective(scores + epsilon) - 2 * objective(scores) + objective(scores - epsilon)) / epsilon**2
        np.testing.assert_allclose(seen[0][0], first, atol=1e-9, rtol=1e-8)
        np.testing.assert_allclose(seen[0][1], second, atol=1e-7, rtol=1e-6)

    def test_weighted_fit_matches_duplicated_positives_when_bins_and_leaf_constraints_match(self):
        settings = {**self.settings, "l2_regularization": 0.0}
        weighted = HistogramGradientBoostingClassifier(n_estimators=8, **settings).fit(self.x, self.y)
        rows = np.repeat(np.arange(self.y.size), np.where(self.y == 1, 2, 1))
        duplicated = HistogramGradientBoostingClassifier(n_estimators=8, **{**settings, "positive_class_weight": 1}).fit(self.x[rows], self.y[rows])
        np.testing.assert_allclose(weighted.predict_proba(self.x), duplicated.predict_proba(self.x), atol=1e-12)

    def test_weighted_checkpoint_continuation_matches_uninterrupted_fit(self):
        complete = HistogramGradientBoostingClassifier(n_estimators=8, **self.settings).fit(self.x, self.y)
        path = Path("results") / "test_positive_class_weight.pkl"
        self.assertFalse(path.exists())
        try:
            partial = HistogramGradientBoostingClassifier(n_estimators=4, **self.settings).fit(self.x, self.y)
            partial.save_checkpoint(path)
            resumed, _ = HistogramGradientBoostingClassifier.load_checkpoint(path)
            self.assertEqual(resumed.positive_class_weight, 2.0)
            resumed.continue_fit(self.x, self.y, 4)
        finally:
            path.unlink(missing_ok=True)
        np.testing.assert_array_equal(resumed.predict_proba(self.x), complete.predict_proba(self.x))

    def test_legacy_checkpoint_defaults_to_unit_weight_and_can_resume(self):
        settings = {**self.settings, "positive_class_weight": 1.0}
        complete = HistogramGradientBoostingClassifier(n_estimators=8, **settings).fit(self.x, self.y)
        path = Path("results") / "test_legacy_class_weight.pkl"
        self.assertFalse(path.exists())
        try:
            partial = HistogramGradientBoostingClassifier(n_estimators=4, **settings).fit(self.x, self.y)
            del partial.positive_class_weight
            partial.save_checkpoint(path)
            resumed, _ = HistogramGradientBoostingClassifier.load_checkpoint(path)
            self.assertEqual(resumed.positive_class_weight, 1.0)
            resumed.continue_fit(self.x, self.y, 4)
        finally:
            path.unlink(missing_ok=True)
        np.testing.assert_array_equal(resumed.predict_proba(self.x), complete.predict_proba(self.x))

    def test_rejects_nonpositive_or_nonfinite_weight(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                HistogramGradientBoostingClassifier(positive_class_weight=value)


if __name__ == "__main__":
    unittest.main()

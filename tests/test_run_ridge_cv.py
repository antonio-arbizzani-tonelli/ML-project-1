"""Checks for nested calibrated ridge evaluation helpers."""

import unittest

import numpy as np

from src.run_ridge_cv import (
    _apply_platt_calibrator,
    _fit_platt_calibrator,
    _validate_suite,
)


class TestRidgeCV(unittest.TestCase):
    def test_platt_calibration_returns_finite_probability_scores(self):
        scores = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        labels = np.array([0, 0, 0, 1, 1], dtype=np.int8)

        calibrator = _fit_platt_calibrator(scores, labels, max_iters=500, gamma=0.1)
        probabilities = _apply_platt_calibrator(scores, calibrator)

        self.assertTrue(np.all(np.isfinite(probabilities)))
        self.assertTrue(np.all((probabilities > 0.0) & (probabilities < 1.0)))
        self.assertGreater(probabilities[-1], probabilities[0])
        self.assertEqual(np.asarray(calibrator["weights"]).shape, (2,))

    def test_ridge_suite_requires_positive_regularization(self):
        suite = {
            "ridge_lambdas": [0.0],
            "nested_threshold": {"inner_fold_count": 2, "inner_seed": 1},
            "calibration": {"max_iters": 10, "gamma": 0.1},
        }
        with self.assertRaises(ValueError):
            _validate_suite(suite)


if __name__ == "__main__":
    unittest.main()

"""Focused checks for the three-fold boosting runner configuration helpers."""

import unittest
from pathlib import Path

import numpy as np

from src.run_boosting_cv import _save_oof_predictions, _tree_profile, _validate_suite


def _suite():
    return {
        "data": {},
        "feature_metadata_path": "metadata.json",
        "preprocessing_config_path": "preprocessing.json",
        "preprocessing_variant": "p2",
        "split_id": "development",
        "fold_count": 3,
        "seed": 4,
        "fixed_threshold": 0.5,
        "models": [{"name": "depth5", "parameters": {}, "checkpoints": [10, 20]}],
    }


class TestBoostingCV(unittest.TestCase):
    def test_accepts_strictly_increasing_checkpoints(self):
        _validate_suite(_suite())

    def test_rejects_checkpoint_that_would_retrain_or_skip_order(self):
        suite = _suite()
        suite["models"][0]["checkpoints"] = [20, 10]
        with self.assertRaises(ValueError):
            _validate_suite(suite)

    def test_nested_request_must_name_an_available_checkpoint(self):
        suite = _suite()
        suite["nested_threshold"] = {
            "inner_fold_count": 2,
            "inner_seed": 5,
            "model_checkpoints": [{"model": "depth5", "checkpoints": [30]}],
        }
        with self.assertRaises(ValueError):
            _validate_suite(suite)

    def test_tree_profile_preserves_all_individual_measurements(self):
        profile = _tree_profile([0.1, 0.2, 0.3])
        self.assertEqual(profile["seconds_per_tree"], [0.1, 0.2, 0.3])
        self.assertAlmostEqual(profile["total_seconds"], 0.6)
        self.assertAlmostEqual(profile["maximum_seconds"], 0.3)

    def test_oof_artifact_preserves_rows_scores_and_nested_thresholds(self):
        path = Path("results") / "fixture_oof.npz"
        path.unlink(missing_ok=True)
        try:
            path, checksum = _save_oof_predictions(
                Path("results"),
                "fixture",
                np.array([11, 21, 31]),
                np.array([1, 2, 1]),
                np.array([0, 1, 0]),
                np.array([0.2, 0.8, 0.1]),
                0.3,
                np.array([0.25, 0.75, 0.25]),
            )
            with np.load(path) as saved:
                np.testing.assert_array_equal(
                    saved["development_indices"], [11, 21, 31]
                )
                np.testing.assert_allclose(saved["probabilities"], [0.2, 0.8, 0.1])
                np.testing.assert_allclose(
                    saved["nested_thresholds"], [0.25, 0.75, 0.25]
                )
                np.testing.assert_array_equal(saved["nested_predictions"], [0, 1, 0])
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(len(checksum), 64)


if __name__ == "__main__":
    unittest.main()

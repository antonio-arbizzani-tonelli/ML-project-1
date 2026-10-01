"""Focused checks for the three-fold boosting runner configuration helpers."""

import unittest
from pathlib import Path
from contextlib import contextmanager

import numpy as np

from src.run_boosting_cv import (
    _array_sha256, _nested_evaluations, _save_inner_oof_predictions,
    _save_oof_predictions, _tree_profile, _validate_suite,
)
from src.run_preprocessing_cv import _stratified_folds


@contextmanager
def _inner_directory():
    directory = Path("results/test_inner_oof_fixture")
    directory.mkdir(parents=True, exist_ok=True)
    try:
        yield directory
    finally:
        for path in directory.iterdir():
            path.unlink()
        directory.rmdir()


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

    def test_inner_oof_preserves_exact_rows_labels_scores_and_hashes(self):
        with _inner_directory() as directory:
            info = _save_inner_oof_predictions(Path(directory), "fixture", 1, 23,
                np.array([11, 21, 31, 41]), np.array([51, 61]),
                np.array([1, 2, 1, 2]), np.array([0, 1, 1, 0]),
                np.array([0.2, 0.8, 0.7, 0.1]))
            with np.load(info["path"]) as saved:
                np.testing.assert_array_equal(saved["training_row_indices"], [11, 21, 31, 41])
                np.testing.assert_array_equal(saved["outer_validation_row_indices"], [51, 61])
                np.testing.assert_array_equal(saved["labels"], [0, 1, 1, 0])
                np.testing.assert_array_equal(saved["probabilities"], [0.2, 0.8, 0.7, 0.1])
                for key, checksum in info["array_sha256"].items():
                    self.assertEqual(_array_sha256(saved[key]), checksum)

    def test_inner_oof_rejects_outer_leakage_and_incomplete_scores(self):
        with _inner_directory() as directory:
            for held_out, scores, ids in (([21], [0.2, 0.8], [1, 2]),
                                          ([31], [np.nan, 0.8], [1, 2]),
                                          ([31], [0.2, 0.8], [0, 2])):
                with self.subTest(held_out=held_out, scores=scores, ids=ids), self.assertRaises(ValueError):
                    _save_inner_oof_predictions(Path(directory), "fixture", 1, 23,
                        np.array([11, 21]), np.array(held_out), np.array(ids),
                        np.array([0, 1]), np.array(scores))

    def test_inner_persistence_does_not_change_thresholds_or_outer_predictions(self):
        rows = np.arange(10, 70)
        labels = np.tile([0, 1], 30)
        features = np.arange(80, dtype=np.float32).reshape(-1, 1)
        folds = [valid for _, valid in _stratified_folds(labels, 3, 19)]
        models = {"tiny": {"parameters": {"max_depth": 2, "min_samples_leaf": 2,
                  "n_bins": 8, "random_seed": 20}}}
        outer_probabilities = {("tiny", 3): np.linspace(0.1, 0.9, labels.size)}
        arguments = ({"tiny": [3]}, models, {"preprocessing": {}}, features,
                     rows, labels, ["A"], [{"name": "A", "semantic_type": "continuous"}],
                     folds, outer_probabilities, 2, 22)
        original, original_thresholds = _nested_evaluations(*arguments)
        with _inner_directory() as directory:
            persisted, persisted_thresholds = _nested_evaluations(*arguments,
                inner_oof_dir=Path(directory), experiment_prefix="fixture")
            np.testing.assert_array_equal(original_thresholds[("tiny", 3)], persisted_thresholds[("tiny", 3)])
            self.assertEqual(original[("tiny", 3)]["pooled_metrics"], persisted[("tiny", 3)]["pooled_metrics"])
            for before, after in zip(original[("tiny", 3)]["folds"], persisted[("tiny", 3)]["folds"]):
                for key in before:
                    if key != "inner_oof_predictions_artifact":
                        self.assertEqual(before[key], after[key])
                self.assertTrue(Path(after["inner_oof_predictions_artifact"]["path"]).exists())


if __name__ == "__main__":
    unittest.main()

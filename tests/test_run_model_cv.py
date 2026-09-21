"""Checks for fold-safe interaction features and staged optimization."""

import unittest
from pathlib import Path

import numpy as np

from implementations import logistic_regression
from src.evaluation import best_f1_threshold
from src.run_model_cv import (
    _add_interactions,
    _fit_checkpoints,
    _nested_threshold_evaluation,
    _nested_threshold_settings,
    _objective_gradient_norm,
    _save_oof_predictions,
    _threshold_diagnostics,
)


class TestModelCV(unittest.TestCase):
    def test_interaction_scale_uses_training_rows_only(self):
        train = np.array([[1, 1, 1], [1, 2, 3], [1, 3, 5]], dtype=np.float32)
        validation = np.array([[1, 100, 200]], dtype=np.float32)
        result_train, result_validation, names = _add_interactions(
            train, validation, ["A", "B"], [["A", "B"]]
        )
        products = train[:, 1] * train[:, 2]
        expected = (products - products.mean()) / products.std()
        np.testing.assert_allclose(result_train[:, -1], expected, rtol=1e-6)
        self.assertAlmostEqual(
            result_validation[0, -1],
            (20000 - products.mean()) / products.std(),
            places=3,
        )
        self.assertEqual(names[-1], "A__times__B")

    def test_warm_started_checkpoints_match_one_continuous_fit(self):
        design = np.array([[1, -1], [1, 0], [1, 1], [1, 2]], dtype=np.float32)
        labels = np.array([0, 0, 1, 1])
        checkpoints = list(
            _fit_checkpoints(
                labels, design, {"lambda": 0.0, "gamma": 0.1, "checkpoints": [3, 8]}
            )
        )
        expected, loss = logistic_regression(labels, design, np.zeros(2), 8, 0.1)
        np.testing.assert_allclose(checkpoints[-1][1], expected)
        self.assertAlmostEqual(checkpoints[-1][2], loss)

    def test_objective_gradient_norm_includes_l2_penalty(self):
        design = np.array([[1, -1], [1, 1]], dtype=np.float64)
        labels = np.array([0, 1], dtype=np.int8)
        weights = np.array([0.3, -0.4])
        probabilities = 1.0 / (1.0 + np.exp(-(design @ weights)))
        expected = np.linalg.norm(
            design.T @ (probabilities - labels) / labels.size + 2.0 * 0.25 * weights
        )
        self.assertAlmostEqual(
            _objective_gradient_norm(labels, design, weights, 0.25), expected
        )

    def test_threshold_diagnostics_selects_each_cross_fit_threshold_elsewhere(self):
        labels = np.array([1, 0, 1, 0, 1, 0], dtype=np.int8)
        probabilities = np.array([0.9, 0.2, 0.8, 0.1, 0.7, 0.3])
        folds = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]

        diagnostics = _threshold_diagnostics(labels, probabilities, folds)

        self.assertEqual(len(diagnostics["fold_optima"]), 3)
        self.assertEqual(len(diagnostics["cross_fit"]["folds"]), 3)
        self.assertGreaterEqual(
            diagnostics["fold_optimum_threshold_summary"]["maximum"],
            diagnostics["fold_optimum_threshold_summary"]["minimum"],
        )
        for detail in diagnostics["cross_fit"]["folds"]:
            fold = detail["evaluation_fold"] - 1
            mask = np.ones(labels.size, dtype=bool)
            mask[folds[fold]] = False
            expected = best_f1_threshold(labels[mask], probabilities[mask]).threshold
            self.assertAlmostEqual(detail["threshold"], expected)

    def test_nested_threshold_evaluation_keeps_each_outer_label_out_of_selection(self):
        features = np.arange(24, dtype=np.float32).reshape(12, 2)
        labels = np.array([0, 1] * 6, dtype=np.int8)
        outer_folds = [
            np.array([0, 1, 2, 3]),
            np.array([4, 5, 6, 7]),
            np.array([8, 9, 10, 11]),
        ]
        result = _nested_threshold_evaluation(
            {"preprocessing": {"name": "raw_median_scale", "exclude_features": []}},
            features,
            np.arange(labels.size),
            labels,
            ["first", "second"],
            [],
            {"lambda": 0.0, "gamma": 0.1, "checkpoints": [3]},
            3,
            [],
            outer_folds,
            np.where(labels == 1, 0.9, 0.1),
            2,
            123,
        )
        self.assertEqual(len(result["folds"]), 3)
        self.assertAlmostEqual(result["pooled_metrics"]["f1"], 1.0)
        for detail in result["folds"]:
            self.assertEqual(detail["outer_training_rows"], 8)
            self.assertEqual(detail["outer_validation_rows"], 4)
            self.assertEqual(len(detail["inner_training_gradient_norms"]), 2)

    def test_nested_threshold_settings_rejects_unusable_configuration(self):
        with self.assertRaises(ValueError):
            _nested_threshold_settings({"nested_threshold": {"checkpoints": [1000]}})

    def test_oof_predictions_preserve_row_fold_label_and_score_mapping(self):
        development_indices = np.array([11, 21, 31, 41])
        labels = np.array([0, 1, 0, 1], dtype=np.int8)
        probabilities = np.array([0.1, 0.8, 0.2, 0.9])
        folds = [np.array([0, 2]), np.array([1, 3])]
        artifact = None
        try:
            artifact, checksum = _save_oof_predictions(
                Path("results/experiments"),
                "threshold-artifact-test",
                development_indices,
                folds,
                labels,
                probabilities,
                [np.array([0.1, -0.2]), np.array([0.3, -0.4])],
                ["feature"],
            )
            self.assertTrue(artifact.exists())
            self.assertEqual(len(checksum), 64)
            with np.load(artifact) as saved:
                np.testing.assert_array_equal(
                    saved["development_indices"], development_indices
                )
                np.testing.assert_array_equal(saved["fold_ids"], np.array([1, 2, 1, 2]))
                np.testing.assert_array_equal(saved["labels"], labels)
                np.testing.assert_allclose(saved["probabilities"], probabilities)
                np.testing.assert_allclose(
                    saved["fold_weights"], np.array([[0.1, -0.2], [0.3, -0.4]])
                )
                np.testing.assert_array_equal(
                    saved["feature_names"], np.array(["intercept", "feature"])
                )
        finally:
            if artifact is not None:
                artifact.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

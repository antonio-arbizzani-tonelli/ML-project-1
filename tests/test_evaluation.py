"""Hand-checkable tests for binary-classification evaluation utilities."""

import unittest

import numpy as np

from src import evaluation


class TestEvaluation(unittest.TestCase):
    """Verify every metric from examples whose outcomes can be counted by hand."""

    def test_confusion_counts_and_derived_metrics(self):
        """Derive all metrics from TP=2, FP=1, TN=2, and FN=1."""

        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 1, 0, 0])
        counts, metrics = evaluation.classification_metrics(y_true, y_pred)
        self.assertEqual(
            counts,
            evaluation.ConfusionCounts(2, 1, 2, 1),
        )
        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)
        self.assertAlmostEqual(metrics["specificity"], 2 / 3)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 2 / 3)

    def test_all_negative_baseline_exposes_misleading_accuracy(self):
        """An all-negative model has 50% balanced accuracy with both classes."""

        y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        _, metrics = evaluation.classification_metrics(y_true, np.zeros(10))
        self.assertAlmostEqual(metrics["accuracy"], 0.8)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)

    def test_score_threshold_includes_equal_scores_as_positive(self):
        """Make the threshold boundary explicit and reproducible."""

        np.testing.assert_array_equal(
            evaluation.labels_from_scores([0.2, 0.5, 0.8], threshold=0.5),
            np.array([0, 1, 1], dtype=np.int8),
        )

    def test_precision_recall_curve_handles_ties_as_one_threshold(self):
        """At threshold 0.8, both tied scores are included together."""

        curve = evaluation.precision_recall_curve([1, 0, 1, 0], [0.9, 0.8, 0.8, 0.1])
        np.testing.assert_array_equal(curve.thresholds, [0.9, 0.8, 0.1])
        np.testing.assert_allclose(curve.precision, [1.0, 2 / 3, 0.5])
        np.testing.assert_allclose(curve.recall, [0.5, 1.0, 1.0])

    def test_average_precision_is_area_under_hand_checkable_curve(self):
        """The curve gains 0.5 recall at precision 1 then at precision 2/3."""

        self.assertAlmostEqual(
            evaluation.average_precision([1, 0, 1, 0], [0.9, 0.8, 0.8, 0.1]),
            0.5 + 0.5 * (2 / 3),
        )

    def test_best_f1_threshold_uses_accuracy_to_break_f1_tie(self):
        """Two feasible thresholds have F1=2/3; prefer the more accurate one."""

        choice = evaluation.best_f1_threshold(
            [1, 1, 0, 0, 0, 0], [0.9, 0.8, 0.8, 0.8, 0.1, 0.1]
        )
        self.assertEqual(choice.threshold, 0.9)
        self.assertAlmostEqual(choice.f1, 2 / 3)
        self.assertAlmostEqual(choice.accuracy, 5 / 6)

    def test_binary_log_loss_has_known_value_and_handles_extremes(self):
        """Perfect probabilities yield finite loss after safe clipping."""

        self.assertAlmostEqual(
            evaluation.binary_log_loss([1, 0], [0.8, 0.2]), -np.log(0.8)
        )
        self.assertTrue(np.isfinite(evaluation.binary_log_loss([1, 0], [1.0, 0.0])))

    def test_invalid_label_or_score_inputs_raise_clear_errors(self):
        """Reject incompatible encodings, lengths, and non-finite scores."""

        with self.assertRaises(ValueError):
            evaluation.confusion_counts([1, -1], [1, 0])
        with self.assertRaises(ValueError):
            evaluation.labels_from_scores([0.1, np.nan])
        with self.assertRaises(ValueError):
            evaluation.precision_recall_curve([1, 0], [0.4])


if __name__ == "__main__":
    unittest.main()

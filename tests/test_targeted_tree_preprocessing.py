"""Check fold-safe nominal encoding and selective exact binning."""

import unittest
from pathlib import Path

import numpy as np

from src.feature_preprocessing import tree_feature_matrices
from src.numpy_boosting import HistogramBinner, HistogramGradientBoostingClassifier
from src.run_boosting_cv import _resolved_parameters
from tools.targeted_preprocessing_analysis import _partitions


class TestTargetedTreePreprocessing(unittest.TestCase):
    def test_preflight_covers_all_nine_training_partitions(self):
        y = np.tile([0, 1], 60)
        development = np.arange(y.size)
        parts = list(_partitions(y, development, {"seed": 19,
            "nested_threshold": {"inner_seed": 22}}))
        self.assertEqual(len(parts), 9)
        for _, train, valid in parts:
            self.assertEqual(np.intersect1d(train, valid).size, 0)
        for fold in range(3):
            outer_train = parts[fold][1]
            for inner in parts[3 + fold * 2:5 + fold * 2]:
                np.testing.assert_array_equal(np.union1d(inner[1], inner[2]), outer_train)

    def test_selective_binning_leaves_other_columns_unchanged(self):
        x = np.column_stack((np.r_[np.zeros(99), 1], np.r_[np.zeros(99), 1]))
        original = HistogramBinner().fit(x)
        targeted = HistogramBinner(64, "exact_selected", [0]).fit(x)
        np.testing.assert_array_equal(targeted.exact_columns_, [True, False])
        self.assertEqual(np.unique(targeted.transform(x)[:, 0]).size, 2)
        np.testing.assert_array_equal(original.edges_[1], targeted.edges_[1])
        np.testing.assert_array_equal(original.transform(x)[:, 1], targeted.transform(x)[:, 1])

    def test_invalid_selectors_fail(self):
        for indices in (None, [], [-1], [0, 0], [True]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                HistogramBinner(64, "exact_selected", indices)
        with self.assertRaises(ValueError):
            HistogramBinner(64, "quantile", [0])
        with self.assertRaises(ValueError):
            HistogramBinner(64, "exact_selected", [1]).fit(np.zeros((3, 1)))
        with self.assertRaises(ValueError):
            HistogramBinner(2, "exact_selected", [0]).fit(np.arange(3).reshape(-1, 1))

    def test_activity_vocabulary_uses_training_and_keeps_88_and_missing(self):
        metadata = [{"name": n, "semantic_type": "categorical",
            "special_values": [], "documented_values": []}
            for n in ("EXRACT11", "EXRACT21", "OTHER")]
        plan = {"tree_one_hot_features": ["EXRACT11", "EXRACT21"],
            "missing_code_overrides": {n: [{"code": 77, "reason": "unknown"},
                {"code": 99, "reason": "refused"}] for n in ("EXRACT11", "EXRACT21")}}
        train = np.array([[1, 88, 5], [2, 1, 6], [77, 99, 7]], dtype=np.float32)
        valid = np.array([[3, 88, 8], [99, 77, 9], [1, 1, 10]], dtype=np.float32)
        t, v, names = tree_feature_matrices(train, valid,
            ["EXRACT11", "EXRACT21", "OTHER"], metadata, plan)
        pos = {n: i for i, n in enumerate(names)}
        self.assertNotIn("EXRACT11", names)
        self.assertNotIn("EXRACT11__is_3", names)
        self.assertEqual(v[0, pos["EXRACT11__unknown"]], 1)
        self.assertEqual(v[0, pos["EXRACT21__is_88"]], 1)
        self.assertEqual(v[2, pos["EXRACT11__is_1"]], 1)
        for n, i in pos.items():
            if n.startswith(("EXRACT11__", "EXRACT21__")):
                self.assertTrue(np.isnan(t[2, i]))
                self.assertTrue(np.isnan(v[1, i]))
        np.testing.assert_array_equal(t[:, pos["OTHER"]], train[:, 2])
        np.testing.assert_array_equal(v[:, pos["OTHER"]], valid[:, 2])
        params = _resolved_parameters({"binning_strategy": "exact_selected",
            "exact_feature_prefixes": ["EXRACT11__", "EXRACT21__"]}, names)
        binner = HistogramBinner(64, **params).fit(t)
        self.assertEqual(int(binner.exact_columns_.sum()), len(names) - 1)

    def test_unknown_feature_selection_is_rejected(self):
        for config in ({"exact_feature_names": ["ABSENT"]},
                       {"exact_feature_prefixes": ["ABSENT__"]}):
            with self.assertRaises(ValueError):
                _resolved_parameters(config, ["A"])

    def test_selected_checkpoint_continues_exactly(self):
        x = np.array([[0, 2], [1, 3], [np.nan, 4], [0, 5]], dtype=np.float32)
        y = np.array([0, 1, 1, 0])
        options = dict(max_depth=2, min_samples_leaf=1, random_seed=20,
            binning_strategy="exact_selected", exact_feature_indices=[0])
        path = Path("results/test_targeted_checkpoint.pkl")
        try:
            full = HistogramGradientBoostingClassifier(n_estimators=4, **options).fit(x, y)
            partial = HistogramGradientBoostingClassifier(n_estimators=2, **options).fit(x, y)
            partial.save_checkpoint(path)
            restored, _ = HistogramGradientBoostingClassifier.load_checkpoint(path)
            restored.continue_fit(x, y, 2)
            np.testing.assert_array_equal(full.decision_function(x), restored.decision_function(x))
        finally:
            path.unlink(missing_ok=True)

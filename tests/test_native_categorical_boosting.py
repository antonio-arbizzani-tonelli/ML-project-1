"""Native categorical split semantics and exact disabled Phase 14 parity."""

import tempfile
from pathlib import Path
import unittest

import numpy as np

from src import numpy_boosting_categorical as native
from src import numpy_boosting_phase14 as historical


def categorical_tree(features, gradients, hessians=None, *, leaf=1, gain=0.0, l2=1.0):
    features = np.asarray(features, dtype=np.uint8).reshape(-1, 1)
    if hessians is None:
        hessians = np.ones(features.shape[0])
    return native._NewtonTree(
        1, leaf, l2, gain, None, np.random.default_rng(17), "recursive",
        categorical_features=np.array([True]),
    ).fit(features, np.array([254]), np.asarray(gradients), np.asarray(hessians))


class TestNativeCategoricalBinner(unittest.TestCase):
    def test_exact_categories_are_not_limited_by_numeric_bin_count(self):
        rng = np.random.default_rng(8)
        features = np.column_stack((np.arange(100), rng.normal(size=100))).astype(np.float32)
        features[7, 1] = np.nan
        actual = native.HistogramBinner(64, categorical_feature_indices=[0]).fit(features)
        reference = historical.HistogramBinner(64).fit(features)
        encoded = actual.transform(features)
        np.testing.assert_array_equal(encoded[:, 0], np.arange(100))
        np.testing.assert_array_equal(encoded[:, 1], reference.transform(features)[:, 1])
        np.testing.assert_array_equal(actual.edges_[1], reference.edges_[1])
        self.assertEqual(actual.bin_counts_[0], 100)
        self.assertEqual(actual.categorical_columns_.tolist(), [True, False])
        self.assertEqual(actual.category_values_[0].tolist(), list(range(100)))
        self.assertIsNone(actual.category_values_[1])

    def test_training_vocabulary_is_exact_and_unseen_values_are_missing(self):
        binner = native.HistogramBinner(2, categorical_feature_indices=(0,)).fit(
            np.array([[3.0], [8.0], [20.0], [np.nan]])
        )
        np.testing.assert_array_equal(
            binner.transform(np.array([[8], [20], [3], [np.nan], [2], [4], [99]]))[:, 0],
            [1, 2, 0, 255, 255, 255, 255],
        )
        # The same raw code can have a different identity in a different fold.
        another = native.HistogramBinner(2, categorical_feature_indices=(0,)).fit(
            np.array([[8.0], [100.0]])
        )
        self.assertEqual(another.transform(np.array([[8.0]]))[0, 0], 0)
        np.testing.assert_array_equal(binner.category_values_[0], [3, 8, 20])

    def test_all_missing_and_constant_training_columns(self):
        empty = native.HistogramBinner(categorical_feature_indices=[0]).fit(np.full((5, 1), np.nan))
        np.testing.assert_array_equal(empty.transform([[np.nan], [3], [-20]])[:, 0], [255, 255, 255])
        self.assertEqual(empty.category_values_[0].size, 0)
        fixed = native.HistogramBinner(categorical_feature_indices=[0]).fit([[7], [7]])
        np.testing.assert_array_equal(fixed.transform([[7], [8], [np.nan]])[:, 0], [0, 255, 255])

    def test_level_limit_and_invalid_indices_are_rejected(self):
        valid = native.HistogramBinner(categorical_feature_indices=[0]).fit(np.arange(254).reshape(-1, 1))
        self.assertEqual(valid.bin_counts_[0], 254)
        self.assertEqual(valid.transform([[253]])[0, 0], 253)
        with self.assertRaisesRegex(ValueError, "254"):
            native.HistogramBinner(categorical_feature_indices=[0]).fit(np.arange(255).reshape(-1, 1))
        for indices in ([0, 0], [-1], [True], [0.5], None, 1):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                native.HistogramBinner(categorical_feature_indices=indices)
        with self.assertRaisesRegex(ValueError, "width"):
            native.HistogramBinner(categorical_feature_indices=[2]).fit(np.ones((4, 2)))
        with self.assertRaisesRegex(ValueError, "both"):
            native.HistogramBinner(8, "exact_selected", [0], [0])
        with self.assertRaisesRegex(ValueError, "recursive"):
            native.HistogramGradientBoostingClassifier(
                histogram_strategy="levelwise", categorical_feature_indices=[0]
            )


class TestNativeCategoricalSplits(unittest.TestCase):
    def test_noncontiguous_codes_are_grouped_by_membership(self):
        raw = np.repeat([10, 20, 30, 40], 8).reshape(-1, 1)
        labels = np.repeat([0, 1, 0, 1], 8)
        parameters = dict(n_estimators=1, learning_rate=1, max_depth=1,
                          min_samples_leaf=2, n_bins=64, random_seed=7)
        categorical = native.HistogramGradientBoostingClassifier(
            **parameters, categorical_feature_indices=[0]
        ).fit(raw, labels)
        numeric = historical.HistogramGradientBoostingClassifier(**parameters).fit(raw, labels)
        np.testing.assert_array_equal(categorical.predict(raw), labels)
        self.assertLess(np.mean(numeric.predict(raw) == labels), 1)
        node = categorical.trees_[0].root_
        self.assertIsNone(node.split_bin)
        self.assertEqual(set(node.left_categories), {1, 3})
        self.assertEqual(categorical.categorical_feature_indices, [0])

    def test_both_missing_directions_are_evaluated(self):
        bins = np.repeat([0, 1, 255], 4)
        right_missing = categorical_tree(bins, np.repeat([-2, 2, 2], 4)).root_
        left_missing = categorical_tree(bins, np.repeat([-2, 2, -2], 4)).root_
        self.assertEqual(right_missing.left_categories, (0,))
        self.assertFalse(right_missing.missing_go_left)
        self.assertEqual(left_missing.left_categories, (0,))
        self.assertTrue(left_missing.missing_go_left)

    def test_observed_versus_missing_and_all_missing_nodes(self):
        bins = np.repeat([0, 1, 255], 5)
        tree = categorical_tree(bins, np.repeat([2, 2, -2], 5))
        self.assertEqual(set(tree.root_.left_categories), {0, 1})
        self.assertFalse(tree.root_.missing_go_left)
        single = categorical_tree(np.repeat([0, 255], 5), np.repeat([2, -2], 5))
        self.assertEqual(single.root_.left_categories, (0,))
        empty = categorical_tree([255] * 10, np.arange(10) - 4.5)
        self.assertIsNone(empty.root_.feature)
        np.testing.assert_array_equal(empty.predict(np.array([[255], [0]])), [0, 0])

    def test_unknown_categories_follow_the_learned_missing_route(self):
        features = np.repeat([1.0, 2.0, np.nan], 8).reshape(-1, 1)
        labels = np.repeat([0, 1, 1], 8)
        model = native.HistogramGradientBoostingClassifier(
            n_estimators=3, max_depth=2, min_samples_leaf=2,
            categorical_feature_indices=[0], random_seed=4,
        ).fit(features, labels)
        scores = model.predict_proba([[np.nan], [99.0], [-3.0]])
        np.testing.assert_array_equal(scores, np.repeat(scores[:1], 3, axis=0))

    def test_gain_and_leaf_sizes_match_explicit_ordered_prefix_search(self):
        bins = np.repeat([0, 3, 7, 255], [3, 4, 2, 3])
        gradients = np.array([-0.3, -0.4, -0.2, 0.1, 0.2, 0.3, 0.2, -0.1, -0.2, 0.2, 0.4, 0.5])
        hessians = np.linspace(0.15, 0.4, len(bins))
        l2 = 1.0
        tree = categorical_tree(bins, gradients, hessians, leaf=3, l2=l2)
        groups = [0, 3, 7]
        ordered = sorted(groups, key=lambda code: gradients[bins == code].sum() / hessians[bins == code].sum())
        score = lambda mask: gradients[mask].sum() ** 2 / (hessians[mask].sum() + l2)
        parent = gradients.sum() ** 2 / (hessians.sum() + l2)
        best = None
        for prefix in range(1, len(ordered) + 1):
            for missing_left in (False, True):
                left = np.isin(bins, ordered[:prefix]) | ((bins == 255) & missing_left)
                if min(left.sum(), (~left).sum()) < 3:
                    continue
                gain = 0.5 * (score(left) + score(~left) - parent)
                if best is None or gain > best[0]:
                    best = gain, ordered[:prefix], missing_left, left
        self.assertGreater(best[0], 0)
        self.assertEqual(tree.root_.left_categories, tuple(best[1]))
        self.assertEqual(tree.root_.missing_go_left, best[2])
        for node, rows in ((tree.root_.left, best[3]), (tree.root_.right, ~best[3])):
            self.assertAlmostEqual(node.value, -gradients[rows].sum() / (hessians[rows].sum() + l2))
        stopped = categorical_tree(bins, gradients, hessians, leaf=3, gain=best[0] + 1, l2=l2)
        self.assertIsNone(stopped.root_.feature)
        too_small = categorical_tree([0, 0, 1, 1, 1], [-1, -1, 1, 1, 1], leaf=3)
        self.assertIsNone(too_small.root_.feature)

    def test_recoding_untied_categories_preserves_predictions_and_ties_are_deterministic(self):
        features = np.repeat([10, 20, 30, 40], 12).reshape(-1, 1)
        labels = np.concatenate([np.r_[np.ones(n), np.zeros(12 - n)] for n in (0, 12, 2, 9)])
        recoded = np.repeat([900, -7, 22, 3], 12).reshape(-1, 1)
        parameters = dict(n_estimators=8, max_depth=2, min_samples_leaf=2,
                          categorical_feature_indices=[0], random_seed=13)
        first = native.HistogramGradientBoostingClassifier(**parameters).fit(features, labels)
        second = native.HistogramGradientBoostingClassifier(**parameters).fit(recoded, labels)
        np.testing.assert_array_equal(first.predict_proba(features), second.predict_proba(recoded))
        # Stable ties are deterministic on the same vocabulary. Arbitrarily
        # renamed categories with identical G/H are not claimed to be unique.
        tied = np.repeat([0, 1, 2], 4)
        gradients = np.repeat([-1, -1, 2], 4)
        one = categorical_tree(tied, gradients)
        two = categorical_tree(tied, gradients)
        self.assertEqual(one.root_, two.root_)


class TestCategoricalContinuationAndNumericParity(unittest.TestCase):
    def test_disabled_categorical_option_is_exactly_the_historical_model(self):
        rng = np.random.default_rng(91)
        features = rng.normal(size=(113, 6)).astype(np.float32)
        features[::5, 2] = np.nan
        features[:, 0] = np.arange(len(features)) % 4
        labels = (rng.random(len(features)) < 0.4).astype(np.int8)
        probe = np.r_[features[::3], np.full((2, features.shape[1]), np.nan)]
        for strategy in ("recursive", "levelwise"):
            for binning in ("quantile", "exact_low_cardinality", "exact_selected"):
                for weight in (1.0, 2.0):
                    with self.subTest(strategy=strategy, binning=binning, weight=weight):
                        parameters = dict(n_estimators=5, max_depth=3, min_samples_leaf=3,
                                          n_bins=8, max_features=3, random_seed=20260920,
                                          histogram_strategy=strategy, binning_strategy=binning,
                                          exact_feature_indices=[0] if binning == "exact_selected" else None,
                                          positive_class_weight=weight)
                        reference = historical.HistogramGradientBoostingClassifier(**parameters).fit(features, labels)
                        actual = native.HistogramGradientBoostingClassifier(**parameters).fit(features, labels)
                        np.testing.assert_array_equal(actual.training_scores_, reference.training_scores_)
                        np.testing.assert_array_equal(actual.predict_proba(probe), reference.predict_proba(probe))
                        np.testing.assert_array_equal(actual.binner_.transform(probe), reference.binner_.transform(probe))
                        self.assertEqual(actual.random_generator_state_, reference.random_generator_state_)

    def test_checkpoint_resume_and_staged_predictions_are_exact(self):
        rng = np.random.default_rng(12)
        features = rng.normal(size=(150, 5)).astype(np.float32)
        features[:, 1] = np.arange(150) % 5
        features[::7, 1] = np.nan
        labels = ((features[:, 1] == 1) | (features[:, 0] > 0.7)).astype(np.int8)
        parameters = dict(n_estimators=50, max_depth=3, min_samples_leaf=3, max_features=3,
                          n_bins=8, categorical_feature_indices=[1], random_seed=20260920)
        full = native.HistogramGradientBoostingClassifier(**parameters).fit(features, labels)
        partial = native.HistogramGradientBoostingClassifier(**dict(parameters, n_estimators=25)).fit(features, labels)
        local = Path(__file__).resolve().parents[1] / ".local"
        local.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as temporary:
            path = Path(temporary) / "categorical.pkl"
            partial.save_checkpoint(path, {"categorical_feature_indices": [1]})
            restored, metadata = native.HistogramGradientBoostingClassifier.load_checkpoint(path)
            self.assertEqual(metadata, {"categorical_feature_indices": [1]})
            self.assertEqual(restored.categorical_feature_indices, [1])
            restored.continue_fit(features, labels, 25)
        np.testing.assert_array_equal(restored.training_scores_, full.training_scores_)
        np.testing.assert_array_equal(restored.predict_proba(features), full.predict_proba(features))
        np.testing.assert_array_equal(restored.binner_.category_values_[1], full.binner_.category_values_[1])
        staged = dict(full.staged_predict_proba(features, [25, 50]))
        np.testing.assert_array_equal(staged[25], partial.predict_proba(features)[:, 1])
        np.testing.assert_array_equal(staged[50], full.predict_proba(features)[:, 1])


if __name__ == "__main__":
    unittest.main()

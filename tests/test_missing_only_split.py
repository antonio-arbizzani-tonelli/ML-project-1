"""Regression checks for the final observed-versus-missing split candidate."""

import unittest

import numpy as np

from src.numpy_boosting import HistogramGradientBoostingClassifier


class TestMissingOnlySplit(unittest.TestCase):
    def _fit(self, features, labels, strategy, **settings):
        return HistogramGradientBoostingClassifier(
            n_estimators=1, histogram_strategy=strategy, **settings
        ).fit(features, labels)

    def _assert_pure_split(self, model, features, finite_count):
        root = model.trees_[0].root_
        self.assertEqual(root.feature, 0)
        self.assertEqual(root.split_bin, int(model.binner_.bin_counts_[0]) - 1)
        self.assertFalse(root.missing_go_left)
        self.assertIsNone(root.left.feature)
        self.assertIsNone(root.right.feature)
        scores = model.decision_function(features)
        np.testing.assert_array_equal(scores[:finite_count],
                                      np.repeat(scores[0], finite_count))
        np.testing.assert_array_equal(scores[finite_count:],
                                      np.repeat(scores[-1], len(scores) - finite_count))
        self.assertGreater(scores[-1], scores[0])
        np.testing.assert_array_equal(model.training_scores_, scores)

    def test_quantile_missing_only_split_with_every_finite_bin_occupied(self):
        features = np.r_[np.arange(40), np.repeat(np.nan, 20)].reshape(-1, 1)
        labels = np.r_[np.zeros(40), np.ones(20)]
        for strategy in ("recursive", "levelwise"):
            with self.subTest(strategy=strategy):
                model = self._fit(features, labels, strategy, n_bins=4,
                                  max_depth=1, min_samples_leaf=5)
                bins = model.binner_.transform(features)
                np.testing.assert_array_equal(np.unique(bins[:40]), np.arange(4))
                self._assert_pure_split(model, features, 40)
                probability = labels.mean()
                hessian = probability * (1 - probability)
                self.assertAlmostEqual(model.trees_[0].root_.left.value,
                                       -(40 * probability) / (40 * hessian + 1))
                self.assertAlmostEqual(model.trees_[0].root_.right.value,
                                       -(20 * (probability - 1)) / (20 * hessian + 1))

    def test_active_64_bin_depth_and_leaf_settings_allow_a_pure_split(self):
        features = np.r_[np.arange(1024), np.repeat(np.nan, 500)].reshape(-1, 1)
        labels = np.r_[np.zeros(1024), np.ones(500)]
        for strategy in ("recursive", "levelwise"):
            with self.subTest(strategy=strategy):
                model = self._fit(features, labels, strategy, n_bins=64,
                                  max_depth=5, min_samples_leaf=200)
                bins = model.binner_.transform(features)
                np.testing.assert_array_equal(np.unique(bins[:1024]), np.arange(64))
                self._assert_pure_split(model, features, 1024)

    def test_missing_only_candidate_still_obeys_minimum_leaf_size(self):
        for strategy in ("recursive", "levelwise"):
            with self.subTest(strategy=strategy):
                # Four missing rows cannot form a child with a five-row minimum.
                features = np.r_[np.arange(40), np.repeat(np.nan, 4)].reshape(-1, 1)
                labels = np.r_[np.zeros(40), np.ones(4)]
                model = self._fit(features, labels, strategy, n_bins=4,
                                  max_depth=1, min_samples_leaf=5)
                root = model.trees_[0].root_
                self.assertIsNotNone(root.feature)
                self.assertLess(root.split_bin, 3)
                bins = model.binner_.transform(features)[:, root.feature]
                left = np.where(bins == 255, root.missing_go_left,
                                bins <= root.split_bin)
                self.assertGreaterEqual(int(left.sum()), 5)
                self.assertGreaterEqual(int((~left).sum()), 5)

                # One observed row also cannot form a child, so this node stops.
                features = np.r_[1.0, np.repeat(np.nan, 9)].reshape(-1, 1)
                labels = np.r_[0, np.ones(9)]
                model = self._fit(features, labels, strategy, n_bins=4,
                                  max_depth=1, min_samples_leaf=2)
                self.assertIsNone(model.trees_[0].root_.feature)

    def test_missing_only_candidate_still_obeys_minimum_gain(self):
        features = np.r_[np.arange(40), np.repeat(np.nan, 20)].reshape(-1, 1)
        labels = np.r_[np.zeros(40), np.ones(20)]
        probability = labels.mean()
        gradients = probability - labels
        hessians = np.full(len(labels), probability * (1 - probability))
        gain = 0.5 * (
            gradients[:40].sum() ** 2 / (hessians[:40].sum() + 1)
            + gradients[40:].sum() ** 2 / (hessians[40:].sum() + 1)
            - gradients.sum() ** 2 / (hessians.sum() + 1)
        )
        for strategy in ("recursive", "levelwise"):
            with self.subTest(strategy=strategy):
                model = self._fit(features, labels, strategy, n_bins=4,
                                  max_depth=1, min_samples_leaf=5, min_gain=gain + 1)
                self.assertIsNone(model.trees_[0].root_.feature)

    def test_existing_leading_empty_bucket_keeps_the_first_tied_candidate(self):
        features = np.r_[np.zeros(6), np.repeat(np.nan, 6)].reshape(-1, 1)
        labels = np.r_[np.zeros(6), np.ones(6)]
        for strategy in ("recursive", "levelwise"):
            with self.subTest(strategy=strategy):
                model = self._fit(features, labels, strategy, n_bins=4,
                                  max_depth=1, min_samples_leaf=2)
                self.assertEqual(model.trees_[0].root_.split_bin, 0)
                self.assertTrue(model.trees_[0].root_.missing_go_left)

    def test_all_finite_predictions_match_the_before_fix_reference(self):
        # Reference scores from the frozen implementation before this correction.
        features = np.random.default_rng(433).normal(size=(16, 3)).astype(np.float32)
        labels = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1])
        expected = {
            "recursive": [-0.06225051480366607, -0.02034407293736273,
                0.11839729413817743, -0.13676667968181902, -0.13676667968181902,
                0.08350027679762122, -0.03897911969955998, 0.11839729413817743,
                0.11839729413817743, -0.05996322047495945, 0.11839729413817743,
                -0.09429560006186782, 0.11839729413817743, 0.06308081377586958,
                0.11839729413817743, -0.03897911969955998],
            "levelwise": [-0.12455828317786181, -0.031098254271098958,
                0.12689105116118177, -0.16400566095464195, -0.16400566095464195,
                0.12689105116118177, -0.0705456320478791, 0.12689105116118177,
                0.12689105116118177, -0.0705456320478791, 0.12689105116118177,
                -0.0705456320478791, 0.12689105116118177, 0.12689105116118177,
                0.12689105116118177, -0.0705456320478791],
        }
        for strategy, scores in expected.items():
            with self.subTest(strategy=strategy):
                model = HistogramGradientBoostingClassifier(n_estimators=4,
                    max_depth=2, min_samples_leaf=2, n_bins=4, max_features=2,
                    random_seed=29, histogram_strategy=strategy).fit(features, labels)
                np.testing.assert_allclose(model.decision_function(features), scores,
                                           rtol=1e-13, atol=1e-14)

    def test_exact_binning_retains_its_existing_final_boundary(self):
        features = np.array([[0], [0], [1], [1], [np.nan], [np.nan]])
        labels = np.array([0, 0, 0, 0, 1, 1])
        for strategy in ("recursive", "levelwise"):
            with self.subTest(strategy=strategy):
                model = self._fit(features, labels, strategy,
                    binning_strategy="exact_low_cardinality", max_depth=1,
                    min_samples_leaf=2)
                self._assert_pure_split(model, features, 4)

    def test_all_missing_column_stops_without_an_empty_child(self):
        features = np.full((12, 1), np.nan)
        labels = np.tile([0, 1], 6)
        for strategy in ("recursive", "levelwise"):
            with self.subTest(strategy=strategy):
                model = self._fit(features, labels, strategy, max_depth=5,
                                  min_samples_leaf=2)
                self.assertIsNone(model.trees_[0].root_.feature)


if __name__ == "__main__":
    unittest.main()

"""Structural trials: exact disabled parity, sampling and global leaf growth."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from src import numpy_boosting_phase14 as historical
from src import numpy_boosting_structural as structural


def make_tree(*, depth=5, leaf=2, leaves=8, seed=19, l2=1.0, gain=0.0, max_features=None):
    return structural._NewtonTree(
        depth, leaf, l2, gain, max_features, np.random.default_rng(seed), "recursive",
        growth_policy="leafwise", max_leaves=leaves,
    )


def assert_tree_rows(test, tree, binned, *, min_leaf, depth_limit, leaf_limit):
    leaves = []

    def visit(node, rows, depth):
        test.assertEqual(node.row_count, len(rows))
        test.assertLessEqual(depth, depth_limit)
        if node.feature is None:
            leaves.append((rows, depth))
            test.assertGreaterEqual(len(rows), min_leaf)
            return
        values = binned[rows, node.feature]
        left = values <= node.split_bin
        left[values == 255] = node.missing_go_left
        visit(node.left, rows[left], depth + 1)
        visit(node.right, rows[~left], depth + 1)

    visit(tree.root_, np.arange(len(binned)), 0)
    test.assertLessEqual(len(leaves), leaf_limit)
    return leaves


class TestStratifiedTreeSampling(unittest.TestCase):
    def test_samples_are_ordered_unique_proportional_and_keep_rare_classes(self):
        labels = np.r_[np.zeros(83), np.ones(17)]
        rng = np.random.default_rng(12)
        rows = structural._stratified_sample_rows(labels, .8, rng)
        self.assertEqual(len(rows), 80)
        self.assertTrue(np.all(np.diff(rows) > 0))
        self.assertGreaterEqual(rows.min(), 0)
        self.assertLess(rows.max(), len(labels))
        self.assertLessEqual(abs(np.sum(labels[rows]) - 80 * .17), 1)
        rare = np.r_[np.zeros(99), 1]
        selected = structural._stratified_sample_rows(rare, .1, rng)
        self.assertEqual(len(selected), 10)
        self.assertEqual(np.sum(rare[selected]), 1)
        for labels in (np.zeros(11), np.ones(11)):
            selected = structural._stratified_sample_rows(labels, .8, rng)
            self.assertEqual(len(selected), 8)
        with self.assertRaisesRegex(ValueError, "every observed class"):
            structural._stratified_sample_rows(np.array([0, 1]), .4, rng)

    def test_full_sample_consumes_no_rng(self):
        rng = np.random.default_rng(33)
        before = copy.deepcopy(rng.bit_generator.state)
        np.testing.assert_array_equal(structural._stratified_sample_rows([0, 1, 0], 1., rng), [0, 1, 2])
        self.assertEqual(before, rng.bit_generator.state)

    def test_each_tree_samples_train_rows_but_bins_and_updates_use_all_rows(self):
        rng = np.random.default_rng(63)
        features = rng.normal(size=(123, 6)).astype(np.float32)
        features[::13, 3] = np.nan
        labels = ((features[:, 0] > .2) | (features[:, 2] < -.6)).astype(np.int8)
        rows_per_tree, fit_inputs = [], []
        original_sample, original_fit = structural._stratified_sample_rows, structural._NewtonTree.fit

        def capture_sample(*args):
            rows = original_sample(*args)
            rows_per_tree.append(rows.copy())
            return rows

        def capture_fit(tree, values, bins, gradients, hessians):
            fit_inputs.append((values.copy(), gradients.copy(), hessians.copy()))
            return original_fit(tree, values, bins, gradients, hessians)

        parameters = dict(n_estimators=6, max_depth=3, min_samples_leaf=3,
                          n_bins=8, max_features=3, random_seed=20260920, subsample=.8)
        with mock.patch.object(structural, "_stratified_sample_rows", side_effect=capture_sample), \
                mock.patch.object(structural._NewtonTree, "fit", new=capture_fit):
            model = structural.HistogramGradientBoostingClassifier(**parameters).fit(features, labels)
        reference_binner = historical.HistogramBinner(8).fit(features)
        binned = model.binner_.transform(features)
        np.testing.assert_array_equal(binned, reference_binner.transform(features))
        self.assertEqual(len({tuple(rows) for rows in rows_per_tree}), 6)
        scores = np.full(len(labels), model.base_score_)
        for index, (tree, rows, inputs) in enumerate(zip(model.trees_, rows_per_tree, fit_inputs)):
            values, gradients, hessians = inputs
            probabilities = structural._sigmoid(scores)
            np.testing.assert_array_equal(values, binned[rows])
            np.testing.assert_array_equal(gradients, (probabilities - labels)[rows])
            np.testing.assert_array_equal(hessians, np.maximum(probabilities * (1 - probabilities), 1e-12)[rows])
            self.assertEqual(tree.sample_row_count_, int(np.floor(.8 * len(labels))))
            self.assertEqual(tree.training_row_count_, len(labels))
            self.assertEqual(tree.sample_class_counts_, tuple(int(np.sum(labels[rows] == v)) for v in (0, 1)))
            self.assertEqual(tree.sample_row_indices_sha256_, structural._row_indices_sha256(rows))
            self.assertTrue(all(tree.sample_class_counts_))
            self.assertFalse(hasattr(tree, "sample_indices_"))
            scores += model.learning_rate * tree.predict(binned)
            if index == 0:
                outside = np.ones(len(labels), dtype=bool)
                outside[rows] = False
                self.assertTrue(np.any(scores[outside] != model.base_score_))
        np.testing.assert_array_equal(scores, model.training_scores_)
        np.testing.assert_array_equal(scores, model.decision_function(features))
        repeat = structural.HistogramGradientBoostingClassifier(**parameters).fit(features, labels)
        self.assertEqual([t.sample_row_indices_sha256_ for t in model.trees_],
                         [t.sample_row_indices_sha256_ for t in repeat.trees_])
        np.testing.assert_array_equal(model.predict_proba(features), repeat.predict_proba(features))
        before = copy.deepcopy(model.sampling_generator_state_)
        model.predict_proba(np.full((10, features.shape[1]), 999.))
        self.assertEqual(model.sampling_generator_state_, before)
        for column in range(features.shape[1]):
            np.testing.assert_array_equal(model.binner_.edges_[column], reference_binner.edges_[column])

    def test_sampling_draws_do_not_advance_feature_rng(self):
        rng = np.random.default_rng(4)
        features = rng.normal(size=(90, 7)).astype(np.float32)
        labels = (features[:, 1] > .2).astype(int)
        model = structural.HistogramGradientBoostingClassifier(
            n_estimators=1, max_depth=3, min_samples_leaf=3,
            max_features=3, n_bins=8, random_seed=61, subsample=.8,
        ).fit(features, labels)
        sampling_rng = np.random.default_rng(np.random.SeedSequence([61, structural._SAMPLING_SEED_SALT]))
        rows = structural._stratified_sample_rows(labels, .8, sampling_rng)
        feature_rng = np.random.default_rng(61)
        probabilities = structural._sigmoid(np.full(len(labels), model.base_score_))
        binned = model.binner_.transform(features)
        direct = structural._NewtonTree(3, 3, 1., 0., 3, feature_rng, "recursive").fit(
            binned[rows], model.binner_.bin_counts_, (probabilities-labels)[rows],
            np.maximum(probabilities*(1-probabilities), 1e-12)[rows],
        )
        self.assertEqual(model.random_generator_state_, feature_rng.bit_generator.state)
        self.assertEqual(model.sampling_generator_state_, sampling_rng.bit_generator.state)
        np.testing.assert_array_equal(model.trees_[0].predict(binned), direct.predict(binned))


class TestGlobalLeafwiseGrowth(unittest.TestCase):
    def test_highest_gain_leaf_wins_instead_of_depth_first_left_child(self):
        binned = np.repeat([[0, 0], [0, 1], [1, 0], [1, 1]], 10, axis=0).astype(np.uint8)
        gradients = np.repeat([-6., -2., 0., 8.], 10)
        tree = make_tree(leaves=3).fit(binned, [2, 2], gradients, np.ones(40))
        self.assertEqual(tree.root_.feature, 0)
        self.assertEqual([item["node_id"] for item in tree.split_history_], [0, 2])
        self.assertIsNone(tree.root_.left.feature)
        self.assertEqual(tree.root_.right.feature, 1)
        self.assertEqual(tree.leaf_count_, 3)
        assert_tree_rows(self, tree, binned, min_leaf=2, depth_limit=5, leaf_limit=3)

    def test_cached_leaf_candidates_are_not_recomputed_and_ties_are_stable(self):
        binned = np.repeat([[0, 0], [0, 1], [1, 0], [1, 1]], 10, axis=0).astype(np.uint8)
        gradients = np.repeat([-6., -2., 2., 6.], 10)
        rows_seen = []
        original = structural._NewtonTree._leafwise_candidate

        def capture(tree, values, bins, gradients, hessians, rows, depth):
            rows_seen.append(tuple(rows))
            return original(tree, values, bins, gradients, hessians, rows, depth)

        with mock.patch.object(structural._NewtonTree, "_leafwise_candidate", new=capture):
            tree = make_tree(leaves=4).fit(binned, [2, 2], gradients, np.ones(40))
        self.assertEqual(len(rows_seen), len(set(rows_seen)))
        self.assertEqual([item["node_id"] for item in tree.split_history_], [0, 1, 2])
        repeat = make_tree(leaves=4).fit(binned, [2, 2], gradients, np.ones(40))
        self.assertEqual(tree.root_, repeat.root_)

    def test_one_split_matches_historical_evaluator_including_endpoint_behavior(self):
        rng = np.random.default_rng(12)
        binned = rng.integers(0, 5, size=(60, 4), dtype=np.uint8)
        binned[::5, 1] = 255
        gradients = rng.normal(size=60)
        hessians = rng.uniform(.1, .3, size=60)
        for exact in (None, np.array([False, True, False, False])):
            with self.subTest(exact=exact):
                reference = historical._NewtonTree(1, 3, 1., 0., 3, np.random.default_rng(11), "recursive", exact).fit(
                    binned, [5]*4, gradients, hessians)
                actual = structural._NewtonTree(8, 3, 1., 0., 3, np.random.default_rng(11), "recursive", exact,
                                                 growth_policy="leafwise", max_leaves=2).fit(binned, [5]*4, gradients, hessians)
                np.testing.assert_array_equal(actual.predict(binned), reference.predict(binned))
                for attr in ("feature", "split_bin", "missing_go_left"):
                    self.assertEqual(getattr(actual.root_, attr), getattr(reference.root_, attr))
        # The historical quantile search omits the final boundary when there
        # are multiple bins. This trial deliberately does not fix that rule.
        binned = np.repeat([0, 1, 255], 8).reshape(-1, 1).astype(np.uint8)
        gradients = np.repeat([2., 2., -2.], 8)
        reference = historical._NewtonTree(1, 2, 1., 0., None, np.random.default_rng(1), "recursive").fit(
            binned, [2], gradients, np.ones(24))
        actual = make_tree(leaves=2).fit(binned, [2], gradients, np.ones(24))
        np.testing.assert_array_equal(actual.predict(binned), reference.predict(binned))
        self.assertEqual(actual.root_.split_bin, 0)

    def test_both_missing_directions_and_minimum_leaf_size(self):
        bins = np.repeat([0, 1, 255], 8).reshape(-1, 1).astype(np.uint8)
        for missing_gradient, expected_left in ((2., False), (-2., True)):
            gradients = np.repeat([-2., 2., missing_gradient], 8)
            tree = make_tree(leaves=2, leaf=4).fit(bins, [2], gradients, np.ones(24))
            self.assertEqual(tree.root_.missing_go_left, expected_left)
            assert_tree_rows(self, tree, bins, min_leaf=4, depth_limit=5, leaf_limit=2)
        tiny = make_tree(leaf=13).fit(bins, [2], np.repeat([-2., 2., 2.], 8), np.ones(24))
        self.assertIsNone(tiny.root_.feature)
        stopped = make_tree(gain=1e9).fit(bins, [2], np.repeat([-2., 2., 2.], 8), np.ones(24))
        self.assertIsNone(stopped.root_.feature)

    def test_budget_one_and_reachable_32_leaf_depth_bound(self):
        combinations = ((np.arange(32)[:, None] >> np.arange(5)) & 1).astype(np.uint8)
        binned = np.repeat(combinations, 10, axis=0)
        gradients = np.repeat(np.random.default_rng(14).normal(size=32), 10)
        tree = make_tree(depth=8, leaf=5, leaves=32, l2=0.).fit(binned, [2]*5, gradients, np.ones(320))
        leaves = assert_tree_rows(self, tree, binned, min_leaf=5, depth_limit=8, leaf_limit=32)
        self.assertEqual(len(leaves), tree.leaf_count_)
        self.assertEqual(len(leaves), 32)
        self.assertEqual(sum(len(rows) for rows, _ in leaves), len(binned))
        small_depth = make_tree(depth=2, leaf=5, leaves=32, l2=0.).fit(binned, [2]*5, gradients, np.ones(320))
        self.assertLessEqual(small_depth.leaf_count_, 4)
        one = make_tree(leaves=1)
        state = copy.deepcopy(one.random_generator.bit_generator.state)
        one.fit(binned, [2]*5, gradients, np.ones(320))
        self.assertIsNone(one.root_.feature)
        self.assertEqual(one.leaf_count_, 1)
        self.assertEqual(one.split_history_, [])
        self.assertEqual(state, one.random_generator.bit_generator.state)


class TestStructuralParityAndContinuation(unittest.TestCase):
    def test_disabled_options_preserve_historical_bins_scores_and_feature_rng(self):
        rng = np.random.default_rng(16)
        features = rng.normal(size=(113, 6)).astype(np.float32)
        features[:, 0] = np.arange(len(features)) % 4
        features[::7, 3] = np.nan
        labels = (rng.random(len(features)) < .4).astype(int)
        probe = np.r_[features[::3], np.full((2, 6), np.nan)]
        for strategy in ("recursive", "levelwise"):
            for binning in ("quantile", "exact_low_cardinality", "exact_selected"):
                for weight in (1., 2.):
                    with self.subTest(strategy=strategy, binning=binning, weight=weight):
                        parameters = dict(n_estimators=5, max_depth=3, min_samples_leaf=3, n_bins=8,
                                          max_features=3, random_seed=20260920, histogram_strategy=strategy,
                                          binning_strategy=binning, positive_class_weight=weight,
                                          exact_feature_indices=[0] if binning == "exact_selected" else None)
                        reference = historical.HistogramGradientBoostingClassifier(**parameters).fit(features, labels)
                        actual = structural.HistogramGradientBoostingClassifier(**parameters).fit(features, labels)
                        np.testing.assert_array_equal(actual.training_scores_, reference.training_scores_)
                        np.testing.assert_array_equal(actual.predict_proba(probe), reference.predict_proba(probe))
                        np.testing.assert_array_equal(actual.binner_.transform(probe), reference.binner_.transform(probe))
                        self.assertEqual(actual.random_generator_state_, reference.random_generator_state_)
                        self.assertIsNone(actual.sampling_generator_state_)
                        for tree in actual.trees_:
                            self.assertEqual(tree.sample_row_count_, len(labels))
                            self.assertEqual(tree.sample_row_indices_sha256_, structural._row_indices_sha256(np.arange(len(labels))))

    def test_25_tree_interruption_checkpoint_resumes_exactly_for_both_trials(self):
        rng = np.random.default_rng(24)
        features = rng.normal(size=(150, 7)).astype(np.float32)
        features[::7, 3] = np.nan
        labels = ((features[:, 0] > .2) | (features[:, 2] < -.6)).astype(int)
        local = Path(__file__).resolve().parents[1] / ".local"
        local.mkdir(exist_ok=True)
        for override in ({"subsample": .8}, {"growth_policy": "leafwise", "max_leaves": 8}):
            with self.subTest(override=override), tempfile.TemporaryDirectory(dir=local) as temporary:
                parameters = dict(n_estimators=50, max_depth=4, min_samples_leaf=3, n_bins=8,
                                  max_features=3, random_seed=20260920, **override)
                full = structural.HistogramGradientBoostingClassifier(**parameters).fit(features, labels)
                checkpoint = Path(temporary) / "model.pkl"

                def stop(step, model):
                    model.save_checkpoint(checkpoint, {"parameters": parameters})
                    raise RuntimeError("interrupted after saved checkpoint")

                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    structural.HistogramGradientBoostingClassifier(**parameters).fit(features, labels, [25], stop)
                resumed, metadata = structural.HistogramGradientBoostingClassifier.load_checkpoint(checkpoint)
                self.assertEqual(metadata, {"parameters": parameters})
                self.assertEqual(resumed.n_estimators, 50)
                self.assertEqual(len(resumed.trees_), 25)
                partial = resumed.predict_proba(features)[:, 1]
                resumed.continue_fit(features, labels, 25)
                np.testing.assert_array_equal(full.predict_proba(features), resumed.predict_proba(features))
                np.testing.assert_array_equal(full.training_scores_, resumed.training_scores_)
                self.assertEqual(full.random_generator_state_, resumed.random_generator_state_)
                self.assertEqual(full.sampling_generator_state_, resumed.sampling_generator_state_)
                self.assertEqual([t.sample_row_indices_sha256_ for t in full.trees_],
                                 [t.sample_row_indices_sha256_ for t in resumed.trees_])
                staged = dict(full.staged_predict_proba(features, [25, 50]))
                np.testing.assert_array_equal(staged[25], partial)
                np.testing.assert_array_equal(staged[50], resumed.predict_proba(features)[:, 1])

    def test_parameter_validation(self):
        for subsample in (0, -.1, 1.1, np.nan, np.inf, True, "0.8"):
            with self.subTest(subsample=subsample), self.assertRaises(ValueError):
                structural.HistogramGradientBoostingClassifier(subsample=subsample)
        for leaves in (None, 0, -1, 2.5, True):
            with self.subTest(leaves=leaves), self.assertRaises(ValueError):
                structural.HistogramGradientBoostingClassifier(growth_policy="leafwise", max_leaves=leaves)
        for parameters in ({"growth_policy": "unknown"}, {"max_leaves": 32},
                           {"growth_policy": "leafwise", "max_leaves": 4, "histogram_strategy": "levelwise"}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                structural.HistogramGradientBoostingClassifier(**parameters)


if __name__ == "__main__":
    unittest.main()

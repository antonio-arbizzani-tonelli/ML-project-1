"""Run in a fresh process: this study imports the historical numerical snapshot."""
import unittest
import numpy as np
from tools.phase37_undersampling_ratios import configuration, partitions, select_positions


class RatioSamplingTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.random.default_rng(19).permutation(np.r_[np.ones(15, dtype=np.int8), np.zeros(185, dtype=np.int8)])

    def test_budget_all_positives_and_unique_original_order(self):
        for ratio in (5, 7):
            selected = select_positions(self.labels, ratio, 20261008)
            self.assertEqual(len(selected), 15*(ratio+1))
            self.assertEqual(int(self.labels[selected].sum()), 15)
            self.assertTrue(np.all(np.isin(np.flatnonzero(self.labels), selected)))
            np.testing.assert_array_equal(selected, np.unique(selected))

    def test_negative_samples_are_nested_for_all_tested_seeds(self):
        for seed in range(20):
            small = select_positions(self.labels, 5, seed)
            large = select_positions(self.labels, 7, seed)
            self.assertTrue(np.all(np.isin(small, large)))
            np.testing.assert_array_equal(small, select_positions(self.labels, 5, seed))

    def test_independent_permutation_oracle(self):
        negative = np.flatnonzero(self.labels == 0)
        permutation = np.random.default_rng(42).permutation(negative)
        expected = np.sort(np.r_[np.flatnonzero(self.labels == 1), permutation[:75]])
        np.testing.assert_array_equal(select_positions(self.labels, 5, 42), expected)

    def test_cap_without_replacement_and_invalid_inputs(self):
        np.testing.assert_array_equal(select_positions(np.array([1, 0, 1, 0]), 7, 2), np.arange(4))
        for labels, ratio in ((np.ones(10), 5), (np.zeros(10), 7), (np.array([0, 2, 1]), 5), (self.labels, 0), (self.labels, 5.0)):
            with self.assertRaises(ValueError):
                select_positions(labels, ratio, 1)

    def test_nested_partitions_preserve_full_validation_and_exclude_reserved_rows(self):
        labels = np.tile(self.labels, 6)
        development = np.arange(200, 1200)
        suite = {"seed": 20260919, "nested_threshold": {"inner_seed": 20260922}}
        tasks = list(partitions(labels, development, suite))
        self.assertEqual(len(tasks), 9)
        np.testing.assert_array_equal(np.sort(np.concatenate([t[2] for t in tasks[:3]])), development)
        for index, (_, train, valid) in enumerate(tasks):
            self.assertEqual(np.intersect1d(train, valid).size, 0)
            self.assertTrue(np.all(np.isin(np.r_[train, valid], development)))
            if index >= 3:
                self.assertEqual(np.intersect1d(np.r_[train, valid], tasks[(index-3)//2][2]).size, 0)
            for ratio in (5, 7):
                selected = select_positions(labels[train], ratio, index)
                self.assertEqual(np.intersect1d(train[selected], valid).size, 0)
                self.assertEqual(int(labels[train[selected]].sum()), int(labels[train].sum()))

    def test_configs_hold_historical_model_and_threshold_protocol_fixed(self):
        small, large = configuration(5), configuration(7)
        for key in ("models", "nested_threshold", "data", "preprocessing_config_path", "seed", "split_id"):
            self.assertEqual(small[key], large[key])
        self.assertEqual(small["sampling"]["seed"], large["sampling"]["seed"])


if __name__ == "__main__":
    unittest.main()

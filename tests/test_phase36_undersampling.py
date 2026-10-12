"""Sampling invariants that protect minority rows and nested validation boundaries."""
import unittest
import numpy as np
from tools.phase36_undersampling import partitions, select_positions


class UndersamplingTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.r_[np.ones(10, dtype=np.int8), np.zeros(190, dtype=np.int8)]

    def test_random_keeps_every_positive_and_exact_budget_without_duplicates(self):
        selected = select_positions(self.labels, "random", 42)
        self.assertEqual(selected.size, 40)
        np.testing.assert_array_equal(selected[:10], np.arange(10))
        self.assertEqual(np.unique(selected).size, selected.size)
        np.testing.assert_array_equal(selected, select_positions(self.labels, "random", 42))
        self.assertFalse(np.array_equal(selected, select_positions(self.labels, "random", 43)))

    def test_hybrid_preserves_uniform_half_and_hardest_remaining_negatives(self):
        scores = np.linspace(0, 1, self.labels.size)
        selected = select_positions(self.labels, "hybrid_hard", 42, scores)
        uniform = np.random.default_rng(42).choice(np.arange(10, 200), 15, replace=False)
        remaining = np.setdiff1d(np.arange(10, 200), uniform)
        # A separately constructed oracle ranks monotonically increasing scores.
        expected = np.sort(np.r_[np.arange(10), uniform, remaining[-15:]])
        np.testing.assert_array_equal(selected, expected)

    def test_hybrid_ties_are_resolved_by_original_position(self):
        selected = select_positions(self.labels, "hybrid_hard", 42, np.zeros(200))
        uniform = np.random.default_rng(42).choice(np.arange(10, 200), 15, replace=False)
        remaining = np.setdiff1d(np.arange(10, 200), uniform)
        np.testing.assert_array_equal(selected, np.sort(np.r_[np.arange(10), uniform, remaining[:15]]))

    def test_no_oversampling_when_negative_budget_exceeds_available_rows(self):
        labels = np.array([1, 0, 1, 0, 1])
        for method in ("random", "hybrid_hard"):
            np.testing.assert_array_equal(select_positions(labels, method, 1, np.zeros(5)), np.arange(5))

    def test_invalid_targets_and_pilot_scores_rejected(self):
        for labels in (np.zeros(10), np.ones(10), np.array([0, 2, 1])):
            with self.assertRaises(ValueError):
                select_positions(labels, "random", 1)
        for scores in (np.zeros(199), np.full(200, np.nan)):
            with self.assertRaises(ValueError):
                select_positions(self.labels, "hybrid_hard", 1, scores)

    def test_nested_partitions_exclude_outer_validation_and_reserved_rows(self):
        labels = np.tile(self.labels, 6)
        development = np.arange(200, 1200)
        suite = {"seed": 20260919, "nested_threshold": {"inner_seed": 20260922}}
        tasks = list(partitions(labels, development, suite))
        self.assertEqual(len(tasks), 9)
        np.testing.assert_array_equal(np.sort(np.concatenate([t[2] for t in tasks[:3]])), development)
        for i, (_, train, valid) in enumerate(tasks):
            self.assertEqual(np.intersect1d(train, valid).size, 0)
            self.assertTrue(np.all(np.isin(np.r_[train, valid], development)))
            if i >= 3:
                outer = (i - 3) // 2
                self.assertEqual(np.intersect1d(np.r_[train, valid], tasks[outer][2]).size, 0)
            selected = select_positions(labels[train], "random", 100 + i)
            self.assertEqual(np.intersect1d(train[selected], valid).size, 0)
            self.assertEqual(int(labels[train[selected]].sum()), int(labels[train].sum()))


if __name__ == "__main__":
    unittest.main()

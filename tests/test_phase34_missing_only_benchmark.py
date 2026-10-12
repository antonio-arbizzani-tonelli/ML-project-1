"""Independent partition and diagnostic checks for the controlled fix study."""

import unittest
from types import SimpleNamespace

import numpy as np

from tools.phase34_missing_only_benchmark import _partitions, _pure_missing_nodes


class TestMissingOnlyBenchmark(unittest.TestCase):
    def test_nine_partitions_keep_inner_fits_inside_outer_training(self):
        labels = np.tile([0, 0, 0, 0, 1], 30)
        development = np.random.default_rng(8).permutation(len(labels))
        suite = {"seed": 19, "nested_threshold": {"inner_seed": 22}}
        partitions = list(_partitions(labels, development, suite))
        self.assertEqual(len(partitions), 9)
        held_out = []
        for name, fold, train, valid in partitions:
            self.assertEqual(np.intersect1d(train, valid).size, 0)
            if "_inner_" not in name:
                held_out.append(valid)
                np.testing.assert_array_equal(np.sort(np.r_[train, valid]), np.sort(development))
            else:
                outer_train = partitions[fold-1][2]
                np.testing.assert_array_equal(np.sort(np.r_[train, valid]), np.sort(outer_train))
                self.assertEqual(np.intersect1d(valid, held_out[fold-1]).size, 0)
        np.testing.assert_array_equal(np.sort(np.concatenate(held_out)), np.sort(development))

    def test_missing_split_counter_detects_node_local_equivalent_boundaries(self):
        leaf = SimpleNamespace(feature=None)
        node = SimpleNamespace(feature=0, split_bin=0, missing_go_left=False, left=leaf, right=leaf)
        model = SimpleNamespace(trees_=[SimpleNamespace(root_=node)],
            binner_=SimpleNamespace(bin_counts_=np.array([4]), transform=lambda x: x))
        binned = np.array([[0], [0], [255], [255]], dtype=np.uint8)
        result = _pure_missing_nodes(model, binned)
        self.assertEqual(result["selected_nodes"], 1)
        self.assertEqual(result["global_endpoint_nodes"], 0)
        self.assertEqual(result["by_feature_index"], {"0": 1})

    def test_missing_split_counter_excludes_mixed_observed_children(self):
        leaf = SimpleNamespace(feature=None)
        node = SimpleNamespace(feature=0, split_bin=1, missing_go_left=False, left=leaf, right=leaf)
        model = SimpleNamespace(trees_=[SimpleNamespace(root_=node)],
            binner_=SimpleNamespace(bin_counts_=np.array([4]), transform=lambda x: x))
        result = _pure_missing_nodes(model, np.array([[0], [2], [255]], dtype=np.uint8))
        self.assertEqual(result["selected_nodes"], 0)


if __name__ == "__main__":
    unittest.main()

"""Verify that the raw booster benchmark receives unchanged source values."""

import unittest

import numpy as np

from src.feature_preprocessing import tree_feature_matrices


class TestRawTreePreprocessing(unittest.TestCase):
    def test_all_source_types_codes_and_missing_values_are_preserved(self):
        names = ["ID", "DATE", "WEIGHT", "BINARY", "DAYS", "EMPTY"]
        metadata = [
            {"name": "ID", "semantic_type": "identifier"},
            {"name": "DATE", "semantic_type": "date"},
            {"name": "WEIGHT", "semantic_type": "survey_design"},
            {
                "name": "BINARY",
                "semantic_type": "binary",
                "documented_values": [
                    {"code": "1", "label": "Yes"},
                    {"code": "2", "label": "No"},
                ],
                "special_values": [
                    {"code": "7", "label": "Don't know"},
                    {"code": "9", "label": "Refused"},
                ],
            },
            {
                "name": "DAYS",
                "semantic_type": "count",
                "documented_values": [{"code": "88", "label": "None"}],
                "special_values": [{"code": "77", "label": "Don't know"}],
            },
            {"name": "EMPTY", "semantic_type": "continuous"},
        ]
        train = np.array(
            [[101, 201501, 0.5, 7, 88, np.nan],
             [102, 201502, 2.5, 2, 77, np.nan]], dtype=np.float32
        )
        validation = np.array(
            [[103, 201503, 3.5, 9, np.nan, np.nan],
             [104, 201504, 1.5, 1, 99, np.nan]], dtype=np.float32
        )
        raw_train, raw_validation, raw_names = tree_feature_matrices(
            train, validation, names, metadata, {"name": "raw_tree_passthrough"}
        )

        np.testing.assert_array_equal(raw_train, train)
        np.testing.assert_array_equal(raw_validation, validation)
        self.assertEqual(raw_names, names)
        self.assertEqual(raw_train.dtype, np.float32)
        self.assertEqual(raw_validation.dtype, np.float32)

    def test_outputs_are_independent_copies(self):
        train = np.array([[7.0], [np.nan]], dtype=np.float32)
        validation = np.array([[9.0]], dtype=np.float32)
        names = ["ANSWER"]
        raw_train, raw_validation, raw_names = tree_feature_matrices(
            train, validation, names, [], {"name": "raw_tree_passthrough"}
        )

        self.assertFalse(np.shares_memory(raw_train, train))
        self.assertFalse(np.shares_memory(raw_validation, validation))
        raw_train[0, 0] = 0
        raw_validation[0, 0] = 0
        raw_names[0] = "CHANGED"
        np.testing.assert_array_equal(train, [[7.0], [np.nan]])
        np.testing.assert_array_equal(validation, [[9.0]])
        self.assertEqual(names, ["ANSWER"])

    def test_common_shape_and_infinite_value_checks_still_apply(self):
        cases = [
            (np.array([1.0]), np.array([[2.0]]), ["X"]),
            (np.array([[1.0]]), np.array([[2.0, 3.0]]), ["X"]),
            (np.array([[1.0]]), np.array([[2.0]]), ["X", "Y"]),
            (np.array([[np.inf]]), np.array([[2.0]]), ["X"]),
            (np.array([[1.0]]), np.array([[-np.inf]]), ["X"]),
        ]
        for train, validation, names in cases:
            with self.subTest(train=train, validation=validation, names=names):
                with self.assertRaises(ValueError):
                    tree_feature_matrices(
                        train, validation, names, [], {"name": "raw_tree_passthrough"}
                    )

    def test_cleaning_options_are_rejected_instead_of_silently_ignored(self):
        for option, value in (
            ("exclude_features", []),
            ("missing_code_overrides", {"X": [{"code": 7, "reason": "unknown"}]}),
            ("tree_one_hot_features", ["X"]),
            ("imputation", None),
        ):
            with self.subTest(option=option):
                with self.assertRaisesRegex(ValueError, "without cleaning options"):
                    tree_feature_matrices(
                        np.array([[7.0]]), np.array([[9.0]]), ["X"], [],
                        {"name": "raw_tree_passthrough", option: value},
                    )

    def test_duplicate_source_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Feature names must be unique"):
            tree_feature_matrices(
                np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]]), ["X", "X"], [],
                {"name": "raw_tree_passthrough"},
            )


if __name__ == "__main__":
    unittest.main()

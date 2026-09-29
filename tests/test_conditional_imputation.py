"""Tests for the optional, fold-fitted imputation challenger."""

import unittest

import numpy as np

from src.feature_preprocessing import tree_feature_matrices


class TestConditionalImputation(unittest.TestCase):
    def setUp(self):
        self.names = ["ANSWER", "ANCHOR", "MEASURE", "SKIPPED", "DERIVED"]
        self.metadata = [
            {
                "name": "ANSWER", "source_kind": "core_survey", "semantic_type": "categorical",
                "special_values": [
                    {"code": "7", "label": "Don't know/Not Sure"},
                    {"code": "9", "label": "Refused"},
                    {"code": "8", "label": "Don't know/Refused/Missing"},
                ],
            },
            {"name": "ANCHOR", "source_kind": "core_survey", "semantic_type": "ordinal"},
            {"name": "MEASURE", "source_kind": "derived", "semantic_type": "continuous"},
            {"name": "SKIPPED", "source_kind": "derived", "semantic_type": "continuous"},
            {
                "name": "DERIVED", "source_kind": "derived", "semantic_type": "binary",
                "special_values": [{"code": "9", "label": "Don't know/Not Sure"}],
            },
        ]
        self.train = np.asarray(
            [
                [1, 1, 10, np.nan, 1], [1, 1, 12, np.nan, 1],
                [2, 2, 20, 4, 2], [2, 2, 22, 5, 2],
                [7, 1, np.nan, np.nan, 9],
                [9, 2, np.nan, np.nan, 9],
                [np.nan, 1, np.nan, np.nan, np.nan],
            ], dtype=np.float32
        )
        self.validation = np.asarray(
            [[7, 1, np.nan, np.nan, 9], [9, 2, np.nan, np.nan, 9],
             [np.nan, 1, np.nan, np.nan, np.nan],
             [8, 1, np.nan, np.nan, np.nan]], dtype=np.float32
        )
        self.settings = {
            "nonresponse_source_kinds": ["core_survey"],
            "continuous_blank_features": ["MEASURE"],
            "anchor_features": ["ANCHOR"],
            "minimum_observed": 2,
            "anchor_bins": 4,
            "ridge": 0.01,
        }

    def test_imputes_only_selected_states_and_preserves_source(self):
        original = self.train.copy()
        train, validation, names = tree_feature_matrices(
            self.train, self.validation, self.names, self.metadata,
            {"imputation": self.settings},
        )
        pos = {name: index for index, name in enumerate(names)}
        self.assertIn(train[4, pos["ANSWER"]], (1, 2))
        self.assertIn(validation[1, pos["ANSWER"]], (1, 2))
        self.assertTrue(np.isnan(train[6, pos["ANSWER"]]))
        self.assertTrue(np.isnan(validation[3, pos["ANSWER"]]))
        self.assertTrue(np.isfinite(validation[0, pos["MEASURE"]]))
        self.assertTrue(np.isnan(validation[0, pos["SKIPPED"]]))
        self.assertTrue(np.isnan(validation[0, pos["DERIVED"]]))
        self.assertEqual(validation[0, pos["ANSWER__was_unknown"]], 1)
        self.assertEqual(validation[1, pos["ANSWER__was_refused"]], 1)
        self.assertEqual(validation[2, pos["ANSWER__was_unknown"]], 0)
        self.assertEqual(validation[3, pos["ANSWER__was_unknown"]], 0)
        self.assertEqual(validation[0, pos["MEASURE__was_blank"]], 1)
        np.testing.assert_array_equal(self.train, original)

    def test_derived_nonresponse_is_explicitly_configurable(self):
        settings = dict(self.settings, nonresponse_source_kinds=["core_survey", "derived"])
        _, validation, names = tree_feature_matrices(
            self.train, self.validation, self.names, self.metadata,
            {"imputation": settings},
        )
        pos = {name: index for index, name in enumerate(names)}
        self.assertTrue(np.isfinite(validation[0, pos["DERIVED"]]))
        self.assertTrue(np.isnan(validation[2, pos["DERIVED"]]))
        self.assertEqual(validation[0, pos["DERIVED__was_unknown"]], 1)

    def test_flags_only_does_not_fill_missing_values(self):
        settings = dict(self.settings, fill_values=False)
        _, validation, names = tree_feature_matrices(
            self.train, self.validation, self.names, self.metadata,
            {"imputation": settings},
        )
        pos = {name: index for index, name in enumerate(names)}
        self.assertTrue(np.isnan(validation[0, pos["ANSWER"]]))
        self.assertTrue(np.isnan(validation[0, pos["MEASURE"]]))
        self.assertEqual(validation[0, pos["ANSWER__was_unknown"]], 1)

    def test_nonresponse_allowlist_limits_replaced_features(self):
        settings = dict(
            self.settings,
            nonresponse_source_kinds=["core_survey", "derived"],
            nonresponse_features=["ANSWER"],
        )
        _, validation, names = tree_feature_matrices(
            self.train, self.validation, self.names, self.metadata,
            {"imputation": settings},
        )
        pos = {name: index for index, name in enumerate(names)}
        self.assertTrue(np.isfinite(validation[0, pos["ANSWER"]]))
        self.assertTrue(np.isnan(validation[0, pos["DERIVED"]]))
        self.assertNotIn("DERIVED__was_unknown", pos)

    def test_bmi_is_derived_after_height_and_weight_are_imputed(self):
        names = ["HTM4", "WTKG3", "_BMI5", "ANCHOR"]
        metadata = [
            {"name": name, "source_kind": "derived", "semantic_type": "continuous"}
            for name in names[:3]
        ] + [{"name": "ANCHOR", "source_kind": "core_survey", "semantic_type": "ordinal"}]
        train = np.array(
            [
                [1.6, 60, 23.44, 1], [1.6, 64, 25.00, 1],
                [1.8, 80, 24.69, 2], [1.8, 84, 25.93, 2],
                [np.nan, 60, np.nan, 1], [1.8, np.nan, np.nan, 2],
            ], dtype=np.float32,
        )
        validation = np.array(
            [[np.nan, 62, np.nan, 1], [1.8, np.nan, np.nan, 2],
             [1.6, 60, 23.44, 1]], dtype=np.float32,
        )
        settings = {
            "continuous_blank_features": ["HTM4", "WTKG3"],
            "derive_bmi_from_height_weight": True,
            "anchor_features": ["ANCHOR"],
            "minimum_observed": 2,
            "ridge": 0.01,
        }
        _, transformed, output_names = tree_feature_matrices(
            train, validation, names, metadata, {"imputation": settings}
        )
        pos = {name: index for index, name in enumerate(output_names)}
        for row in (0, 1):
            self.assertTrue(np.isfinite(transformed[row, pos["_BMI5"]]))
            expected = round(
                float(transformed[row, pos["WTKG3"]])
                / float(transformed[row, pos["HTM4"]]) ** 2, 2
            )
            self.assertAlmostEqual(transformed[row, pos["_BMI5"]], expected, places=2)
            self.assertEqual(transformed[row, pos["_BMI5__was_blank"]], 1)
        self.assertEqual(transformed[2, pos["_BMI5"]], validation[2, 2])
        self.assertEqual(transformed[2, pos["_BMI5__was_blank"]], 0)


if __name__ == "__main__":
    unittest.main()

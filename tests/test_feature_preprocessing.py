"""Tests for codebook-aware, fold-safe Phase 3 preprocessing."""

import unittest
from pathlib import Path

import numpy as np

from src.feature_preprocessing import (
    FeaturePreprocessor,
    load_feature_metadata,
    tree_feature_matrices,
)


def metadata():
    """Return a compact registry with the response conventions under test."""

    return [
        {
            "name": "BINARY",
            "semantic_type": "binary",
            "documented_values": [
                {"code": "1", "label": "Yes"},
                {"code": "2", "label": "No"},
                {"code": "7", "label": "Don't know/Not Sure"},
                {"code": "9", "label": "Refused"},
            ],
            "special_values": [
                {"code": "7", "label": "Don't know/Not Sure"},
                {"code": "9", "label": "Refused"},
            ],
        },
        {
            "name": "DAYS",
            "semantic_type": "count",
            "documented_values": [
                {"code": "1 - 30", "label": "Number of days"},
                {"code": "88", "label": "None"},
                {"code": "77", "label": "Don't know/Not sure"},
                {"code": "99", "label": "Refused"},
            ],
            "special_values": [
                {"code": "77", "label": "Don't know/Not sure"},
                {"code": "99", "label": "Refused"},
            ],
        },
        {
            "name": "CATEGORY",
            "semantic_type": "categorical",
            "documented_values": [],
            "special_values": [],
        },
        {
            "name": "CONTINUOUS",
            "semantic_type": "continuous",
            "documented_values": [],
            "special_values": [],
        },
    ]


class TestFeaturePreprocessor(unittest.TestCase):
    """Check semantic response conversion and the training-only fitting boundary."""

    def test_special_codes_are_not_treated_as_numeric_measurements(self):
        plan = {
            "name": "codebook_aware",
            "detailed_missing_features": ["BINARY"],
            "missing_indicator_min_fraction": 0.0,
        }
        train = np.array([[1.0, 88.0, 1.0], [2.0, 3.0, 2.0], [7.0, 77.0, 1.0]])
        validation = np.array([[9.0, 99.0, 2.0]])
        preprocessor = FeaturePreprocessor(
            ["BINARY", "DAYS", "CATEGORY"], metadata(), plan
        ).fit(train)
        transformed = preprocessor.transform(validation)
        names = ("intercept",) + preprocessor.output_feature_names
        positions = {name: index for index, name in enumerate(names)}
        self.assertEqual(transformed[0, positions["BINARY__refused"]], 1.0)
        self.assertEqual(transformed[0, positions["DAYS__missing"]], 1.0)
        self.assertEqual(transformed[0, positions["BINARY__unknown"]], 0.0)
        self.assertTrue(np.all(np.isfinite(transformed)))
        days_train = preprocessor.transform(train)
        self.assertLess(
            days_train[0, positions["DAYS"]], days_train[1, positions["DAYS"]]
        )

    def test_tree_preprocessing_keeps_cleaned_missing_values_unimputed(self):
        plan = {"name": "codebook_aware"}
        train = np.array([[1.0, 88.0, 1.0], [2.0, 3.0, 2.0], [7.0, 77.0, 1.0]])
        validation = np.array([[9.0, 99.0, 2.0]])

        clean_train, clean_validation, names = tree_feature_matrices(
            train, validation, ["BINARY", "DAYS", "CATEGORY"], metadata(), plan
        )

        positions = {name: index for index, name in enumerate(names)}
        self.assertTrue(np.isnan(clean_train[2, positions["BINARY"]]))
        self.assertTrue(np.isnan(clean_train[2, positions["DAYS"]]))
        self.assertTrue(np.isnan(clean_validation[0, positions["BINARY"]]))
        self.assertTrue(np.isnan(clean_validation[0, positions["DAYS"]]))
        self.assertEqual(clean_train.dtype, np.float32)

    def test_categories_and_medians_are_fit_without_validation_information(self):
        plan = {"name": "codebook_aware", "one_hot_features": ["CATEGORY"]}
        train = np.array([[1.0, 1.0, 10.0], [2.0, 2.0, 20.0], [1.0, 3.0, np.nan]])
        validation = np.array([[1.0, 500.0, 30.0]])
        preprocessor = FeaturePreprocessor(
            ["BINARY", "DAYS", "CATEGORY"], metadata(), plan
        ).fit(train)
        self.assertNotIn("CATEGORY__is_30", preprocessor.output_feature_names)
        transformed = preprocessor.transform(validation)
        self.assertTrue(np.all(np.isfinite(transformed)))

    def test_one_hot_categories_can_keep_non_response_reasons(self):
        plan = {
            "name": "codebook_aware",
            "one_hot_features": ["BINARY"],
            "detailed_missing_features": ["BINARY"],
        }
        train = np.array([[1.0, 1.0, 10.0], [2.0, 2.0, 20.0], [7.0, 3.0, 10.0]])
        validation = np.array([[9.0, 1.0, 20.0]])
        preprocessor = FeaturePreprocessor(
            ["BINARY", "DAYS", "CATEGORY"], metadata(), plan
        ).fit(train)
        names = ("intercept",) + preprocessor.output_feature_names
        positions = {name: index for index, name in enumerate(names)}
        transformed = preprocessor.transform(validation)
        self.assertIn("BINARY__is_2", names)
        self.assertEqual(transformed[0, positions["BINARY__refused"]], 1.0)
        self.assertEqual(transformed[0, positions["BINARY__unknown"]], 0.0)

    def test_piecewise_features_fit_knots_on_training_rows_only(self):
        plan = {
            "name": "codebook_aware",
            "piecewise_linear_features": [
                {"feature": "CONTINUOUS", "quantiles": [0.5]}
            ],
        }
        train = np.array(
            [[1.0, 1.0, 10.0, 0.0], [2.0, 2.0, 20.0, 2.0], [1.0, 3.0, 10.0, 4.0]]
        )
        validation = np.array([[1.0, 1.0, 20.0, 100.0]])
        preprocessor = FeaturePreprocessor(
            ["BINARY", "DAYS", "CATEGORY", "CONTINUOUS"], metadata(), plan
        ).fit(train)
        names = ("intercept",) + preprocessor.output_feature_names
        position = names.index("CONTINUOUS__hinge_q0.5")
        transformed = preprocessor.transform(validation)
        train_hinge = np.maximum(train[:, 3] - 2.0, 0.0)
        expected = (98.0 - train_hinge.mean()) / train_hinge.std()
        self.assertAlmostEqual(transformed[0, position], expected, places=5)

    def test_bphigh4_codebook_correction_marks_7_and_9_as_non_responses(self):
        metadata_path = Path("configs/feature_metadata.json")
        metadata = load_feature_metadata(metadata_path)
        plan = {
            "name": "codebook_aware",
            "one_hot_features": ["BPHIGH4"],
            "detailed_missing_features": ["BPHIGH4"],
        }
        values = np.array([[1.0], [3.0], [7.0], [9.0]])
        processor = FeaturePreprocessor(["BPHIGH4"], metadata, plan).fit(values)
        names = ("intercept",) + processor.output_feature_names
        transformed = processor.transform(values)
        self.assertEqual(transformed[2, names.index("BPHIGH4__unknown")], 1.0)
        self.assertEqual(transformed[3, names.index("BPHIGH4__refused")], 1.0)


if __name__ == "__main__":
    unittest.main()

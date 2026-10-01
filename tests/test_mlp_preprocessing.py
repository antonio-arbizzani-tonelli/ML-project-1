"""Check codebook semantics, finite inputs and training-only MLP transformations."""

import unittest

import numpy as np

from src.mlp_preprocessing import MLPPreprocessor, _frequency_values


def metadata(name, kind, codes=()):
    return {"name": name, "semantic_type": kind,
            "documented_values": [{"code": code, "label": label} for code, label in codes]}


class TestMLPPreprocessing(unittest.TestCase):
    def test_scale_and_categories_are_fitted_only_on_training_and_raw_is_preserved(self):
        names = ["numeric", "nominal", "binary", "ordinal"]
        entries = [metadata("numeric", "continuous"), metadata("nominal", "categorical"),
                   metadata("binary", "binary", [(1, "Yes"), (2, "No")]), metadata("ordinal", "ordinal")]
        train = np.array([[1, 10, 1, 1], [3, 20, 2, 3], [np.nan, np.nan, np.nan, np.nan]])
        original = train.copy()
        processor = MLPPreprocessor(names, entries, {}).fit(train)
        transformed = processor.transform(train)
        names = list(processor.output_feature_names)
        self.assertEqual(transformed.dtype, np.float32)
        np.testing.assert_array_equal(train, original)
        self.assertAlmostEqual(float(np.mean(transformed[:, names.index("numeric")])), 0)
        self.assertAlmostEqual(float(np.std(transformed[:, names.index("numeric")])), 1)
        np.testing.assert_array_equal(transformed[:, names.index("binary")], [1, 0, 0])
        before = processor.audit()
        unseen = processor.transform(np.array([[100, 99, 2, 2]]))
        self.assertEqual(processor.audit(), before)
        self.assertEqual(unseen[0, names.index("nominal__unseen")], 1)
        self.assertEqual(unseen[0, names.index("nominal__missing")], 0)
        self.assertGreater(unseen[0, names.index("numeric")], 50)
        self.assertNotIn("intercept", names)

    def test_nominal_reference_missing_and_new_category_are_distinct(self):
        processor = MLPPreprocessor(["C"], [metadata("C", "categorical")], {}).fit([[1], [2]])
        transformed = processor.transform([[1], [2], [np.nan], [3]])
        np.testing.assert_array_equal(transformed, np.eye(4, dtype=np.float32))

    def test_singleton_binary_training_preserves_documented_mapping(self):
        processor = MLPPreprocessor(["B"], [metadata("B", "binary", [(1, "Yes"), (2, "No")])], {}).fit([[1], [1]])
        np.testing.assert_array_equal(processor.transform([[1], [2], [np.nan]]), [[1, 0], [0, 0], [1, 1]])

    def test_all_missing_column_and_unknown_binary_remain_finite(self):
        processor = MLPPreprocessor(["N", "B"], [metadata("N", "count"), metadata("B", "binary", [(1, "Yes"), (2, "No")])], {}).fit([[np.nan, 1], [np.nan, 2]])
        result = processor.transform([[np.nan, 3], [4, 2]])
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertEqual(result[0, 1], 1)
        self.assertEqual(result[0, 3], 1)

    def test_documented_zero_codes_and_nonresponses_remain_different(self):
        plan = {"missing_code_overrides": {"ALCDAY5": [{"code": 777, "reason": "unknown"}, {"code": 999, "reason": "refused"}]},
                "zero_code_overrides": {"ALCDAY5": [888]}}
        processor = MLPPreprocessor(["ALCDAY5"], [metadata("ALCDAY5", "count")], plan,
                                    frequency_features=["ALCDAY5"]).fit([[107], [230], [888], [777]])
        result = processor.transform([[888], [777], [999]])
        np.testing.assert_array_equal(result[:, 1], [0, 1, 1])
        self.assertNotEqual(result[0, 0], result[1, 0])

    def test_frequency_units_convert_consistently_and_censoring_is_not_never(self):
        rates, flags = _frequency_values("FRUIT1", np.array([101, 207, 330, 300, 555, np.nan]))
        np.testing.assert_allclose(rates[:5], [1, 1, 1, 0, 0])
        self.assertTrue(flags["less_than_monthly"][3])
        self.assertFalse(flags["less_than_monthly"][4])
        self.assertTrue(np.isnan(rates[5]))
        rates, _ = _frequency_values("EXEROFT1", np.array([107, 230]))
        np.testing.assert_allclose(rates, [1, 1])
        with self.assertRaises(ValueError):
            _frequency_values("FRUIT1", np.array([444]))

    def test_auxiliary_missing_flag_and_invalid_configuration(self):
        processor = MLPPreprocessor(["A", "N"], [metadata("A", "ordinal"), metadata("N", "continuous")],
            {"exclude_features": ["A"], "auxiliary_missing_features": ["A"]}).fit([[np.nan, 1], [2, 3]])
        np.testing.assert_array_equal(processor.transform([[np.nan, 2], [2, 2]])[:, -1], [1, 0])
        with self.assertRaises(RuntimeError):
            MLPPreprocessor(["N"], [metadata("N", "continuous")], {}).transform([[1]])
        with self.assertRaises(ValueError):
            processor.transform([[1, np.inf]])


if __name__ == "__main__":
    unittest.main()

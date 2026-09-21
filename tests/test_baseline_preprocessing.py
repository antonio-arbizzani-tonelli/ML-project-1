"""Tests for the common fold-safe feature treatment used by first baselines."""

import unittest

import numpy as np

from src.baseline_preprocessing import (
    fit_baseline_preprocessor,
    transform_baseline_features,
)


class TestBaselinePreprocessing(unittest.TestCase):
    """Check that fitting uses development values and creates an intercept."""

    def test_median_and_scaling_are_fitted_only_on_development_rows(self):
        """A validation outlier cannot change training medians, means, or scales."""

        development = np.array([[1.0, np.nan], [3.0, 5.0], [5.0, 7.0]])
        validation = np.array([[np.nan, 9.0]])
        preprocessor = fit_baseline_preprocessor(development)
        design = transform_baseline_features(validation, preprocessor)
        np.testing.assert_allclose(preprocessor.medians, [3.0, 6.0])
        np.testing.assert_allclose(preprocessor.means, [3.0, 6.0])
        np.testing.assert_allclose(design[:, 0], [1.0])
        np.testing.assert_allclose(design[:, 1], [0.0])
        self.assertGreater(design[0, 2], 1.0)

    def test_all_missing_development_column_is_rejected(self):
        """The baseline does not invent a fill value without training evidence."""

        with self.assertRaisesRegex(ValueError, "at least one non-missing"):
            fit_baseline_preprocessor(np.array([[np.nan], [np.nan]]))


if __name__ == "__main__":
    unittest.main()

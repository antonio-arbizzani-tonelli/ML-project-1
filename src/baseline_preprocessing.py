"""Shared, deliberately simple feature treatment for the first baselines.

This module is intentionally narrow.  It establishes one fold-safe numerical
representation for ridge and logistic regression before feature-type-aware
preprocessing is compared in later experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BaselinePreprocessor:
    """Statistics fitted on development rows for median fill and scaling."""

    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray


def _validate_features(features, name: str) -> np.ndarray:
    """Return a two-dimensional numeric feature matrix with no infinite values."""

    features = np.asarray(features)
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array.")
    if not np.issubdtype(features.dtype, np.number):
        raise ValueError(f"{name} must have a numeric dtype.")
    if np.any(np.isinf(features)):
        raise ValueError(f"{name} must not contain infinite values.")
    return features


def fit_baseline_preprocessor(development_features) -> BaselinePreprocessor:
    """Fit median imputation and z-score scaling on development features only.

    Columns that are constant after imputation receive scale one.  This keeps
    the representation finite while leaving their standardized values at zero.
    An all-missing column is rejected because it has no development-derived
    fill value.
    """

    development_features = _validate_features(
        development_features, "development_features"
    )
    values = np.asarray(development_features, dtype=np.float64)
    if np.any(np.all(np.isnan(values), axis=0)):
        raise ValueError(
            "Each feature needs at least one non-missing development value."
        )
    medians = np.nanmedian(values, axis=0)
    imputed = np.where(np.isnan(values), medians, values)
    means = np.mean(imputed, axis=0)
    scales = np.std(imputed, axis=0)
    scales = np.where(scales == 0.0, 1.0, scales)
    return BaselinePreprocessor(medians=medians, means=means, scales=scales)


def transform_baseline_features(
    features, preprocessor: BaselinePreprocessor
) -> np.ndarray:
    """Impute, standardize, and prepend one unregularized-intercept column.

    The result uses ``float32`` to keep the cached design matrices compact.
    The mandatory ridge and logistic functions convert their inputs to
    ``float64`` internally for their numerical calculations.
    """

    features = _validate_features(features, "features")
    if not isinstance(preprocessor, BaselinePreprocessor):
        raise TypeError("preprocessor must be a BaselinePreprocessor instance.")
    if features.shape[1] != preprocessor.medians.size:
        raise ValueError(
            "features have a different column count from the fitted preprocessor."
        )
    values = np.asarray(features, dtype=np.float64)
    imputed = np.where(np.isnan(values), preprocessor.medians, values)
    standardized = (imputed - preprocessor.means) / preprocessor.scales
    design = np.empty((features.shape[0], features.shape[1] + 1), dtype=np.float32)
    design[:, 0] = 1.0
    design[:, 1:] = standardized
    return design

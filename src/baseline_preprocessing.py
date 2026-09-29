"""Shared feature treatment for the first linear baselines.

Fill values and scaling statistics are learned from training rows only. Binary,
categorical, and ordinal columns use their training-fold mode; numeric measures
and counts use their training-fold median.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BaselinePreprocessor:
    """Fill values and scaling statistics fitted on development rows."""

    fill_values: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    @property
    def medians(self) -> np.ndarray:
        """Backward-compatible alias; values may also contain categorical modes."""

        return self.fill_values


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


def _mode(values: np.ndarray) -> float:
    """Return the most frequent observed value, breaking ties by lowest code."""

    categories, counts = np.unique(values, return_counts=True)
    return float(categories[np.argmax(counts)])


def fit_baseline_preprocessor(
    development_features, semantic_types: Sequence[str] | None = None
) -> BaselinePreprocessor:
    """Fit type-aware imputation and z-score scaling on development rows only.

    Columns that are constant after imputation receive scale one.  This keeps
    the representation finite while leaving their standardized values at zero.
    An all-missing column is rejected because it has no development-derived
    fill value. ``binary``, ``categorical``, and ``ordinal`` columns use the
    training mode. Other columns use the training median. When semantic types
    are omitted, every column retains the historical numeric-median behavior.
    """

    development_features = _validate_features(
        development_features, "development_features"
    )
    values = np.asarray(development_features, dtype=np.float64)
    if np.any(np.all(np.isnan(values), axis=0)):
        raise ValueError(
            "Each feature needs at least one non-missing development value."
        )
    if semantic_types is None:
        categorical_mask = np.zeros(values.shape[1], dtype=bool)
    else:
        if len(semantic_types) != values.shape[1]:
            raise ValueError("semantic_types must have one entry per feature column.")
        categorical_mask = np.asarray(
            [kind in {"binary", "categorical", "ordinal"} for kind in semantic_types],
            dtype=bool,
        )
    fill_values = np.empty(values.shape[1], dtype=np.float64)
    for index in range(values.shape[1]):
        observed = values[np.isfinite(values[:, index]), index]
        fill_values[index] = (
            _mode(observed)
            if categorical_mask[index]
            else float(np.median(observed))
        )
    imputed = np.where(np.isnan(values), fill_values, values)
    means = np.mean(imputed, axis=0)
    scales = np.std(imputed, axis=0)
    scales = np.where(scales == 0.0, 1.0, scales)
    return BaselinePreprocessor(fill_values=fill_values, means=means, scales=scales)


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
    if features.shape[1] != preprocessor.fill_values.size:
        raise ValueError(
            "features have a different column count from the fitted preprocessor."
        )
    values = np.asarray(features, dtype=np.float64)
    imputed = np.where(np.isnan(values), preprocessor.fill_values, values)
    standardized = (imputed - preprocessor.means) / preprocessor.scales
    design = np.empty((features.shape[0], features.shape[1] + 1), dtype=np.float32)
    design[:, 0] = 1.0
    design[:, 1:] = standardized
    return design

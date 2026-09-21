"""Codebook-aware, fold-safe preprocessing for Project 1 experiments.

The raw project matrix intentionally preserves BRFSS response codes.  This
module turns documented non-responses into missing values, keeps semantic
numeric values on a coherent scale, and optionally adds a small, configured
set of missingness and categorical features.  All learned quantities are fit
only on the training rows supplied to :meth:`FeaturePreprocessor.fit`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

NUMERIC_SEMANTIC_TYPES = frozenset(
    {"binary", "continuous", "count", "ordinal", "categorical"}
)


def load_feature_metadata(path: Path | str) -> list[dict[str, Any]]:
    """Load the codebook-backed feature registry and validate its basic shape."""

    with Path(path).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, list) or not metadata:
        raise ValueError("Feature metadata must be a non-empty JSON list.")
    names = []
    for entry in metadata:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("name"), str):
            raise ValueError("Each feature metadata entry requires a string name.")
        names.append(entry["name"])
    if len(set(names)) != len(names):
        raise ValueError("Feature metadata contains duplicate feature names.")
    return [dict(entry) for entry in metadata]


def tree_feature_matrices(
    training_features: np.ndarray,
    validation_features: np.ndarray,
    feature_names: Sequence[str],
    metadata: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Clean source values for histogram trees without imputing missingness.

    This shares the codebook interpretation with :class:`FeaturePreprocessor`,
    but does not add an intercept, z-score columns, or missingness indicators.
    Histogram trees route missing values themselves, so replacing them with a
    median would discard a useful state. Binary mappings are fitted from the
    training rows and applied unchanged to validation rows.
    """

    training_features = np.asarray(training_features, dtype=np.float32)
    validation_features = np.asarray(validation_features, dtype=np.float32)
    if (
        training_features.ndim != 2
        or validation_features.ndim != 2
        or training_features.shape[1] != validation_features.shape[1]
        or training_features.shape[1] != len(feature_names)
    ):
        raise ValueError("Tree feature matrices must align with every feature name.")
    if np.any(np.isinf(training_features)) or np.any(np.isinf(validation_features)):
        raise ValueError("Tree feature matrices must not contain infinite values.")

    processor = FeaturePreprocessor(feature_names, metadata, plan)
    output_names = [
        name
        for name in processor._retained_names()
        if str(processor.metadata_by_name[name].get("semantic_type"))
        in NUMERIC_SEMANTIC_TYPES
    ]
    if not output_names:
        raise ValueError("Tree preprocessing retained no usable source features.")
    output_train = np.empty(
        (training_features.shape[0], len(output_names)), dtype=np.float32
    )
    output_validation = np.empty(
        (validation_features.shape[0], len(output_names)), dtype=np.float32
    )
    for output_index, name in enumerate(output_names):
        entry = processor.metadata_by_name[name]
        semantic_type = str(entry.get("semantic_type"))
        index = processor._name_to_index[name]
        missing_codes = processor._missing_codes(name, entry)
        zero_codes = processor._zero_codes(name, entry)
        clean_train, _ = processor._clean_values(
            training_features[:, index], missing_codes, zero_codes
        )
        clean_validation, _ = processor._clean_values(
            validation_features[:, index], missing_codes, zero_codes
        )
        if not np.any(np.isfinite(clean_train)):
            raise ValueError(f"{name} has no usable value in this training fold.")
        if semantic_type == "binary":
            binary_map = _binary_mapping(entry, clean_train)
            for code, mapped in binary_map.items():
                clean_train[clean_train == code] = mapped
                clean_validation[clean_validation == code] = mapped
        output_train[:, output_index] = clean_train
        output_validation[:, output_index] = clean_validation
    return output_train, output_validation, output_names


def _as_finite_code(value: Any) -> float | None:
    """Return a numeric response code or ``None`` for blank/non-numeric labels."""

    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    if not isinstance(value, str):
        return None
    token = value.strip()
    if not token or token.upper() == "BLANK":
        return None
    try:
        numeric = float(token)
    except ValueError:
        return None
    return numeric if np.isfinite(numeric) else None


def _normalise_label(value: Any) -> str:
    """Normalise codebook labels enough for the documented response phrases."""

    return " ".join(str(value).lower().replace("’", "'").replace("´", "'").split())


def _missing_reason(label: Any) -> str | None:
    """Classify a documented non-response without treating valid codes as missing."""

    text = _normalise_label(label)
    if not text or text.startswith(("no missing", "not missing", "included", "has ")):
        return None
    if "don't know" in text or "dont know" in text or "not sure" in text:
        return "unknown"
    if "refused" in text:
        return "refused"
    if "not asked" in text or text == "missing" or text.startswith("missing "):
        return "blank"
    if text in {
        "refused/missing",
        "don't know/refused/missing",
        "dont know/refused/missing",
    }:
        return "blank"
    return None


def _metadata_code_maps(
    entry: Mapping[str, Any],
) -> tuple[dict[float, str], set[float]]:
    """Extract non-response and documented zero codes from one metadata entry."""

    missing_codes: dict[float, str] = {}
    for item in entry.get("special_values", []):
        if not isinstance(item, Mapping):
            continue
        code = _as_finite_code(item.get("code"))
        reason = _missing_reason(item.get("label"))
        if code is not None and reason is not None:
            missing_codes[code] = reason

    zero_codes: set[float] = set()
    if entry.get("semantic_type") == "count":
        for item in entry.get("documented_values", []):
            if not isinstance(item, Mapping):
                continue
            code = _as_finite_code(item.get("code"))
            if code is not None and _normalise_label(item.get("label")) == "none":
                zero_codes.add(code)
    return missing_codes, zero_codes


def _binary_mapping(entry: Mapping[str, Any], values: np.ndarray) -> dict[float, float]:
    """Map binary response values to zero/one while preserving documented meaning."""

    documented: dict[float, float] = {}
    for item in entry.get("documented_values", []):
        if not isinstance(item, Mapping):
            continue
        code = _as_finite_code(item.get("code"))
        label = _normalise_label(item.get("label"))
        if code is None:
            continue
        if label == "yes":
            documented[code] = 1.0
        elif label == "no":
            documented[code] = 0.0

    observed = np.unique(values[np.isfinite(values)])
    if observed.size <= 1:
        return {float(code): 0.0 for code in observed}
    if observed.size == 2:
        first, second = float(observed[0]), float(observed[1])
        if first in documented and second in documented:
            return {first: documented[first], second: documented[second]}
        return {first: 0.0, second: 1.0}
    return {}


@dataclass(frozen=True)
class _ColumnState:
    """Fitted representation of one source column or one auxiliary flag."""

    name: str
    index: int
    semantic_type: str
    missing_codes: Mapping[float, str]
    zero_codes: frozenset[float]
    binary_map: Mapping[float, float]
    fill_value: float | None
    mean: float | None
    scale: float | None
    one_hot_categories: tuple[float, ...]
    add_combined_indicator: bool
    detailed_indicators: tuple[str, ...]
    auxiliary_only: bool = False


@dataclass(frozen=True)
class _PiecewiseLinearState:
    """Fitted hinge terms derived from one continuous source feature."""

    name: str
    index: int
    missing_codes: Mapping[float, str]
    zero_codes: frozenset[float]
    fill_value: float
    quantiles: tuple[float, ...]
    knots: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]


class FeaturePreprocessor:
    """Fit and apply one codebook-aware, explicit feature transformation.

    The ``plan`` is an ordinary JSON-compatible mapping.  It specifies only
    feature-policy choices; medians, scales, binary mappings, and categories
    are learned from the training array during ``fit``.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        metadata: Sequence[Mapping[str, Any]],
        plan: Mapping[str, Any],
    ) -> None:
        self.feature_names = tuple(feature_names)
        self._name_to_index = {
            name: index for index, name in enumerate(self.feature_names)
        }
        if len(self._name_to_index) != len(self.feature_names):
            raise ValueError("Feature names must be unique.")
        metadata_by_name = {str(entry["name"]): dict(entry) for entry in metadata}
        absent = set(self.feature_names) - set(metadata_by_name)
        if absent:
            raise ValueError(
                f"Feature metadata is missing: {', '.join(sorted(absent))}"
            )
        self.metadata_by_name = metadata_by_name
        self.plan = dict(plan)
        self._states: list[_ColumnState] | None = None
        self._piecewise_states: list[_PiecewiseLinearState] = []
        self.output_feature_names: tuple[str, ...] = ()

    def _missing_codes(self, name: str, entry: Mapping[str, Any]) -> dict[float, str]:
        codes, _ = _metadata_code_maps(entry)
        overrides = self.plan.get("missing_code_overrides", {}).get(name, [])
        for override in overrides:
            if not isinstance(override, Mapping):
                raise ValueError(f"Missing-code override for {name} must be a mapping.")
            code = _as_finite_code(override.get("code"))
            reason = str(override.get("reason", "blank"))
            if code is None or reason not in {"unknown", "refused", "blank"}:
                raise ValueError(f"Invalid missing-code override for {name}.")
            codes[code] = reason
        return codes

    def _zero_codes(self, name: str, entry: Mapping[str, Any]) -> frozenset[float]:
        _, codes = _metadata_code_maps(entry)
        for code_value in self.plan.get("zero_code_overrides", {}).get(name, []):
            code = _as_finite_code(code_value)
            if code is None:
                raise ValueError(f"Invalid zero-code override for {name}.")
            codes.add(code)
        return frozenset(codes)

    @staticmethod
    def _clean_values(
        source: np.ndarray,
        missing_codes: Mapping[float, str],
        zero_codes: frozenset[float],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Replace documented non-responses with NaN and retain their source masks."""

        values = np.asarray(source, dtype=np.float64).copy()
        reasons = {
            "unknown": np.zeros(values.size, dtype=bool),
            "refused": np.zeros(values.size, dtype=bool),
            "blank": np.isnan(values),
        }
        for code, reason in missing_codes.items():
            matched = values == code
            reasons[reason] |= matched
            values[matched] = np.nan
        for code in zero_codes:
            values[values == code] = 0.0
        return values, reasons

    def _retained_names(self) -> list[str]:
        excluded_types = set(self.plan.get("exclude_semantic_types", []))
        excluded_names = set(self.plan.get("exclude_features", []))
        unknown = excluded_names - set(self.feature_names)
        if unknown:
            raise ValueError(
                f"Plan excludes unknown features: {', '.join(sorted(unknown))}"
            )
        return [
            name
            for name in self.feature_names
            if name not in excluded_names
            and self.metadata_by_name[name].get("semantic_type") not in excluded_types
        ]

    def fit(self, training_features: np.ndarray) -> "FeaturePreprocessor":
        """Learn feature values, scales, and categories from training rows only."""

        training_features = np.asarray(training_features)
        if training_features.ndim != 2 or training_features.shape[1] != len(
            self.feature_names
        ):
            raise ValueError(
                "training_features must have one column per configured feature name."
            )
        if np.any(np.isinf(training_features)):
            raise ValueError("training_features must not contain infinite values.")

        one_hot_features = set(self.plan.get("one_hot_features", []))
        detailed_features = set(self.plan.get("detailed_missing_features", []))
        auxiliary_features = set(self.plan.get("auxiliary_missing_features", []))
        for group_name, names in (
            ("one_hot_features", one_hot_features),
            ("detailed_missing_features", detailed_features),
            ("auxiliary_missing_features", auxiliary_features),
        ):
            unknown = names - set(self.feature_names)
            if unknown:
                raise ValueError(
                    f"Plan {group_name} contains unknown features: {', '.join(sorted(unknown))}"
                )

        indicator_threshold = self.plan.get("missing_indicator_min_fraction")
        if (
            indicator_threshold is not None
            and not 0.0 <= float(indicator_threshold) <= 1.0
        ):
            raise ValueError(
                "missing_indicator_min_fraction must lie between zero and one."
            )
        piecewise_specs = self.plan.get("piecewise_linear_features", [])
        if not isinstance(piecewise_specs, list):
            raise ValueError(
                "piecewise_linear_features must be a list when configured."
            )

        states: list[_ColumnState] = []
        output_names: list[str] = []
        for name in self._retained_names():
            entry = self.metadata_by_name[name]
            semantic_type = str(entry.get("semantic_type"))
            if semantic_type not in NUMERIC_SEMANTIC_TYPES:
                continue
            index = self._name_to_index[name]
            missing_codes = self._missing_codes(name, entry)
            zero_codes = self._zero_codes(name, entry)
            values, reasons = self._clean_values(
                training_features[:, index], missing_codes, zero_codes
            )
            missing = ~np.isfinite(values)
            if np.all(missing):
                raise ValueError(f"{name} has no usable value in this training fold.")
            detailed = (
                ("unknown", "refused", "blank") if name in detailed_features else ()
            )

            if name in one_hot_features:
                categories = tuple(
                    float(value) for value in np.unique(values[~missing])
                )
                if not categories:
                    raise ValueError(f"{name} has no categories in this training fold.")
                states.append(
                    _ColumnState(
                        name=name,
                        index=index,
                        semantic_type=semantic_type,
                        missing_codes=missing_codes,
                        zero_codes=zero_codes,
                        binary_map={},
                        fill_value=None,
                        mean=None,
                        scale=None,
                        one_hot_categories=categories,
                        add_combined_indicator=False,
                        detailed_indicators=detailed,
                    )
                )
                output_names.extend(f"{name}__is_{value:g}" for value in categories[1:])
                if detailed:
                    output_names.extend(f"{name}__{reason}" for reason in detailed)
                else:
                    output_names.append(f"{name}__missing")
                continue

            binary_map = (
                _binary_mapping(entry, values) if semantic_type == "binary" else {}
            )
            if binary_map:
                converted = values.copy()
                for code, mapped in binary_map.items():
                    converted[values == code] = mapped
                values = converted
            fill_value = float(np.nanmedian(values))
            imputed = np.where(missing, fill_value, values)
            if semantic_type == "binary" and binary_map:
                mean, scale = 0.0, 1.0
            else:
                mean = float(np.mean(imputed))
                scale = float(np.std(imputed))
                if scale == 0.0:
                    scale = 1.0
            add_combined = bool(
                not detailed
                and indicator_threshold is not None
                and float(np.mean(missing)) >= float(indicator_threshold)
            )
            states.append(
                _ColumnState(
                    name=name,
                    index=index,
                    semantic_type=semantic_type,
                    missing_codes=missing_codes,
                    zero_codes=zero_codes,
                    binary_map=binary_map,
                    fill_value=fill_value,
                    mean=mean,
                    scale=scale,
                    one_hot_categories=(),
                    add_combined_indicator=add_combined,
                    detailed_indicators=detailed,
                )
            )
            output_names.append(name)
            if detailed:
                output_names.extend(f"{name}__{reason}" for reason in detailed)
            elif add_combined:
                output_names.append(f"{name}__missing")

        for name in sorted(auxiliary_features):
            entry = self.metadata_by_name[name]
            index = self._name_to_index[name]
            states.append(
                _ColumnState(
                    name=name,
                    index=index,
                    semantic_type=str(entry.get("semantic_type")),
                    missing_codes=self._missing_codes(name, entry),
                    zero_codes=self._zero_codes(name, entry),
                    binary_map={},
                    fill_value=None,
                    mean=None,
                    scale=None,
                    one_hot_categories=(),
                    add_combined_indicator=True,
                    detailed_indicators=(),
                    auxiliary_only=True,
                )
            )
            output_names.append(f"{name}__missing")

        piecewise_states: list[_PiecewiseLinearState] = []
        seen_piecewise_sources: set[str] = set()
        retained = set(self._retained_names())
        for specification in piecewise_specs:
            if not isinstance(specification, Mapping):
                raise ValueError(
                    "Each piecewise-linear feature specification must be a mapping."
                )
            name = specification.get("feature")
            quantiles = specification.get("quantiles")
            if not isinstance(name, str) or name not in self.feature_names:
                raise ValueError(
                    "A piecewise-linear feature must name a configured source column."
                )
            if name not in retained:
                raise ValueError(
                    f"Piecewise-linear source {name} is excluded from the representation."
                )
            if name in seen_piecewise_sources:
                raise ValueError(
                    f"Piecewise-linear source {name} is configured more than once."
                )
            if self.metadata_by_name[name].get("semantic_type") != "continuous":
                raise ValueError(f"Piecewise-linear source {name} must be continuous.")
            if not isinstance(quantiles, list) or not quantiles:
                raise ValueError(
                    f"Piecewise-linear source {name} requires a non-empty quantile list."
                )
            numeric_quantiles = tuple(float(value) for value in quantiles)
            if any(not 0.0 < value < 1.0 for value in numeric_quantiles):
                raise ValueError(
                    "Piecewise-linear quantiles must lie strictly between zero and one."
                )
            if numeric_quantiles != tuple(sorted(set(numeric_quantiles))):
                raise ValueError(
                    "Piecewise-linear quantiles must be strictly increasing."
                )

            index = self._name_to_index[name]
            values, _ = self._clean_values(
                training_features[:, index],
                self._missing_codes(name, self.metadata_by_name[name]),
                self._zero_codes(name, self.metadata_by_name[name]),
            )
            observed = values[np.isfinite(values)]
            if observed.size == 0:
                raise ValueError(
                    f"{name} has no usable values for piecewise-linear features."
                )
            fill_value = float(np.nanmedian(values))
            knots = tuple(
                float(value) for value in np.quantile(observed, numeric_quantiles)
            )
            if len(set(knots)) != len(knots):
                raise ValueError(
                    f"Piecewise-linear quantiles for {name} produce duplicate knots."
                )
            imputed = np.where(np.isfinite(values), values, fill_value)
            means = []
            scales = []
            for knot in knots:
                hinge = np.maximum(imputed - knot, 0.0)
                scale = float(np.std(hinge))
                if scale == 0.0 or not np.isfinite(scale):
                    raise ValueError(
                        f"Piecewise-linear hinge for {name} has zero or non-finite scale."
                    )
                means.append(float(np.mean(hinge)))
                scales.append(scale)
            piecewise_states.append(
                _PiecewiseLinearState(
                    name=name,
                    index=index,
                    missing_codes=self._missing_codes(
                        name, self.metadata_by_name[name]
                    ),
                    zero_codes=self._zero_codes(name, self.metadata_by_name[name]),
                    fill_value=fill_value,
                    quantiles=numeric_quantiles,
                    knots=knots,
                    means=tuple(means),
                    scales=tuple(scales),
                )
            )
            output_names.extend(
                f"{name}__hinge_q{quantile:g}" for quantile in numeric_quantiles
            )
            seen_piecewise_sources.add(name)

        if len(set(output_names)) != len(output_names):
            raise ValueError(
                "The preprocessing plan produces duplicate output feature names."
            )
        self._states = states
        self._piecewise_states = piecewise_states
        self.output_feature_names = tuple(output_names)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply fitted training-fold decisions to a feature matrix."""

        if self._states is None:
            raise RuntimeError("fit must be called before transform.")
        features = np.asarray(features)
        if features.ndim != 2 or features.shape[1] != len(self.feature_names):
            raise ValueError(
                "features must have one column per configured feature name."
            )
        if np.any(np.isinf(features)):
            raise ValueError("features must not contain infinite values.")

        result = np.empty(
            (features.shape[0], len(self.output_feature_names) + 1), dtype=np.float32
        )
        result[:, 0] = 1.0
        output_index = 1
        for state in self._states:
            values, reasons = self._clean_values(
                features[:, state.index], state.missing_codes, state.zero_codes
            )
            missing = ~np.isfinite(values)
            if state.auxiliary_only:
                result[:, output_index] = missing
                output_index += 1
                continue
            if state.one_hot_categories:
                for category in state.one_hot_categories[1:]:
                    result[:, output_index] = values == category
                    output_index += 1
                if state.detailed_indicators:
                    for reason in state.detailed_indicators:
                        result[:, output_index] = reasons[reason]
                        output_index += 1
                else:
                    result[:, output_index] = missing
                    output_index += 1
                continue
            if state.binary_map:
                converted = values.copy()
                for code, mapped in state.binary_map.items():
                    converted[values == code] = mapped
                values = converted
            imputed = np.where(missing, state.fill_value, values)
            result[:, output_index] = (imputed - state.mean) / state.scale
            output_index += 1
            if state.detailed_indicators:
                for reason in state.detailed_indicators:
                    result[:, output_index] = reasons[reason]
                    output_index += 1
            elif state.add_combined_indicator:
                result[:, output_index] = missing
                output_index += 1
        for state in self._piecewise_states:
            values, _ = self._clean_values(
                features[:, state.index], state.missing_codes, state.zero_codes
            )
            imputed = np.where(np.isfinite(values), values, state.fill_value)
            for knot, mean, scale in zip(state.knots, state.means, state.scales):
                result[:, output_index] = (
                    np.maximum(imputed - knot, 0.0) - mean
                ) / scale
                output_index += 1
        if output_index != result.shape[1]:
            raise RuntimeError("The transformed matrix has an unexpected column count.")
        if not np.all(np.isfinite(result)):
            raise ValueError("Preprocessing produced a non-finite design matrix.")
        return result

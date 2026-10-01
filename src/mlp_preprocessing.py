"""Dedicated finite MLP representation; the existing tree preprocessing is unchanged."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.feature_preprocessing import FeaturePreprocessor, NUMERIC_SEMANTIC_TYPES, _as_finite_code, _normalise_label


FOOD_FREQUENCIES = frozenset({"FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1"})
WEEK_MONTH_FREQUENCIES = frozenset({"ALCDAY5", "EXEROFT1", "EXEROFT2"})


def _frequency_values(name, values):
    """Convert documented encoded frequencies to daily rates, retaining units.

    Food code 300 is an interval (less than monthly), not an exact rate: use
    zero in the numeric channel plus a distinct indicator. No exact frequency
    is asserted. Monthly responses use the survey's 30-day convention.
    """
    rate = np.full(values.shape, np.nan, dtype=np.float64)
    flags = {}
    if name in FOOD_FREQUENCIES:
        daily = (values >= 101) & (values <= 199)
        weekly = (values >= 201) & (values <= 299)
        monthly = (values >= 301) & (values <= 399)
        below_monthly = values == 300
        never = (values == 555) | (values == 0)
        rate[daily] = values[daily] - 100
        rate[weekly] = (values[weekly] - 200) / 7
        rate[monthly] = (values[monthly] - 300) / 30
        rate[below_monthly | never] = 0
        flags = {"per_day": daily, "per_week": weekly, "per_month": monthly,
                 "less_than_monthly": below_monthly}
    elif name in WEEK_MONTH_FREQUENCIES:
        weekly = (values >= 101) & (values <= 199)
        monthly = (values >= 201) & (values <= 299)
        rate[weekly] = (values[weekly] - 100) / 7
        rate[monthly] = (values[monthly] - 200) / 30
        rate[values == 0] = 0
        flags = {"per_week": weekly, "per_month": monthly}
    else:
        raise ValueError(f"No documented frequency conversion exists for {name}.")
    invalid = np.isfinite(values) & ~np.isfinite(rate)
    if np.any(invalid):
        raise ValueError(f"Unexpected finite frequency codes for {name}: {np.unique(values[invalid]).tolist()}")
    return rate, flags


@dataclass
class _MLPColumn:
    """Training-only state for one source feature."""
    name: str
    index: int
    kind: str
    missing_codes: dict
    zero_codes: frozenset
    categories: tuple
    binary_map: dict
    fill: float
    mean: float
    scale: float
    frequency_flags: tuple
    missing_fraction: float


class MLPPreprocessor:
    """Reuse codebook cleaning, with full nominal one-hot and explicit missing masks.

    Continuous/count/ordinal columns are imputed and standardized. Binaries,
    nominal one-hot, missing and unseen-category flags remain 0/1. There is no
    intercept column and no tree binning. All-missing training columns use a
    zero numeric placeholder and a missing flag; no held-out values are read.
    """

    def __init__(self, feature_names, metadata, cleaning_plan, frequency_features=()):
        self.cleaner = FeaturePreprocessor(feature_names, metadata, cleaning_plan)
        self.feature_names = tuple(feature_names)
        self.frequency_features = tuple(frequency_features)
        if len(set(self.frequency_features)) != len(self.frequency_features):
            raise ValueError("frequency_features must be unique.")
        retained = set(self.cleaner._retained_names())
        if set(self.frequency_features) - retained:
            raise ValueError("A frequency feature is absent or excluded.")
        if set(self.frequency_features) - (FOOD_FREQUENCIES | WEEK_MONTH_FREQUENCIES):
            raise ValueError("A frequency conversion is not documented.")
        self.states_: list[_MLPColumn] | None = None
        self.output_feature_names: tuple[str, ...] = ()

    def _matrix(self, features):
        """Validate source shape while allowing raw NaN response values."""
        features = np.asarray(features)
        if features.ndim != 2 or features.shape[1] != len(self.feature_names):
            raise ValueError("Source features must align with feature_names.")
        if np.any(np.isinf(features)):
            raise ValueError("Source features must not contain infinity.")
        return features

    def fit(self, training_features):
        """Learn numeric statistics and nominal vocabularies on training rows only."""
        training_features = self._matrix(training_features)
        if training_features.shape[0] == 0:
            raise ValueError("Training features must contain rows.")
        states, output_names = [], []
        for name in self.cleaner._retained_names():
            entry = self.cleaner.metadata_by_name[name]
            kind = str(entry.get("semantic_type"))
            if kind not in NUMERIC_SEMANTIC_TYPES:
                continue
            index = self.cleaner._name_to_index[name]
            missing_codes = self.cleaner._missing_codes(name, entry)
            zero_codes = self.cleaner._zero_codes(name, entry)
            values, _ = self.cleaner._clean_values(training_features[:, index], missing_codes, zero_codes)
            flag_names = ()
            if name in self.frequency_features:
                values, flags = _frequency_values(name, values)
                flag_names = tuple(flags)
            observed = values[np.isfinite(values)]
            categories = tuple(float(v) for v in np.unique(observed))
            binary_map = {}
            if kind == "binary":
                documented = {}
                for item in entry.get("documented_values", []):
                    code = _as_finite_code(item.get("code"))
                    label = _normalise_label(item.get("label"))
                    if code is not None and label in {"yes", "no"}:
                        documented[code] = float(label == "yes")
                if documented and set(categories) <= set(documented):
                    binary_map = documented
                elif set(categories) <= {0.0, 1.0}:
                    binary_map = {0.0: 0.0, 1.0: 1.0}
                elif len(categories) <= 2:
                    # Metadata is fixed, not learned from validation; use its
                    # substantive codes to preserve a singleton training level.
                    codes = sorted({code for item in entry.get("documented_values", [])
                                    if (code := _as_finite_code(item.get("code"))) is not None
                                    and code not in missing_codes})
                    vocabulary = codes if len(codes) == 2 else list(categories)
                    binary_map = {code: float(i) for i, code in enumerate(vocabulary)}
                else:
                    kind = "categorical"
            fill, mean, scale = 0.0, 0.0, 1.0
            if kind == "categorical":
                output_names.extend(f"{name}__is_{value:g}" for value in categories)
                output_names.extend((f"{name}__missing", f"{name}__unseen"))
            else:
                if binary_map:
                    mapped = np.full(values.shape, np.nan)
                    for code, target in binary_map.items():
                        mapped[values == code] = target
                    values = mapped
                    observed = values[np.isfinite(values)]
                if observed.size:
                    if kind in {"binary", "ordinal"}:
                        levels, counts = np.unique(observed, return_counts=True)
                        fill = float(levels[np.argmax(counts)])
                    else:
                        fill = float(np.median(observed))
                imputed = np.where(np.isfinite(values), values, fill)
                if kind != "binary":
                    mean, scale = float(np.mean(imputed)), float(np.std(imputed))
                    if scale == 0:
                        scale = 1.0
                output_names.extend((f"{name}__daily_rate" if flag_names else name,
                                     f"{name}__missing"))
                output_names.extend(f"{name}__{flag}" for flag in flag_names)
            states.append(_MLPColumn(name, index, kind, missing_codes, zero_codes,
                                     categories, binary_map, fill, mean, scale,
                                     flag_names, float(np.mean(~np.isfinite(values)))))
        for name in self.cleaner.plan.get("auxiliary_missing_features", []):
            if name in self.cleaner._retained_names():
                raise ValueError("Auxiliary missing features must be excluded sources.")
            if name not in self.cleaner._name_to_index:
                raise ValueError("Unknown auxiliary missing feature.")
            entry = self.cleaner.metadata_by_name[name]
            states.append(_MLPColumn(name, self.cleaner._name_to_index[name], "auxiliary",
                self.cleaner._missing_codes(name, entry), self.cleaner._zero_codes(name, entry),
                (), {}, 0, 0, 1, (), 0))
            output_names.append(f"{name}__missing")
        if not output_names or len(set(output_names)) != len(output_names):
            raise ValueError("MLP representation is empty or has duplicate output names.")
        self.states_, self.output_feature_names = states, tuple(output_names)
        return self

    def transform(self, features):
        """Apply the frozen training representation, with finite float32 output."""
        if self.states_ is None:
            raise RuntimeError("fit must be called before transform.")
        features = self._matrix(features)
        result = np.empty((features.shape[0], len(self.output_feature_names)), dtype=np.float32)
        position = 0
        for state in self.states_:
            values, _ = self.cleaner._clean_values(features[:, state.index], state.missing_codes, state.zero_codes)
            flags = {}
            if state.frequency_flags:
                values, flags = _frequency_values(state.name, values)
            missing = ~np.isfinite(values)
            if state.kind == "auxiliary":
                result[:, position] = missing
                position += 1
            elif state.kind == "categorical":
                for category in state.categories:
                    result[:, position] = values == category
                    position += 1
                result[:, position] = missing
                result[:, position + 1] = ~missing & ~np.isin(values, state.categories)
                position += 2
            else:
                if state.binary_map:
                    mapped = np.full(values.shape, np.nan)
                    for code, target in state.binary_map.items():
                        mapped[values == code] = target
                    values = mapped
                    missing = ~np.isfinite(values)
                result[:, position] = (np.where(missing, state.fill, values) - state.mean) / state.scale
                result[:, position + 1] = missing
                position += 2
                for flag in state.frequency_flags:
                    result[:, position] = flags[flag]
                    position += 1
        if position != result.shape[1] or not np.all(np.isfinite(result)):
            raise ValueError("MLP preprocessing produced an invalid design matrix.")
        return result

    def audit(self):
        """Describe fitted source types, dimensionality and missingness."""
        if self.states_ is None:
            raise RuntimeError("fit must be called before audit.")
        return {"source_count": len(self.states_), "output_count": len(self.output_feature_names),
                "output_names": list(self.output_feature_names),
                "columns": [{"name": s.name, "kind": s.kind, "missing_fraction": s.missing_fraction,
                             "category_count": len(s.categories), "fill": s.fill,
                             "mean": s.mean, "scale": s.scale,
                             "frequency_flags": list(s.frequency_flags)} for s in self.states_]}

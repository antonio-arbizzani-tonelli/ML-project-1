"""Fold-fitted, feature-only imputation for codebook nonresponses.

Only documented unknown/refused codes are imputed in questionnaire columns.
Blank cells remain blank unless a continuous feature is explicitly listed in
the experiment plan. The target label is never passed to this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


_QUESTION_SOURCES = frozenset({"core_survey", "optional_module"})
_ALLOWED_SOURCES = _QUESTION_SOURCES | frozenset({"administrative", "derived"})


def _response_code_groups(processor, name: str, entry: Mapping) -> dict[str, list[float]]:
    """Select codes known to mean an answer was withheld or unknown.

    Combined labels such as "Don't know/Refused/Missing" do not identify whether
    the question was asked. They remain missing instead of being imputed.
    """
    from src.feature_preprocessing import _as_finite_code, _normalise_label

    codes = processor._missing_codes(name, entry)
    ambiguous = set()
    for item in entry.get("special_values", []):
        if not isinstance(item, Mapping):
            continue
        code = _as_finite_code(item.get("code"))
        label = _normalise_label(item.get("label"))
        if code is not None and ("missing" in label or "not asked" in label):
            ambiguous.add(code)
    return {
        reason: [
            code for code, recorded_reason in codes.items()
            if recorded_reason == reason and code not in ambiguous
        ]
        for reason in ("unknown", "refused")
    }


class _AnchorBinner:
    """Represent available predictors as small categorical bins plus missing."""

    def __init__(self, maximum_bins: int) -> None:
        self.maximum_bins = maximum_bins
        self.rules: list[tuple[str, np.ndarray]] = []

    def fit(self, values: np.ndarray) -> "_AnchorBinner":
        self.rules = []
        for column in values.T:
            observed = column[np.isfinite(column)]
            unique = np.unique(observed)
            if unique.size == 0:
                self.rules.append(("empty", unique))
            elif unique.size <= self.maximum_bins:
                self.rules.append(("exact", unique))
            else:
                quantiles = np.linspace(0, 1, self.maximum_bins + 1)[1:-1]
                self.rules.append(("edges", np.unique(np.quantile(observed, quantiles))))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if values.shape[1] != len(self.rules):
            raise ValueError("Anchor columns changed after fitting.")
        result = np.zeros(values.shape, dtype=np.uint8)
        for index, (kind, reference) in enumerate(self.rules):
            column = values[:, index]
            known = np.isfinite(column)
            if kind == "empty" or not np.any(known):
                continue
            positions = np.searchsorted(reference, column[known])
            if kind == "exact":
                clipped = np.minimum(positions, max(reference.size - 1, 0))
                matched = (positions < reference.size) & (
                    reference[clipped] == column[known]
                )
                encoded = np.where(matched, positions + 1, 0)
            else:
                encoded = positions + 1
            result[known, index] = encoded.astype(np.uint8)
        return result


def _categorical_predictions(
    train_bins: np.ndarray,
    observed_values: np.ndarray,
    observed_mask: np.ndarray,
    prediction_bins: np.ndarray,
    smoothing: float,
    batch_size: int,
) -> np.ndarray:
    """Naive Bayes predicts an observed, valid response code for each row."""
    classes, labels = np.unique(observed_values[observed_mask], return_inverse=True)
    counts = np.bincount(labels, minlength=classes.size).astype(np.float64)
    log_prior = np.log((counts + smoothing) / (counts.sum() + smoothing * classes.size))
    tables = []
    for column_index in range(train_bins.shape[1]):
        bins = train_bins[observed_mask, column_index].astype(np.int64)
        bin_count = int(max(np.max(train_bins[:, column_index]), np.max(prediction_bins[:, column_index]))) + 1
        table = np.bincount(labels * bin_count + bins, minlength=classes.size * bin_count)
        table = table.reshape(classes.size, bin_count).astype(np.float64)
        log_table = np.log((table + smoothing) / (counts[:, None] + smoothing * bin_count))
        tables.append(log_table)
    result = np.empty(prediction_bins.shape[0], dtype=np.float32)
    for start in range(0, result.size, batch_size):
        end = min(start + batch_size, result.size)
        scores = np.broadcast_to(log_prior, (end - start, classes.size)).copy()
        for index, table in enumerate(tables):
            scores += table[:, prediction_bins[start:end, index]].T
        result[start:end] = classes[np.argmax(scores, axis=1)]
    return result


def _continuous_predictions(
    train_bins: np.ndarray,
    observed_values: np.ndarray,
    observed_mask: np.ndarray,
    prediction_bins: np.ndarray,
    ridge: float,
    maximum_training_rows: int,
) -> np.ndarray:
    """Regularized regression predicts a continuous value from anchor bins."""
    rows = np.flatnonzero(observed_mask)
    if rows.size > maximum_training_rows:
        # Deterministic thinning keeps the fold result reproducible.
        rows = rows[np.linspace(0, rows.size - 1, maximum_training_rows, dtype=np.int64)]
    design = np.column_stack((np.ones(rows.size), train_bins[rows].astype(np.float64)))
    target = observed_values[rows].astype(np.float64)
    means = np.mean(design[:, 1:], axis=0)
    scales = np.std(design[:, 1:], axis=0)
    scales[scales == 0] = 1
    design[:, 1:] = (design[:, 1:] - means) / scales
    gram = design.T @ design
    penalty = np.eye(gram.shape[0]) * ridge * rows.size
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(gram + penalty, design.T @ target)
    predict_design = np.column_stack(
        (np.ones(prediction_bins.shape[0]), prediction_bins.astype(np.float64))
    )
    predict_design[:, 1:] = (predict_design[:, 1:] - means) / scales
    predictions = predict_design @ coefficients
    lower, upper = np.quantile(target, [0.001, 0.999])
    return np.clip(predictions, lower, upper).astype(np.float32)


def impute_tree_pair(
    raw_train: np.ndarray,
    raw_validation: np.ndarray,
    clean_train: np.ndarray,
    clean_validation: np.ndarray,
    output_names: Sequence[str],
    processor,
    settings: Mapping,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Impute only configured states, fitting every model on training rows.

    Anchor bins are frozen from the original cleaned training matrix. Imputed
    values are never fed to another learned imputer. The optional BMI formula
    uses completed height and weight after their predictions. Observed target
    values stay untouched. This function returns new matrices and never
    mutates the source or baseline cleaned arrays.
    """
    if not isinstance(settings, Mapping):
        raise ValueError("imputation must be an object.")
    allowed_sources = settings.get("nonresponse_source_kinds", ["core_survey", "optional_module"])
    if not isinstance(allowed_sources, list) or not set(allowed_sources) <= _ALLOWED_SOURCES:
        raise ValueError("nonresponse_source_kinds contains an unsupported source kind.")
    nonresponse_names = settings.get("nonresponse_features")
    if nonresponse_names is not None and (
        not isinstance(nonresponse_names, list)
        or not nonresponse_names
        or not all(isinstance(name, str) for name in nonresponse_names)
        or len(nonresponse_names) != len(set(nonresponse_names))
    ):
        raise ValueError("nonresponse_features must list unique feature names.")
    continuous_names = settings.get("continuous_blank_features", [])
    if not isinstance(continuous_names, list) or len(continuous_names) != len(set(continuous_names)):
        raise ValueError("continuous_blank_features must list unique feature names.")
    derive_bmi = settings.get("derive_bmi_from_height_weight", False)
    if type(derive_bmi) is not bool:
        raise ValueError("derive_bmi_from_height_weight must be true or false.")
    if derive_bmi and "_BMI5" in continuous_names:
        raise ValueError("_BMI5 cannot be both predicted and derived from height/weight.")
    anchors = settings.get("anchor_features", [])
    if not isinstance(anchors, list) or len(anchors) != len(set(anchors)) or not anchors:
        raise ValueError("imputation requires unique anchor_features.")
    output_names = list(output_names)
    positions = {name: index for index, name in enumerate(output_names)}
    required = set(anchors) | set(continuous_names)
    if nonresponse_names is not None:
        required.update(nonresponse_names)
    if derive_bmi:
        required.update(("HTM4", "WTKG3", "_BMI5"))
    if required - set(positions):
        raise ValueError("Imputation references a feature excluded by preprocessing.")
    for name in continuous_names:
        if processor.metadata_by_name[name].get("semantic_type") != "continuous":
            raise ValueError(f"{name} is not a continuous feature.")
    if nonresponse_names is not None:
        for name in nonresponse_names:
            entry = processor.metadata_by_name[name]
            if entry.get("source_kind") not in allowed_sources:
                raise ValueError(f"{name} has a source kind excluded from imputation.")
            if not any(_response_code_groups(processor, name, entry).values()):
                raise ValueError(f"{name} has no unambiguous unknown/refused code.")
    if derive_bmi:
        for name in ("HTM4", "WTKG3", "_BMI5"):
            if processor.metadata_by_name[name].get("semantic_type") != "continuous":
                raise ValueError(f"{name} must be continuous to derive BMI.")
    minimum_observed = int(settings.get("minimum_observed", 100))
    fill_values = settings.get("fill_values", True)
    if type(fill_values) is not bool:
        raise ValueError("fill_values must be true or false.")
    maximum_bins = int(settings.get("anchor_bins", 12))
    smoothing = float(settings.get("smoothing", 1.0))
    ridge = float(settings.get("ridge", 0.01))
    maximum_training_rows = int(settings.get("maximum_continuous_training_rows", 100000))
    batch_size = int(settings.get("prediction_batch_size", 20000))
    if min(minimum_observed, maximum_bins, maximum_training_rows, batch_size) < 1 or smoothing <= 0 or ridge <= 0 or maximum_bins > 254:
        raise ValueError("Invalid imputation model settings.")

    train = np.array(clean_train, copy=True)
    validation = np.array(clean_validation, copy=True)
    anchor_columns = np.asarray([positions[name] for name in anchors])
    binner = _AnchorBinner(maximum_bins).fit(clean_train[:, anchor_columns])
    train_bins = binner.transform(clean_train[:, anchor_columns])
    validation_bins = binner.transform(clean_validation[:, anchor_columns])
    flag_train: list[np.ndarray] = []
    flag_validation: list[np.ndarray] = []
    flag_names: list[str] = []
    for name in output_names:
        entry = processor.metadata_by_name[name]
        source_kind = entry.get("source_kind")
        response_codes = _response_code_groups(processor, name, entry)
        reasons = (
            ("unknown", "refused")
            if source_kind in allowed_sources
            and (nonresponse_names is None or name in nonresponse_names)
            else ()
        )
        target_continuous_blank = name in continuous_names
        if (not reasons or not any(response_codes[reason] for reason in reasons)) and not target_continuous_blank:
            continue
        raw_index = processor._name_to_index[name]
        raw_training = raw_train[:, raw_index]
        raw_validating = raw_validation[:, raw_index]
        masks_train = {
            reason: np.isin(raw_training, response_codes[reason])
            for reason in reasons
        }
        masks_validation = {
            reason: np.isin(raw_validating, response_codes[reason])
            for reason in reasons
        }
        if target_continuous_blank:
            masks_train["blank"] = np.isnan(raw_training)
            masks_validation["blank"] = np.isnan(raw_validating)
        for reason in masks_train:
            if not response_codes.get(reason) and reason != "blank":
                continue
            flag_names.append(f"{name}__was_{reason}")
            flag_train.append(masks_train[reason].astype(np.float32))
            flag_validation.append(masks_validation[reason].astype(np.float32))
        training_missing = np.logical_or.reduce(tuple(masks_train.values()))
        validation_missing = np.logical_or.reduce(tuple(masks_validation.values()))
        if not fill_values or (not np.any(training_missing) and not np.any(validation_missing)):
            continue
        column = positions[name]
        observed = np.isfinite(clean_train[:, column]) & ~training_missing
        if int(np.sum(observed)) < minimum_observed:
            continue
        selected_anchors = [index for index, anchor in enumerate(anchors) if anchor != name]
        tr_bins = train_bins[:, selected_anchors]
        va_bins = validation_bins[:, selected_anchors]
        if entry.get("semantic_type") == "continuous":
            predictor = lambda bins: _continuous_predictions(
                tr_bins, clean_train[:, column], observed, bins, ridge,
                maximum_training_rows,
            )
        else:
            predictor = lambda bins: _categorical_predictions(
                tr_bins, clean_train[:, column], observed, bins, smoothing, batch_size
            )
        combined_bins = np.concatenate(
            (tr_bins[training_missing], va_bins[validation_missing]), axis=0
        )
        predictions = predictor(combined_bins)
        training_count = int(np.sum(training_missing))
        train[training_missing, column] = predictions[:training_count]
        validation[validation_missing, column] = predictions[training_count:]
    if derive_bmi:
        height = positions["HTM4"]
        weight = positions["WTKG3"]
        bmi = positions["_BMI5"]
        original_bmi = clean_train[:, bmi]
        observed_bmi = original_bmi[np.isfinite(original_bmi)]
        lower, upper = np.min(observed_bmi), np.max(observed_bmi)
        for matrix, original, flags in (
            (train, clean_train, flag_train),
            (validation, clean_validation, flag_validation),
        ):
            was_missing = ~np.isfinite(original[:, bmi])
            flags.append(was_missing.astype(np.float32))
            if not fill_values:
                continue
            heights = matrix[:, height]
            weights = matrix[:, weight]
            ready = (
                was_missing
                & np.isfinite(heights)
                & np.isfinite(weights)
                & (heights > 0)
                & (weights > 0)
            )
            candidate = np.full(matrix.shape[0], np.nan, dtype=np.float64)
            candidate[ready] = np.round(
                weights[ready] / np.square(heights[ready]), 2
            )
            ready &= (candidate >= lower) & (candidate <= upper)
            matrix[ready, bmi] = candidate[ready]
        flag_names.append("_BMI5__was_blank")
    if flag_names:
        train = np.column_stack((train, *flag_train)).astype(np.float32, copy=False)
        validation = np.column_stack((validation, *flag_validation)).astype(np.float32, copy=False)
    return train, validation, output_names + flag_names

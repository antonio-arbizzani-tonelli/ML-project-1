"""Screen logistic convergence and L2 strength on development-only folds."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from implementations import logistic_regression, reg_logistic_regression
from src.evaluation import (
    average_precision,
    best_f1_threshold,
    binary_log_loss,
    classification_metrics,
    labels_from_scores,
)
from src.experiment_logger import save_experiment
from src.feature_preprocessing import load_feature_metadata
from src.run_preprocessing_cv import (
    _as_binary_labels,
    _designs_for_variant,
    _load_feature_names,
    _resolve,
    _sha256,
    _sigmoid,
    _stratified_folds,
)


def _metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> tuple[dict, dict]:
    counts, metrics = classification_metrics(
        labels, labels_from_scores(probabilities, threshold)
    )
    metrics["log_loss"] = binary_log_loss(labels, probabilities)
    metrics["average_precision"] = average_precision(labels, probabilities)
    return metrics, {
        "true_positives": counts.true_positives,
        "false_positives": counts.false_positives,
        "true_negatives": counts.true_negatives,
        "false_negatives": counts.false_negatives,
    }


def _as_internal_binary_labels(labels: np.ndarray) -> np.ndarray:
    """Validate the runner's internal 0/1 label convention."""

    labels = np.asarray(labels)
    if (
        labels.ndim != 1
        or labels.size == 0
        or not np.all((labels == 0) | (labels == 1))
    ):
        raise ValueError(
            "Internal labels must be a non-empty one-dimensional 0/1 array."
        )
    return labels.astype(np.int8, copy=False)


def _validate_models(models: list[dict]) -> None:
    names = []
    for model in models:
        name = model["name"]
        names.append(name)
        checkpoints = model["checkpoints"]
        if not checkpoints or any(
            type(step) is not int or step <= 0 for step in checkpoints
        ):
            raise ValueError("Model checkpoints must be positive integers.")
        if checkpoints != sorted(set(checkpoints)):
            raise ValueError("Model checkpoints must be strictly increasing.")
        if not np.isfinite(model["gamma"]) or model["gamma"] <= 0:
            raise ValueError("Model gamma must be finite and positive.")
        if not np.isfinite(model["lambda"]) or model["lambda"] < 0:
            raise ValueError("Model lambda must be finite and non-negative.")
    if len(set(names)) != len(names):
        raise ValueError("Model names must be unique.")


def _nested_threshold_settings(suite: dict) -> dict | None:
    """Validate optional settings for nested development-only threshold fitting."""

    settings = suite.get("nested_threshold")
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise ValueError("nested_threshold must be an object when configured.")
    required = {"checkpoints", "inner_fold_count", "inner_seed"}
    missing = required - set(settings)
    if missing:
        raise ValueError(f"nested_threshold is missing: {', '.join(sorted(missing))}")
    checkpoints = settings["checkpoints"]
    if not checkpoints or any(
        type(step) is not int or step <= 0 for step in checkpoints
    ):
        raise ValueError("Nested threshold checkpoints must be positive integers.")
    if checkpoints != sorted(set(checkpoints)):
        raise ValueError("Nested threshold checkpoints must be strictly increasing.")
    if (
        type(settings["inner_fold_count"]) is not int
        or settings["inner_fold_count"] < 2
    ):
        raise ValueError(
            "nested_threshold.inner_fold_count must be an integer of at least two."
        )
    if type(settings["inner_seed"]) is not int:
        raise ValueError("nested_threshold.inner_seed must be an integer.")
    return settings


def _fit_checkpoints(labels: np.ndarray, design: np.ndarray, model: dict):
    """Continue one NumPy logistic fit, yielding weights and loss at checkpoints."""

    weights = np.zeros(design.shape[1], dtype=np.float64)
    previous_step = 0
    for step in model["checkpoints"]:
        update_count = step - previous_step
        if model["lambda"] == 0:
            weights, data_loss = logistic_regression(
                labels, design, weights, update_count, model["gamma"]
            )
        else:
            weights, data_loss = reg_logistic_regression(
                labels, design, model["lambda"], weights, update_count, model["gamma"]
            )
        yield step, weights, float(data_loss)
        previous_step = step


def _objective_gradient_norm(
    labels: np.ndarray, design: np.ndarray, weights: np.ndarray, lambda_: float
) -> float:
    """Return the norm of the fitted logistic objective gradient."""

    probabilities = _sigmoid(design @ weights)
    gradient = design.T @ (probabilities - labels) / labels.size
    if lambda_:
        gradient = gradient + 2.0 * lambda_ * weights
    return float(np.linalg.norm(gradient))


def _add_interactions(
    design_train: np.ndarray,
    design_validation: np.ndarray,
    output_names: list[str],
    interactions: list[list[str]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Add specified products, centering and scaling on training rows only."""

    if not interactions:
        return design_train, design_validation, output_names
    positions = {name: index + 1 for index, name in enumerate(output_names)}
    train_columns = []
    validation_columns = []
    new_names = []
    for pair in interactions:
        if len(pair) != 2 or any(name not in positions for name in pair):
            raise ValueError(f"Interaction refers to an absent output feature: {pair}")
        left, right = (positions[name] for name in pair)
        train_product = (
            design_train[:, left].astype(np.float64) * design_train[:, right]
        )
        validation_product = (
            design_validation[:, left].astype(np.float64) * design_validation[:, right]
        )
        mean = float(np.mean(train_product))
        scale = float(np.std(train_product))
        if scale == 0 or not np.isfinite(scale):
            raise ValueError(
                f"Interaction has zero or non-finite training scale: {pair}"
            )
        train_columns.append(((train_product - mean) / scale).astype(np.float32))
        validation_columns.append(
            ((validation_product - mean) / scale).astype(np.float32)
        )
        new_names.append(f"{pair[0]}__times__{pair[1]}")
    train_result = np.column_stack((design_train, *train_columns))
    validation_result = np.column_stack((design_validation, *validation_columns))
    if not np.all(np.isfinite(train_result)) or not np.all(
        np.isfinite(validation_result)
    ):
        raise ValueError("Interaction transform produced non-finite values.")
    return train_result, validation_result, output_names + new_names


def _threshold_diagnostics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    fold_positions: list[np.ndarray],
) -> dict:
    """Measure threshold stability without selecting on an evaluated fold.

    Each fold receives a threshold selected from the OOF predictions in the
    other folds.  This is a diagnostic on the development partition only; it
    does not replace the final outer validation evaluation.
    """

    labels = _as_internal_binary_labels(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.size != labels.size:
        raise ValueError(
            "OOF probabilities must be a one-dimensional label-length array."
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("OOF probabilities must be finite.")
    if len(fold_positions) < 2:
        raise ValueError("Threshold diagnostics require at least two folds.")

    fold_ids = np.zeros(labels.size, dtype=np.int8)
    for fold, positions in enumerate(fold_positions, start=1):
        positions = np.asarray(positions)
        if positions.ndim != 1 or positions.size == 0:
            raise ValueError("Each threshold-diagnostic fold must contain positions.")
        if np.any(positions < 0) or np.any(positions >= labels.size):
            raise ValueError("Threshold-diagnostic positions are out of bounds.")
        if np.any(fold_ids[positions] != 0):
            raise ValueError("Threshold-diagnostic folds must not overlap.")
        fold_ids[positions] = fold
    if np.any(fold_ids == 0):
        raise ValueError("Threshold-diagnostic folds must cover every OOF row.")

    fold_optima = []
    cross_fit_details = []
    cross_fit_predictions = np.empty(labels.size, dtype=np.int8)
    for fold, positions in enumerate(fold_positions, start=1):
        positions = np.asarray(positions)
        fold_choice = best_f1_threshold(labels[positions], probabilities[positions])
        fold_metrics, fold_counts = _metrics(
            labels[positions], probabilities[positions], fold_choice.threshold
        )
        fold_optima.append(
            {
                "fold": fold,
                "rows": int(positions.size),
                "threshold": fold_choice.threshold,
                "selected_metrics": fold_metrics,
                "selected_confusion_counts": fold_counts,
            }
        )

        selection_mask = fold_ids != fold
        selection_choice = best_f1_threshold(
            labels[selection_mask], probabilities[selection_mask]
        )
        evaluation_metrics, evaluation_counts = _metrics(
            labels[positions], probabilities[positions], selection_choice.threshold
        )
        cross_fit_predictions[positions] = labels_from_scores(
            probabilities[positions], selection_choice.threshold
        )
        cross_fit_details.append(
            {
                "evaluation_fold": fold,
                "selection_folds": [
                    other_fold
                    for other_fold in range(1, len(fold_positions) + 1)
                    if other_fold != fold
                ],
                "selection_rows": int(np.sum(selection_mask)),
                "threshold": selection_choice.threshold,
                "evaluation_metrics": evaluation_metrics,
                "evaluation_confusion_counts": evaluation_counts,
            }
        )

    cross_fit_counts, cross_fit_metrics = classification_metrics(
        labels, cross_fit_predictions
    )
    thresholds = np.asarray(
        [detail["threshold"] for detail in fold_optima], dtype=np.float64
    )
    cross_fit_thresholds = np.asarray(
        [detail["threshold"] for detail in cross_fit_details], dtype=np.float64
    )
    return {
        "protocol": "Per-fold optimum is descriptive; cross-fit thresholds are selected on the other OOF folds and evaluated on the excluded fold.",
        "fold_optima": fold_optima,
        "fold_optimum_threshold_summary": {
            "minimum": float(np.min(thresholds)),
            "maximum": float(np.max(thresholds)),
            "mean": float(np.mean(thresholds)),
            "standard_deviation": float(np.std(thresholds)),
        },
        "cross_fit": {
            "folds": cross_fit_details,
            "threshold_summary": {
                "minimum": float(np.min(cross_fit_thresholds)),
                "maximum": float(np.max(cross_fit_thresholds)),
                "mean": float(np.mean(cross_fit_thresholds)),
                "standard_deviation": float(np.std(cross_fit_thresholds)),
            },
            "pooled_metrics": cross_fit_metrics,
            "pooled_confusion_counts": {
                "true_positives": cross_fit_counts.true_positives,
                "false_positives": cross_fit_counts.false_positives,
                "true_negatives": cross_fit_counts.true_negatives,
                "false_negatives": cross_fit_counts.false_negatives,
            },
        },
    }


def _save_oof_predictions(
    output_dir: Path,
    experiment_name: str,
    development_indices: np.ndarray,
    fold_positions: list[np.ndarray],
    labels: np.ndarray,
    probabilities: np.ndarray,
    fold_weights: list[np.ndarray],
    output_names: list[str],
) -> tuple[Path, str]:
    """Persist OOF probabilities, fold weights, and their row/fold mapping."""

    labels = _as_internal_binary_labels(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if development_indices.ndim != 1 or development_indices.size != labels.size:
        raise ValueError("Development indices must align with OOF labels.")
    if probabilities.ndim != 1 or probabilities.size != labels.size:
        raise ValueError("OOF probabilities must align with labels.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("OOF probabilities must be finite.")
    fold_ids = np.zeros(labels.size, dtype=np.int8)
    for fold, positions in enumerate(fold_positions, start=1):
        positions = np.asarray(positions)
        if np.any(fold_ids[positions] != 0):
            raise ValueError("OOF fold positions must not overlap.")
        fold_ids[positions] = fold
    if np.any(fold_ids == 0):
        raise ValueError("OOF fold positions must cover every development row.")
    if len(fold_weights) != len(fold_positions):
        raise ValueError("Each OOF fold must have one fitted weight vector.")
    weights = np.asarray(fold_weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != len(output_names) + 1:
        raise ValueError(
            "Fold weights must match the intercept-plus-output feature names."
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("Fold weights must be finite.")

    artifact_dir = output_dir.parent / "oof_predictions"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    slug = "".join(
        character if character.isalnum() else "-"
        for character in experiment_name.lower()
    )
    artifact_path = artifact_dir / f"{timestamp}_{slug}_oof.npz"
    temporary_path = artifact_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_path,
        development_indices=np.asarray(development_indices, dtype=np.int64),
        fold_ids=fold_ids,
        labels=np.asarray(labels, dtype=np.int8),
        probabilities=probabilities,
        fold_weights=weights,
        feature_names=np.asarray(["intercept", *output_names]),
    )
    os.replace(temporary_path, artifact_path)
    return artifact_path, _sha256(artifact_path)


def _nested_threshold_evaluation(
    variant: dict,
    features: np.ndarray,
    development_indices: np.ndarray,
    development_labels: np.ndarray,
    feature_names: list[str],
    metadata: dict,
    model: dict,
    checkpoint: int,
    interactions: list[list[str]],
    outer_fold_positions: list[np.ndarray],
    outer_probabilities: np.ndarray,
    inner_fold_count: int,
    inner_seed: int,
) -> dict:
    """Evaluate OOF scores with thresholds selected wholly within each outer train fold."""

    if inner_fold_count < 2:
        raise ValueError(
            "Nested threshold selection requires at least two inner folds."
        )
    development_labels = _as_internal_binary_labels(development_labels)
    outer_probabilities = np.asarray(outer_probabilities, dtype=np.float64)
    if outer_probabilities.shape != development_labels.shape:
        raise ValueError("Outer OOF probabilities must align with development labels.")
    if not np.all(np.isfinite(outer_probabilities)):
        raise ValueError("Outer OOF probabilities must be finite.")

    position_count = development_labels.size
    outer_fold_ids = np.zeros(position_count, dtype=np.int8)
    for outer_fold, validation_positions in enumerate(outer_fold_positions, start=1):
        validation_positions = np.asarray(validation_positions, dtype=np.int64)
        if np.any(outer_fold_ids[validation_positions] != 0):
            raise ValueError(
                "Outer folds must not overlap for nested threshold selection."
            )
        outer_fold_ids[validation_positions] = outer_fold
    if np.any(outer_fold_ids == 0):
        raise ValueError("Outer folds must cover every development row.")

    nested_predictions = np.empty(position_count, dtype=np.int8)
    details = []
    all_positions = np.arange(position_count, dtype=np.int64)
    for outer_fold, outer_validation_positions in enumerate(
        outer_fold_positions, start=1
    ):
        outer_validation_positions = np.asarray(
            outer_validation_positions, dtype=np.int64
        )
        outer_training_positions = all_positions[outer_fold_ids != outer_fold]
        outer_training_labels = development_labels[outer_training_positions]
        inner_probabilities = np.empty(outer_training_positions.size, dtype=np.float64)
        inner_gradient_norms = []
        for inner_fold, (
            inner_train_positions,
            inner_validation_positions,
        ) in enumerate(
            _stratified_folds(
                outer_training_labels, inner_fold_count, inner_seed + outer_fold
            ),
            start=1,
        ):
            del inner_fold
            train_rows = development_indices[
                outer_training_positions[inner_train_positions]
            ]
            validation_rows = development_indices[
                outer_training_positions[inner_validation_positions]
            ]
            inner_train_features = np.asarray(features[train_rows], dtype=np.float32)
            inner_validation_features = np.asarray(
                features[validation_rows], dtype=np.float32
            )
            inner_design_train, inner_design_validation, inner_output_names = (
                _designs_for_variant(
                    variant,
                    inner_train_features,
                    inner_validation_features,
                    feature_names,
                    metadata,
                )
            )
            inner_design_train, inner_design_validation, _ = _add_interactions(
                inner_design_train,
                inner_design_validation,
                inner_output_names,
                interactions,
            )
            inner_model = dict(model)
            inner_model["checkpoints"] = [checkpoint]
            _, inner_weights, _ = next(
                _fit_checkpoints(
                    outer_training_labels[inner_train_positions],
                    inner_design_train,
                    inner_model,
                )
            )
            inner_probabilities[inner_validation_positions] = _sigmoid(
                inner_design_validation @ inner_weights
            )
            inner_gradient_norms.append(
                _objective_gradient_norm(
                    outer_training_labels[inner_train_positions],
                    inner_design_train,
                    inner_weights,
                    model["lambda"],
                )
            )
            del (
                inner_train_features,
                inner_validation_features,
                inner_design_train,
                inner_design_validation,
                inner_weights,
            )
            gc.collect()

        threshold_choice = best_f1_threshold(outer_training_labels, inner_probabilities)
        evaluation_metrics, evaluation_counts = _metrics(
            development_labels[outer_validation_positions],
            outer_probabilities[outer_validation_positions],
            threshold_choice.threshold,
        )
        nested_predictions[outer_validation_positions] = labels_from_scores(
            outer_probabilities[outer_validation_positions], threshold_choice.threshold
        )
        inner_selection_metrics, _ = _metrics(
            outer_training_labels, inner_probabilities, threshold_choice.threshold
        )
        details.append(
            {
                "outer_fold": outer_fold,
                "outer_training_rows": int(outer_training_positions.size),
                "outer_validation_rows": int(outer_validation_positions.size),
                "inner_fold_count": inner_fold_count,
                "inner_threshold": threshold_choice.threshold,
                "inner_selection_metrics": inner_selection_metrics,
                "inner_training_gradient_norms": inner_gradient_norms,
                "outer_evaluation_metrics": evaluation_metrics,
                "outer_evaluation_confusion_counts": evaluation_counts,
            }
        )

    pooled_counts, pooled_metrics = classification_metrics(
        development_labels, nested_predictions
    )
    pooled_metrics["log_loss"] = binary_log_loss(
        development_labels, outer_probabilities
    )
    pooled_metrics["average_precision"] = average_precision(
        development_labels, outer_probabilities
    )
    thresholds = np.asarray(
        [detail["inner_threshold"] for detail in details], dtype=np.float64
    )
    return {
        "protocol": "For each outer fold, preprocess and train inner folds using only the outer training rows, select the F1 threshold from inner OOF probabilities, and apply it once to the precomputed outer OOF probabilities.",
        "inner_seed": inner_seed,
        "folds": details,
        "threshold_summary": {
            "minimum": float(np.min(thresholds)),
            "maximum": float(np.max(thresholds)),
            "mean": float(np.mean(thresholds)),
            "standard_deviation": float(np.std(thresholds)),
        },
        "pooled_metrics": pooled_metrics,
        "pooled_confusion_counts": {
            "true_positives": pooled_counts.true_positives,
            "false_positives": pooled_counts.false_positives,
            "true_negatives": pooled_counts.true_negatives,
            "false_negatives": pooled_counts.false_negatives,
        },
    }


def run_suite(config_path: Path, project_root: Path, output_dir: Path) -> list[Path]:
    """Run the configured comparison without loading outer validation labels."""

    with config_path.open(encoding="utf-8") as handle:
        suite = json.load(handle)
    with _resolve(project_root, suite["preprocessing_config_path"]).open(
        encoding="utf-8"
    ) as handle:
        preprocessing_suite = json.load(handle)
    selected = set(suite["preprocessing_variants"])
    variants = [
        item
        for item in preprocessing_suite["variants"]
        if item["experiment_name"] in selected
    ]
    if len(variants) != len(selected):
        raise ValueError(
            "A selected preprocessing variant is absent from the source suite."
        )
    models = suite["models"]
    _validate_models(models)
    experiment_prefix = suite.get("experiment_prefix", "phase4")
    if not isinstance(experiment_prefix, str) or not experiment_prefix.strip():
        raise ValueError(
            "experiment_prefix must be a non-empty string when configured."
        )
    nested_settings = _nested_threshold_settings(suite)
    if nested_settings is not None:
        configured_checkpoints = {
            checkpoint for model in models for checkpoint in model["checkpoints"]
        }
        unavailable = set(nested_settings["checkpoints"]) - configured_checkpoints
        if unavailable:
            raise ValueError(
                "Nested threshold checkpoints are not present in models: "
                + ", ".join(str(value) for value in sorted(unavailable))
            )
    data = preprocessing_suite["data"]
    features_path = _resolve(project_root, data["features_path"])
    labels_path = _resolve(project_root, data["labels_path"])
    split_path = _resolve(project_root, data["split_indices_path"])
    metadata_path = _resolve(project_root, preprocessing_suite["feature_metadata_path"])
    header_path = _resolve(project_root, data["x_train_csv_path"])
    features = np.load(features_path, mmap_mode="r")
    labels = _as_binary_labels(np.load(labels_path))
    with np.load(split_path) as split:
        development_indices = np.asarray(split["development_indices"])
    if features.ndim != 2 or features.shape[0] != labels.size:
        raise ValueError("Feature and label caches must have matching row counts.")
    feature_names = _load_feature_names(header_path)
    if features.shape[1] != len(feature_names):
        raise ValueError("Feature header and cache have different column counts.")
    metadata = load_feature_metadata(metadata_path)
    development_labels = labels[development_indices]
    fold_count = int(suite["fold_count"])
    seed = int(suite["seed"])
    fixed_threshold = float(suite["fixed_threshold"])
    interactions = suite.get("interactions", [])
    if not np.isfinite(fixed_threshold):
        raise ValueError("Fixed threshold must be finite.")
    fingerprints = {
        "model_suite_config": _sha256(config_path),
        "features": _sha256(features_path),
        "labels": _sha256(labels_path),
        "split_indices": _sha256(split_path),
        "feature_metadata": _sha256(metadata_path),
        "preprocessing_config": _sha256(
            _resolve(project_root, suite["preprocessing_config_path"])
        ),
    }
    records = []
    for variant in variants:
        variant_interactions = variant.get("interactions", interactions)
        if not isinstance(variant_interactions, list):
            raise ValueError("Variant interactions must be a list when configured.")
        predictions = {
            (model["name"], step): np.empty(development_indices.size, dtype=np.float64)
            for model in models
            for step in model["checkpoints"]
        }
        fold_details = {key: [] for key in predictions}
        fold_weights = {key: [] for key in predictions}
        fold_positions = []
        started = time.perf_counter()
        max_memory_bytes = 0
        print(f"Preprocessing {variant['experiment_name']}", flush=True)
        for fold, (train_positions, validation_positions) in enumerate(
            _stratified_folds(development_labels, fold_count, seed), start=1
        ):
            fold_positions.append(
                np.asarray(validation_positions, dtype=np.int64).copy()
            )
            train_features = np.asarray(
                features[development_indices[train_positions]], dtype=np.float32
            )
            validation_features = np.asarray(
                features[development_indices[validation_positions]], dtype=np.float32
            )
            design_train, design_validation, output_names = _designs_for_variant(
                variant, train_features, validation_features, feature_names, metadata
            )
            design_train, design_validation, output_names = _add_interactions(
                design_train, design_validation, output_names, variant_interactions
            )
            max_memory_bytes = max(
                max_memory_bytes, design_train.nbytes + design_validation.nbytes
            )
            train_labels = development_labels[train_positions]
            validation_labels = development_labels[validation_positions]
            for model in models:
                for step, weights, data_loss in _fit_checkpoints(
                    train_labels, design_train, model
                ):
                    key = (model["name"], step)
                    fold_probabilities = _sigmoid(design_validation @ weights)
                    predictions[key][validation_positions] = fold_probabilities
                    fold_metrics, _ = _metrics(
                        validation_labels, fold_probabilities, fixed_threshold
                    )
                    gradient_norm = _objective_gradient_norm(
                        train_labels, design_train, weights, model["lambda"]
                    )
                    fold_weights[key].append(weights.copy())
                    fold_details[key].append(
                        {
                            "fold": fold,
                            "training_rows": int(train_positions.size),
                            "validation_rows": int(validation_positions.size),
                            "training_data_loss": data_loss,
                            "training_penalized_objective": data_loss
                            + model["lambda"] * float(weights @ weights),
                            "training_objective_gradient_norm": gradient_norm,
                            "weight_l2_norm": float(np.linalg.norm(weights)),
                            "fixed_threshold_metrics": fold_metrics,
                        }
                    )
                    print(
                        f"  fold={fold} model={model['name']} updates={step} "
                        f"train_loss={data_loss:.5f} val_log_loss={fold_metrics['log_loss']:.5f}",
                        flush=True,
                    )
            del train_features, validation_features, design_train, design_validation
            gc.collect()
        variant_runtime = time.perf_counter() - started
        for model in models:
            for step in model["checkpoints"]:
                key = (model["name"], step)
                scores = predictions[key]
                choice = best_f1_threshold(development_labels, scores)
                tuned_metrics, tuned_counts = _metrics(
                    development_labels, scores, choice.threshold
                )
                fixed_metrics, fixed_counts = _metrics(
                    development_labels, scores, fixed_threshold
                )
                experiment_name = f"{experiment_prefix}_{variant['experiment_name'].replace('phase3_', '').replace('_cv', '')}_{model['name']}_{step}"
                threshold_diagnostics = _threshold_diagnostics(
                    development_labels, scores, fold_positions
                )
                nested_threshold_evaluation = None
                if (
                    nested_settings is not None
                    and step in nested_settings["checkpoints"]
                ):
                    nested_threshold_evaluation = _nested_threshold_evaluation(
                        variant,
                        features,
                        development_indices,
                        development_labels,
                        feature_names,
                        metadata,
                        model,
                        step,
                        variant_interactions,
                        fold_positions,
                        scores,
                        nested_settings["inner_fold_count"],
                        nested_settings["inner_seed"],
                    )
                oof_artifact, oof_artifact_sha256 = _save_oof_predictions(
                    output_dir,
                    experiment_name,
                    development_indices,
                    fold_positions,
                    development_labels,
                    scores,
                    fold_weights[key],
                    output_names,
                )
                config = {
                    "experiment_name": experiment_name,
                    "hypothesis": "Test logistic convergence, L2 strength, and the development-only F1 threshold on a fixed preprocessing representation.",
                    "data": {
                        "dataset_id": "project1_processed_training_arrays",
                        "features_path": data["features_path"],
                        "labels_path": data["labels_path"],
                    },
                    "features": {
                        "preprocessing_source": suite["preprocessing_config_path"],
                        "preprocessing_variant": variant["experiment_name"],
                        "preprocessing": variant["preprocessing"],
                        "interactions": variant_interactions,
                        "output_feature_count": len(output_names),
                    },
                    "missing_values": variant["missing_values"],
                    "split": {
                        "id": suite["split_id"],
                        "outer_partition": "saved development indices only",
                        "fold_count": fold_count,
                        "fold_seed": seed,
                        "development_rows": int(development_indices.size),
                    },
                    "model": {
                        "name": (
                            "regularized_logistic_classifier"
                            if model["lambda"]
                            else "logistic_classifier"
                        ),
                        "parameters": {
                            "lambda": model["lambda"],
                            "gamma": model["gamma"],
                            "max_iters": step,
                            "initial_weights": "zero; warm-started between checkpoints within each fold",
                        },
                    },
                    "threshold": {
                        "value": choice.threshold,
                        "selection_method": "max pooled development OOF F1; accuracy breaks exact F1 ties",
                        "fixed_reference": fixed_threshold,
                        "stability_diagnostic": "leave-one-fold-out OOF threshold selection",
                        "nested_selection_method": (
                            "inner OOF F1 threshold within each outer training fold"
                            if nested_threshold_evaluation is not None
                            else None
                        ),
                    },
                    "seed": seed,
                }
                outcome = {
                    "status": "completed",
                    "evaluation_partition": "development_only_3_fold_oof",
                    "metrics": tuned_metrics,
                    "confusion_counts": tuned_counts,
                    "fixed_threshold_metrics": fixed_metrics,
                    "fixed_threshold_confusion_counts": fixed_counts,
                    "folds": fold_details[key],
                    "threshold_diagnostics": threshold_diagnostics,
                    "nested_threshold_evaluation": nested_threshold_evaluation,
                    "oof_predictions_artifact": {
                        "path": str(oof_artifact.relative_to(project_root)),
                        "sha256": oof_artifact_sha256,
                        "arrays": [
                            "development_indices",
                            "fold_ids",
                            "labels",
                            "probabilities",
                            "fold_weights",
                            "feature_names",
                        ],
                    },
                    "selection_note": "The pooled OOF threshold selects and describes the same predictions, so its F1 is exploratory. The cross-fit diagnostic selects each fold's threshold on the other OOF folds. The saved outer validation partition remains untouched.",
                    "runtime_seconds": variant_runtime,
                    "runtime_basis": "Elapsed time for the complete preprocessing-variant suite, shared by its model records; not individual fit time.",
                    "memory_estimate": {
                        "arrays_bytes": max_memory_bytes,
                        "arrays_mebibytes": max_memory_bytes / 1024**2,
                        "basis": "Largest retained float32 transformed train/validation matrix pair; excludes float64 optimizer copy and temporary arrays.",
                    },
                    "input_sha256": fingerprints,
                }
                records.append(save_experiment(config, outcome, output_dir))
        del predictions
        gc.collect()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("results/experiments"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    )
    for record in run_suite(args.config, root, output_dir):
        with record.open(encoding="utf-8") as handle:
            saved = json.load(handle)
        metrics = saved["outcome"]["metrics"]
        print(
            f"Saved {record.name}: F1={metrics['f1']:.5f}, "
            f"accuracy={metrics['accuracy']:.5f}, "
            f"AP={metrics['average_precision']:.5f}, "
            f"log_loss={metrics['log_loss']:.5f}"
        )


if __name__ == "__main__":
    main()

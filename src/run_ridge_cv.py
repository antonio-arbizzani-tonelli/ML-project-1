"""Compare calibrated ridge classifiers with nested threshold selection.

Ridge regression fits the required signed target convention (-1/+1), so its
raw predictions are not probabilities.  This runner learns a Platt calibrator
from inner out-of-fold ridge scores within every outer training fold.  That
calibrator supplies valid probability-like scores for log loss and the F1
threshold is selected on the same inner predictions before one evaluation on
the outer fold.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from implementations import logistic_regression, ridge_regression
from src.evaluation import (
    average_precision,
    best_f1_threshold,
    binary_log_loss,
    classification_metrics,
    labels_from_scores,
)
from src.experiment_logger import save_experiment
from src.feature_preprocessing import load_feature_metadata
from src.run_model_cv import _add_interactions
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
    """Return all reported metrics and counts at one probability threshold."""

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


def _validate_suite(suite: dict) -> None:
    """Reject incomplete or statistically unusable ridge comparison settings."""

    if not isinstance(suite.get("experiment_prefix", "phase7"), str):
        raise ValueError("experiment_prefix must be a string.")
    lambdas = suite.get("ridge_lambdas")
    if not isinstance(lambdas, list) or not lambdas:
        raise ValueError("ridge_lambdas must be a non-empty list.")
    if any(not np.isfinite(value) or value <= 0 for value in lambdas):
        raise ValueError("ridge_lambdas must contain finite positive values.")
    if len(set(lambdas)) != len(lambdas):
        raise ValueError("ridge_lambdas must not repeat values.")
    nested = suite.get("nested_threshold")
    if not isinstance(nested, dict):
        raise ValueError("nested_threshold must be an object.")
    if (
        type(nested.get("inner_fold_count")) is not int
        or nested["inner_fold_count"] < 2
    ):
        raise ValueError(
            "nested_threshold.inner_fold_count must be an integer of at least two."
        )
    if type(nested.get("inner_seed")) is not int:
        raise ValueError("nested_threshold.inner_seed must be an integer.")
    calibration = suite.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("calibration must be an object.")
    if type(calibration.get("max_iters")) is not int or calibration["max_iters"] <= 0:
        raise ValueError("calibration.max_iters must be a positive integer.")
    if not np.isfinite(calibration.get("gamma", np.nan)) or calibration["gamma"] <= 0:
        raise ValueError("calibration.gamma must be finite and positive.")


def _fit_platt_calibrator(
    scores: np.ndarray, labels: np.ndarray, max_iters: int, gamma: float
) -> dict:
    """Fit a logistic calibration layer on inner out-of-fold ridge scores."""

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if (
        scores.ndim != 1
        or scores.size != labels.size
        or not np.all(np.isfinite(scores))
    ):
        raise ValueError("Calibration scores must be finite and align with labels.")
    mean = float(np.mean(scores))
    scale = float(np.std(scores))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    design = np.column_stack((np.ones(scores.size), (scores - mean) / scale))
    weights, loss = logistic_regression(
        labels, design, np.zeros(2, dtype=np.float64), max_iters, gamma
    )
    return {
        "score_mean": mean,
        "score_scale": scale,
        "weights": weights,
        "training_log_loss": float(loss),
    }


def _apply_platt_calibrator(scores: np.ndarray, calibrator: dict) -> np.ndarray:
    """Map ridge scores to positive-class probabilities using frozen parameters."""

    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("Ridge scores must be finite one-dimensional values.")
    weights = np.asarray(calibrator["weights"], dtype=np.float64)
    if weights.shape != (2,) or not np.all(np.isfinite(weights)):
        raise ValueError(
            "Calibration weights must contain an intercept and score coefficient."
        )
    scale = float(calibrator["score_scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Calibration score scale must be finite and positive.")
    standardized_scores = (scores - float(calibrator["score_mean"])) / scale
    return _sigmoid(weights[0] + weights[1] * standardized_scores)


def _ridge_weights(
    labels: np.ndarray, design: np.ndarray, lambda_: float
) -> tuple[np.ndarray, float]:
    """Fit ridge to signed labels while the evaluator remains in its 0/1 convention."""

    signed_labels = 2.0 * np.asarray(labels, dtype=np.float64) - 1.0
    return ridge_regression(signed_labels, design, lambda_)


def _save_ridge_oof_predictions(
    output_dir: Path,
    experiment_name: str,
    development_indices: np.ndarray,
    fold_positions: list[np.ndarray],
    labels: np.ndarray,
    raw_scores: np.ndarray,
    probabilities: np.ndarray,
    applied_thresholds: np.ndarray,
    fold_weights: list[np.ndarray],
    output_names: list[str],
) -> tuple[Path, str]:
    """Persist ridge scores, calibrated probabilities, thresholds, and row mapping."""

    fold_ids = np.zeros(labels.size, dtype=np.int8)
    for fold, positions in enumerate(fold_positions, start=1):
        fold_ids[np.asarray(positions, dtype=np.int64)] = fold
    if (
        np.any(fold_ids == 0)
        or not np.all(np.isfinite(raw_scores))
        or not np.all(np.isfinite(probabilities))
    ):
        raise ValueError("OOF artifact inputs must cover all rows with finite scores.")
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
        raw_scores=np.asarray(raw_scores, dtype=np.float64),
        probabilities=np.asarray(probabilities, dtype=np.float64),
        applied_thresholds=np.asarray(applied_thresholds, dtype=np.float64),
        fold_weights=np.asarray(fold_weights, dtype=np.float64),
        feature_names=np.asarray(["intercept", *output_names]),
    )
    os.replace(temporary_path, artifact_path)
    return artifact_path, _sha256(artifact_path)


def _evaluate_lambda_nested(
    variant: dict,
    features: np.ndarray,
    development_indices: np.ndarray,
    development_labels: np.ndarray,
    feature_names: list[str],
    metadata: list[dict],
    fold_count: int,
    seed: int,
    interactions: list[list[str]],
    lambda_: float,
    calibration: dict,
    inner_fold_count: int,
    inner_seed: int,
) -> tuple[dict, dict]:
    """Run fully nested ridge score calibration and F1 threshold selection."""

    all_positions = np.arange(development_labels.size, dtype=np.int64)
    outer_folds = [
        np.asarray(validation, dtype=np.int64)
        for _, validation in _stratified_folds(development_labels, fold_count, seed)
    ]
    outer_ids = np.zeros(development_labels.size, dtype=np.int8)
    for fold, positions in enumerate(outer_folds, start=1):
        outer_ids[positions] = fold
    raw_scores = np.empty(development_labels.size, dtype=np.float64)
    probabilities = np.empty(development_labels.size, dtype=np.float64)
    applied_thresholds = np.empty(development_labels.size, dtype=np.float64)
    nested_predictions = np.empty(development_labels.size, dtype=np.int8)
    fold_weights = []
    details = []
    max_memory_bytes = 0
    output_names = None
    started = time.perf_counter()

    for outer_fold, outer_validation_positions in enumerate(outer_folds, start=1):
        outer_training_positions = all_positions[outer_ids != outer_fold]
        outer_training_labels = development_labels[outer_training_positions]
        inner_scores = np.empty(outer_training_positions.size, dtype=np.float64)
        inner_feature_counts = []
        for inner_train_positions, inner_validation_positions in _stratified_folds(
            outer_training_labels, inner_fold_count, inner_seed + outer_fold
        ):
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
            inner_design_train, inner_design_validation, inner_names = (
                _designs_for_variant(
                    variant,
                    inner_train_features,
                    inner_validation_features,
                    feature_names,
                    metadata,
                )
            )
            inner_design_train, inner_design_validation, inner_names = (
                _add_interactions(
                    inner_design_train,
                    inner_design_validation,
                    inner_names,
                    interactions,
                )
            )
            inner_weights, _ = _ridge_weights(
                outer_training_labels[inner_train_positions],
                inner_design_train,
                lambda_,
            )
            inner_scores[inner_validation_positions] = (
                inner_design_validation @ inner_weights
            )
            inner_feature_counts.append(len(inner_names))
            max_memory_bytes = max(
                max_memory_bytes,
                inner_design_train.nbytes + inner_design_validation.nbytes,
            )
            del (
                inner_train_features,
                inner_validation_features,
                inner_design_train,
                inner_design_validation,
            )
            del inner_weights
            gc.collect()

        calibrator = _fit_platt_calibrator(
            inner_scores,
            outer_training_labels,
            calibration["max_iters"],
            calibration["gamma"],
        )
        inner_probabilities = _apply_platt_calibrator(inner_scores, calibrator)
        threshold_choice = best_f1_threshold(outer_training_labels, inner_probabilities)

        outer_train_rows = development_indices[outer_training_positions]
        outer_validation_rows = development_indices[outer_validation_positions]
        outer_train_features = np.asarray(features[outer_train_rows], dtype=np.float32)
        outer_validation_features = np.asarray(
            features[outer_validation_rows], dtype=np.float32
        )
        outer_design_train, outer_design_validation, outer_names = _designs_for_variant(
            variant,
            outer_train_features,
            outer_validation_features,
            feature_names,
            metadata,
        )
        outer_design_train, outer_design_validation, outer_names = _add_interactions(
            outer_design_train, outer_design_validation, outer_names, interactions
        )
        if output_names is None:
            output_names = outer_names
        elif output_names != outer_names:
            raise ValueError(
                "Fold-specific feature names prevent a comparable ridge artifact."
            )
        outer_weights, outer_mse = _ridge_weights(
            outer_training_labels, outer_design_train, lambda_
        )
        outer_raw_scores = outer_design_validation @ outer_weights
        outer_probabilities = _apply_platt_calibrator(outer_raw_scores, calibrator)
        evaluation_metrics, evaluation_counts = _metrics(
            development_labels[outer_validation_positions],
            outer_probabilities,
            threshold_choice.threshold,
        )
        raw_scores[outer_validation_positions] = outer_raw_scores
        probabilities[outer_validation_positions] = outer_probabilities
        applied_thresholds[outer_validation_positions] = threshold_choice.threshold
        nested_predictions[outer_validation_positions] = labels_from_scores(
            outer_probabilities, threshold_choice.threshold
        )
        fold_weights.append(outer_weights.copy())
        max_memory_bytes = max(
            max_memory_bytes, outer_design_train.nbytes + outer_design_validation.nbytes
        )
        details.append(
            {
                "outer_fold": outer_fold,
                "outer_training_rows": int(outer_training_positions.size),
                "outer_validation_rows": int(outer_validation_positions.size),
                "inner_fold_count": inner_fold_count,
                "inner_output_feature_counts": inner_feature_counts,
                "inner_threshold": threshold_choice.threshold,
                "inner_selection_metrics": _metrics(
                    outer_training_labels,
                    inner_probabilities,
                    threshold_choice.threshold,
                )[0],
                "calibrator": {
                    "score_mean": calibrator["score_mean"],
                    "score_scale": calibrator["score_scale"],
                    "weights": calibrator["weights"].tolist(),
                    "inner_oof_log_loss": calibrator["training_log_loss"],
                },
                "outer_ridge_mse": float(outer_mse),
                "outer_weight_l2_norm": float(np.linalg.norm(outer_weights)),
                "outer_evaluation_metrics": evaluation_metrics,
                "outer_evaluation_confusion_counts": evaluation_counts,
            }
        )
        del (
            outer_train_features,
            outer_validation_features,
            outer_design_train,
            outer_design_validation,
        )
        del outer_weights
        gc.collect()

    pooled_counts, pooled_metrics = classification_metrics(
        development_labels, nested_predictions
    )
    pooled_metrics["log_loss"] = binary_log_loss(development_labels, probabilities)
    pooled_metrics["average_precision"] = average_precision(
        development_labels, probabilities
    )
    thresholds = np.asarray([detail["inner_threshold"] for detail in details])
    return {
        "status": "completed",
        "evaluation_partition": "development_only_3_fold_nested_cv",
        "metrics": pooled_metrics,
        "confusion_counts": {
            "true_positives": pooled_counts.true_positives,
            "false_positives": pooled_counts.false_positives,
            "true_negatives": pooled_counts.true_negatives,
            "false_negatives": pooled_counts.false_negatives,
        },
        "nested_threshold_evaluation": {
            "protocol": "For each outer fold, fit preprocessing and ridge on inner training rows, calibrate inner OOF ridge scores, select an F1 threshold from those calibrated scores, then apply the frozen calibrator and threshold once to the outer fold.",
            "inner_seed": inner_seed,
            "folds": details,
            "threshold_summary": {
                "minimum": float(np.min(thresholds)),
                "maximum": float(np.max(thresholds)),
                "mean": float(np.mean(thresholds)),
                "standard_deviation": float(np.std(thresholds)),
            },
        },
        "runtime_seconds": time.perf_counter() - started,
        "memory_estimate": {
            "arrays_bytes": int(max_memory_bytes),
            "arrays_mebibytes": max_memory_bytes / 1024**2,
            "basis": "Largest retained float32 transformed train/validation matrix pair; excludes raw source arrays, float64 normal-equation matrices, and temporary solver allocations.",
        },
    }, {
        "raw_scores": raw_scores,
        "probabilities": probabilities,
        "applied_thresholds": applied_thresholds,
        "fold_positions": outer_folds,
        "fold_weights": fold_weights,
        "output_names": output_names,
    }


def run_suite(
    config_path: Path,
    project_root: Path,
    output_dir: Path,
    selected_lambdas: set[float] | None = None,
) -> list[Path]:
    """Run every positive ridge strength on the shared nested protocol."""

    with config_path.open(encoding="utf-8") as handle:
        suite = json.load(handle)
    _validate_suite(suite)
    preprocessing_config_path = _resolve(
        project_root, suite["preprocessing_config_path"]
    )
    with preprocessing_config_path.open(encoding="utf-8") as handle:
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
    if len(variants) != 1:
        raise ValueError(
            "This ridge comparison requires exactly one shared preprocessing variant."
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
    feature_names = _load_feature_names(header_path)
    if (
        features.ndim != 2
        or features.shape[0] != labels.size
        or features.shape[1] != len(feature_names)
    ):
        raise ValueError(
            "Feature arrays, labels, and CSV header must have matching dimensions."
        )
    metadata = load_feature_metadata(metadata_path)
    development_labels = labels[development_indices]
    fingerprints = {
        "ridge_suite_config": _sha256(config_path),
        "preprocessing_config": _sha256(preprocessing_config_path),
        "features": _sha256(features_path),
        "labels": _sha256(labels_path),
        "split_indices": _sha256(split_path),
        "feature_metadata": _sha256(metadata_path),
    }
    configured_lambdas = set(suite["ridge_lambdas"])
    if selected_lambdas is not None:
        unknown = selected_lambdas - configured_lambdas
        if unknown:
            raise ValueError(f"Requested unknown ridge lambdas: {sorted(unknown)}")
        ridge_lambdas = [
            value for value in suite["ridge_lambdas"] if value in selected_lambdas
        ]
    else:
        ridge_lambdas = suite["ridge_lambdas"]
    records = []
    for lambda_ in ridge_lambdas:
        print(f"Ridge lambda={lambda_}", flush=True)
        outcome, artifact_values = _evaluate_lambda_nested(
            variants[0],
            features,
            development_indices,
            development_labels,
            feature_names,
            metadata,
            int(suite["fold_count"]),
            int(suite["seed"]),
            suite.get("interactions", []),
            float(lambda_),
            suite["calibration"],
            suite["nested_threshold"]["inner_fold_count"],
            suite["nested_threshold"]["inner_seed"],
        )
        experiment_name = (
            f"{suite.get('experiment_prefix', 'phase7')}_"
            f"{variants[0]['experiment_name'].replace('phase3_', '').replace('_cv', '')}_"
            f"ridge_lambda_{str(lambda_).replace('.', 'p')}"
        )
        artifact_path, artifact_sha256 = _save_ridge_oof_predictions(
            output_dir,
            experiment_name,
            development_indices,
            artifact_values["fold_positions"],
            development_labels,
            artifact_values["raw_scores"],
            artifact_values["probabilities"],
            artifact_values["applied_thresholds"],
            artifact_values["fold_weights"],
            artifact_values["output_names"],
        )
        outcome["oof_predictions_artifact"] = {
            "path": str(artifact_path.relative_to(project_root)),
            "sha256": artifact_sha256,
            "arrays": [
                "development_indices",
                "fold_ids",
                "labels",
                "raw_scores",
                "probabilities",
                "applied_thresholds",
                "fold_weights",
                "feature_names",
            ],
        }
        outcome["input_sha256"] = fingerprints
        config = {
            "experiment_name": experiment_name,
            "hypothesis": "A calibrated ridge classifier on corrected P2 plus the established age interactions may rival logistic regression under the same nested F1-threshold protocol.",
            "data": {
                "dataset_id": "project1_processed_training_arrays",
                "features_path": data["features_path"],
                "labels_path": data["labels_path"],
            },
            "features": {
                "preprocessing_source": suite["preprocessing_config_path"],
                "preprocessing_variant": variants[0]["experiment_name"],
                "preprocessing": variants[0]["preprocessing"],
                "interactions": suite.get("interactions", []),
                "output_feature_count": len(artifact_values["output_names"]),
            },
            "missing_values": variants[0]["missing_values"],
            "split": {
                "id": suite["split_id"],
                "outer_partition": "saved development indices only",
                "fold_count": int(suite["fold_count"]),
                "fold_seed": int(suite["seed"]),
                "development_rows": int(development_indices.size),
                "nested_inner_fold_count": suite["nested_threshold"][
                    "inner_fold_count"
                ],
                "nested_inner_seed": suite["nested_threshold"]["inner_seed"],
            },
            "model": {
                "name": "ridge_classifier_with_inner_oof_platt_calibration",
                "parameters": {
                    "lambda": float(lambda_),
                    "ridge_target": "-1/+1",
                    "calibration": suite["calibration"],
                },
            },
            "threshold": {
                "selection_method": "max inner calibrated-OOF F1; accuracy breaks exact F1 ties; one frozen threshold per outer fold",
                "value": None,
            },
            "seed": int(suite["seed"]),
        }
        records.append(save_experiment(config, outcome, output_dir))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("results/experiments"))
    parser.add_argument(
        "--lambdas", nargs="+", type=float, help="Optional configured ridge strengths."
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    )
    selected_lambdas = None if args.lambdas is None else set(args.lambdas)
    for record in run_suite(args.config, root, output_dir, selected_lambdas):
        with record.open(encoding="utf-8") as handle:
            saved = json.load(handle)
        metrics = saved["outcome"]["metrics"]
        print(
            f"Saved {record.name}: F1={metrics['f1']:.5f}, accuracy={metrics['accuracy']:.5f}, "
            f"precision={metrics['precision']:.5f}, recall={metrics['recall']:.5f}, "
            f"AP={metrics['average_precision']:.5f}, log_loss={metrics['log_loss']:.5f}"
        )


if __name__ == "__main__":
    main()

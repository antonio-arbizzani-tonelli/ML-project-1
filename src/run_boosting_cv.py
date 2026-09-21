"""Run reproducible development-only CV for the NumPy histogram booster.

Each outer-fold fit reaches its largest configured checkpoint once.  Scores at
earlier checkpoints are recovered from the same fitted trees, and the runner
records the time spent on every tree.  Optional nested threshold fitting uses
only the corresponding outer training rows.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from src.evaluation import (
    average_precision,
    best_f1_threshold,
    binary_log_loss,
    classification_metrics,
    labels_from_scores,
)
from src.experiment_logger import save_experiment
from src.feature_preprocessing import load_feature_metadata, tree_feature_matrices
from src.numpy_boosting import HistogramGradientBoostingClassifier
from src.run_preprocessing_cv import (
    _as_binary_labels,
    _load_feature_names,
    _resolve,
    _sha256,
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


def _validate_checkpoints(checkpoints: object, name: str) -> list[int]:
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError(f"{name} must be a non-empty list.")
    if any(type(step) is not int or step <= 0 for step in checkpoints):
        raise ValueError(f"{name} must contain positive integers.")
    if checkpoints != sorted(set(checkpoints)):
        raise ValueError(f"{name} must be strictly increasing.")
    return list(checkpoints)


def _validate_suite(suite: dict) -> None:
    required = {
        "data",
        "feature_metadata_path",
        "preprocessing_config_path",
        "preprocessing_variant",
        "split_id",
        "fold_count",
        "seed",
        "fixed_threshold",
        "models",
    }
    absent = required - set(suite)
    if absent:
        raise ValueError(f"Boosting suite is missing: {', '.join(sorted(absent))}")
    if type(suite["fold_count"]) is not int or suite["fold_count"] < 2:
        raise ValueError("fold_count must be an integer of at least two.")
    if type(suite["seed"]) is not int or not np.isfinite(suite["fixed_threshold"]):
        raise ValueError("seed must be an integer and fixed_threshold must be finite.")
    if not isinstance(suite["models"], list) or not suite["models"]:
        raise ValueError("models must be a non-empty list.")
    names = []
    for model in suite["models"]:
        if not isinstance(model, dict) or not isinstance(model.get("name"), str):
            raise ValueError("Each model requires a name.")
        if not isinstance(model.get("parameters"), dict):
            raise ValueError("Each model requires parameters.")
        _validate_checkpoints(model.get("checkpoints"), f"{model['name']}.checkpoints")
        names.append(model["name"])
    if len(names) != len(set(names)):
        raise ValueError("Model names must be unique.")
    nested = suite.get("nested_threshold")
    if nested is not None:
        if not isinstance(nested, dict):
            raise ValueError("nested_threshold must be an object.")
        if (
            type(nested.get("inner_fold_count")) is not int
            or nested["inner_fold_count"] < 2
        ):
            raise ValueError("nested_threshold.inner_fold_count must be at least two.")
        if type(nested.get("inner_seed")) is not int:
            raise ValueError("nested_threshold.inner_seed must be an integer.")
        selected = nested.get("model_checkpoints")
        if not isinstance(selected, list) or not selected:
            raise ValueError(
                "nested_threshold.model_checkpoints must be a non-empty list."
            )
        available = {
            (model["name"], step)
            for model in suite["models"]
            for step in model["checkpoints"]
        }
        requested = set()
        requested_models = set()
        for item in selected:
            if not isinstance(item, dict) or not isinstance(item.get("model"), str):
                raise ValueError("Each nested model checkpoint requires a model name.")
            if item["model"] in requested_models:
                raise ValueError("A nested model may appear only once.")
            requested_models.add(item["model"])
            for step in _validate_checkpoints(
                item.get("checkpoints"), "nested checkpoint list"
            ):
                requested.add((item["model"], step))
        if not requested <= available:
            raise ValueError(
                "Nested threshold requests an unavailable model checkpoint."
            )


def _variant(suite: dict, name: str) -> dict:
    matches = [item for item in suite["variants"] if item["experiment_name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one preprocessing variant named {name}.")
    return matches[0]


def _tree_profile(seconds: list[float]) -> dict:
    values = np.asarray(seconds, dtype=np.float64)
    return {
        "seconds_per_tree": values.tolist(),
        "total_seconds": float(np.sum(values)),
        "mean_seconds": float(np.mean(values)),
        "maximum_seconds": float(np.max(values)),
    }


def _fit_checkpoint_probabilities(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    parameters: dict,
    checkpoints: list[int],
    checkpoint_callback=None,
) -> tuple[dict[int, np.ndarray], dict]:
    parameters = dict(parameters)
    parameters["n_estimators"] = checkpoints[-1]
    started = time.perf_counter()
    model = HistogramGradientBoostingClassifier(**parameters).fit(
        train_features, train_labels, checkpoints, checkpoint_callback
    )
    probabilities = dict(model.staged_predict_proba(validation_features, checkpoints))
    elapsed = time.perf_counter() - started
    profile = _tree_profile(model.tree_fit_seconds_)
    profile["fit_and_validation_prediction_seconds"] = elapsed
    profile["bin_count_minimum"] = int(np.min(model.binner_.bin_counts_))
    profile["bin_count_maximum"] = int(np.max(model.binner_.bin_counts_))
    del model
    return probabilities, profile


def _array_sha256(values: np.ndarray) -> str:
    """Fingerprint one exact ordered array without copying its full contents."""

    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _save_oof_predictions(
    artifact_dir: Path,
    experiment_name: str,
    development_indices: np.ndarray,
    fold_ids: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    exploratory_threshold: float,
    nested_thresholds: np.ndarray | None,
) -> tuple[Path, str]:
    """Persist one row-level OOF artifact for later error analysis."""

    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"{experiment_name}_oof.npz"
    temporary = path.with_suffix(".tmp.npz")
    arrays = {
        "development_indices": np.asarray(development_indices, dtype=np.int64),
        "fold_ids": np.asarray(fold_ids, dtype=np.int8),
        "labels": np.asarray(labels, dtype=np.int8),
        "probabilities": np.asarray(probabilities, dtype=np.float64),
        "exploratory_threshold": np.asarray([exploratory_threshold], dtype=np.float64),
    }
    if nested_thresholds is not None:
        arrays["nested_thresholds"] = np.asarray(nested_thresholds, dtype=np.float64)
        if arrays["nested_thresholds"].shape != arrays["probabilities"].shape:
            raise ValueError(
                "Nested thresholds must provide one finite value per OOF probability."
            )
        if not np.all(np.isfinite(arrays["nested_thresholds"])):
            raise ValueError("Nested thresholds must be finite.")
        arrays["nested_predictions"] = (
            arrays["probabilities"] >= arrays["nested_thresholds"]
        ).astype(np.int8)
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
    return path, _sha256(path)


def _nested_evaluations(
    selected: dict[str, list[int]],
    models: dict[str, dict],
    variant: dict,
    features: np.ndarray,
    development_indices: np.ndarray,
    development_labels: np.ndarray,
    feature_names: list[str],
    metadata: list[dict],
    outer_folds: list[np.ndarray],
    outer_probabilities: dict[tuple[str, int], np.ndarray],
    inner_fold_count: int,
    inner_seed: int,
) -> tuple[dict[tuple[str, int], dict], dict[tuple[str, int], np.ndarray]]:
    """Select every threshold from inner OOF predictions, never outer labels."""

    all_positions = np.arange(development_labels.size, dtype=np.int64)
    outer_ids = np.zeros(development_labels.size, dtype=np.int8)
    for fold, positions in enumerate(outer_folds, start=1):
        outer_ids[positions] = fold
    results = {}
    applied_thresholds = {}
    for name, checkpoints in selected.items():
        model = models[name]
        nested_predictions = {
            step: np.empty(development_labels.size, dtype=np.int8)
            for step in checkpoints
        }
        model_thresholds = {
            step: np.empty(development_labels.size, dtype=np.float64)
            for step in checkpoints
        }
        details = {step: [] for step in checkpoints}
        for outer_fold, outer_validation_positions in enumerate(outer_folds, start=1):
            outer_training_positions = all_positions[outer_ids != outer_fold]
            outer_training_labels = development_labels[outer_training_positions]
            inner_probabilities = {
                step: np.empty(outer_training_positions.size, dtype=np.float64)
                for step in checkpoints
            }
            for inner_train, inner_validation in _stratified_folds(
                outer_training_labels, inner_fold_count, inner_seed + outer_fold
            ):
                train_rows = development_indices[outer_training_positions[inner_train]]
                validation_rows = development_indices[
                    outer_training_positions[inner_validation]
                ]
                train_source = np.asarray(features[train_rows], dtype=np.float32)
                validation_source = np.asarray(
                    features[validation_rows], dtype=np.float32
                )
                train_matrix, validation_matrix, _ = tree_feature_matrices(
                    train_source,
                    validation_source,
                    feature_names,
                    metadata,
                    variant["preprocessing"],
                )
                probabilities, _ = _fit_checkpoint_probabilities(
                    train_matrix,
                    outer_training_labels[inner_train],
                    validation_matrix,
                    model["parameters"],
                    checkpoints,
                )
                for step in checkpoints:
                    inner_probabilities[step][inner_validation] = probabilities[step]
                del (
                    train_source,
                    validation_source,
                    train_matrix,
                    validation_matrix,
                    probabilities,
                )
                gc.collect()
            for step in checkpoints:
                choice = best_f1_threshold(
                    outer_training_labels, inner_probabilities[step]
                )
                metrics, counts = _metrics(
                    development_labels[outer_validation_positions],
                    outer_probabilities[(name, step)][outer_validation_positions],
                    choice.threshold,
                )
                nested_predictions[step][outer_validation_positions] = (
                    labels_from_scores(
                        outer_probabilities[(name, step)][outer_validation_positions],
                        choice.threshold,
                    )
                )
                model_thresholds[step][outer_validation_positions] = choice.threshold
                details[step].append(
                    {
                        "outer_fold": outer_fold,
                        "inner_threshold": choice.threshold,
                        "inner_selection_metrics": _metrics(
                            outer_training_labels,
                            inner_probabilities[step],
                            choice.threshold,
                        )[0],
                        "outer_evaluation_metrics": metrics,
                        "outer_evaluation_confusion_counts": counts,
                    }
                )
        for step in checkpoints:
            counts, metrics = classification_metrics(
                development_labels, nested_predictions[step]
            )
            metrics["log_loss"] = binary_log_loss(
                development_labels, outer_probabilities[(name, step)]
            )
            metrics["average_precision"] = average_precision(
                development_labels, outer_probabilities[(name, step)]
            )
            thresholds = [item["inner_threshold"] for item in details[step]]
            results[(name, step)] = {
                "protocol": "Each outer-fold threshold is selected from inner OOF probabilities fitted and preprocessed only on that outer fold's training rows.",
                "folds": details[step],
                "threshold_summary": {
                    "minimum": min(thresholds),
                    "maximum": max(thresholds),
                    "mean": float(np.mean(thresholds)),
                },
                "pooled_metrics": metrics,
                "pooled_confusion_counts": {
                    "true_positives": counts.true_positives,
                    "false_positives": counts.false_positives,
                    "true_negatives": counts.true_negatives,
                    "false_negatives": counts.false_negatives,
                },
            }
            applied_thresholds[(name, step)] = model_thresholds[step]
    return results, applied_thresholds


def run_suite(config_path: Path, project_root: Path, output_dir: Path) -> list[Path]:
    """Evaluate all configured booster settings on deterministic development folds."""

    with config_path.open(encoding="utf-8") as handle:
        suite = json.load(handle)
    _validate_suite(suite)
    preprocessing_path = _resolve(project_root, suite["preprocessing_config_path"])
    with preprocessing_path.open(encoding="utf-8") as handle:
        preprocessing_suite = json.load(handle)
    variant = _variant(preprocessing_suite, suite["preprocessing_variant"])
    data = suite["data"]
    features_path = _resolve(project_root, data["features_path"])
    labels_path = _resolve(project_root, data["labels_path"])
    split_path = _resolve(project_root, data["split_indices_path"])
    metadata_path = _resolve(project_root, suite["feature_metadata_path"])
    header_path = _resolve(project_root, data["x_train_csv_path"])
    artifact_root = Path(suite.get("artifact_dir", "results/boosting_artifacts"))
    if not artifact_root.is_absolute():
        artifact_root = project_root / artifact_root
    checkpoint_dir = artifact_root / "model_checkpoints"
    oof_dir = artifact_root / "oof_predictions"
    fingerprints = {
        "boosting_suite_config": _sha256(config_path),
        "preprocessing_config": _sha256(preprocessing_path),
        "features": _sha256(features_path),
        "labels": _sha256(labels_path),
        "split_indices": _sha256(split_path),
        "feature_metadata": _sha256(metadata_path),
    }
    features = np.load(features_path, mmap_mode="r")
    labels = _as_binary_labels(np.load(labels_path))
    with np.load(split_path) as split:
        development_indices = np.asarray(split["development_indices"], dtype=np.int64)
    feature_names = _load_feature_names(header_path)
    metadata = load_feature_metadata(metadata_path)
    if features.shape != (labels.size, len(feature_names)):
        raise ValueError("Feature cache, labels, and header do not align.")
    development_labels = labels[development_indices]
    models = {model["name"]: model for model in suite["models"]}
    outer_folds = [
        np.asarray(validation, dtype=np.int64)
        for _, validation in _stratified_folds(
            development_labels, suite["fold_count"], suite["seed"]
        )
    ]
    predictions = {
        (model["name"], step): np.empty(development_labels.size, dtype=np.float64)
        for model in suite["models"]
        for step in model["checkpoints"]
    }
    fold_details = {key: [] for key in predictions}
    checkpoint_artifacts = {key: [] for key in predictions}
    fold_ids = np.zeros(development_labels.size, dtype=np.int8)
    max_memory_bytes = 0
    started = time.perf_counter()
    for fold, validation_positions in enumerate(outer_folds, start=1):
        fold_ids[validation_positions] = fold
        train_mask = np.ones(development_labels.size, dtype=bool)
        train_mask[validation_positions] = False
        train_positions = np.flatnonzero(train_mask)
        train_source = np.asarray(
            features[development_indices[train_positions]], dtype=np.float32
        )
        validation_source = np.asarray(
            features[development_indices[validation_positions]], dtype=np.float32
        )
        train_matrix, validation_matrix, output_names = tree_feature_matrices(
            train_source,
            validation_source,
            feature_names,
            metadata,
            variant["preprocessing"],
        )
        max_memory_bytes = max(
            max_memory_bytes, train_matrix.nbytes + validation_matrix.nbytes
        )
        for model in suite["models"]:
            checkpoints = _validate_checkpoints(
                model["checkpoints"], f"{model['name']}.checkpoints"
            )
            training_rows = np.asarray(
                development_indices[train_positions], dtype=np.int64
            )

            def save_model_checkpoint(
                step, fitted_model, *, model_name=model["name"], outer_fold=fold
            ):
                key = (model_name, step)
                checkpoint_path = (
                    checkpoint_dir
                    / f"{suite.get('experiment_prefix', 'boosting')}_{model_name}_fold{outer_fold}_{step}trees.pkl"
                )
                fitted_model.save_checkpoint(
                    checkpoint_path,
                    {
                        "artifact_type": "trusted_local_numpy_boosting_checkpoint",
                        "experiment_prefix": suite.get("experiment_prefix", "boosting"),
                        "model_name": model_name,
                        "outer_fold": outer_fold,
                        "fitted_tree_count": step,
                        "training_row_indices": training_rows,
                        "training_row_indices_sha256": _array_sha256(training_rows),
                        "training_labels_sha256": _array_sha256(
                            development_labels[train_positions]
                        ),
                        "input_sha256": fingerprints,
                    },
                )
                checkpoint_artifacts[key].append(
                    {
                        "fold": outer_fold,
                        "path": str(checkpoint_path.relative_to(project_root)),
                        "sha256": _sha256(checkpoint_path),
                        "training_rows": int(training_rows.size),
                    }
                )

            probabilities, profile = _fit_checkpoint_probabilities(
                train_matrix,
                development_labels[train_positions],
                validation_matrix,
                model["parameters"],
                checkpoints,
                save_model_checkpoint,
            )
            for step in checkpoints:
                key = (model["name"], step)
                predictions[key][validation_positions] = probabilities[step]
                metrics, _ = _metrics(
                    development_labels[validation_positions],
                    probabilities[step],
                    suite["fixed_threshold"],
                )
                fold_details[key].append(
                    {
                        "fold": fold,
                        "training_rows": int(train_positions.size),
                        "validation_rows": int(validation_positions.size),
                        "fixed_threshold_metrics": metrics,
                        "tree_fit_profile": profile,
                    }
                )
            print(
                f"fold={fold} model={model['name']} trees={checkpoints[-1]} fit={profile['fit_and_validation_prediction_seconds']:.1f}s",
                flush=True,
            )
            del probabilities
        del train_source, validation_source, train_matrix, validation_matrix
        gc.collect()
    nested = suite.get("nested_threshold")
    nested_results = {}
    nested_applied_thresholds = {}
    if nested is not None:
        selected = {
            item["model"]: _validate_checkpoints(
                item["checkpoints"], "nested checkpoint list"
            )
            for item in nested["model_checkpoints"]
        }
        nested_results, nested_applied_thresholds = _nested_evaluations(
            selected,
            models,
            variant,
            features,
            development_indices,
            development_labels,
            feature_names,
            metadata,
            outer_folds,
            predictions,
            nested["inner_fold_count"],
            nested["inner_seed"],
        )
    runtime = time.perf_counter() - started
    records = []
    for model in suite["models"]:
        for step in model["checkpoints"]:
            key = (model["name"], step)
            choice = best_f1_threshold(development_labels, predictions[key])
            exploratory_metrics, exploratory_counts = _metrics(
                development_labels, predictions[key], choice.threshold
            )
            fixed_metrics, fixed_counts = _metrics(
                development_labels, predictions[key], suite["fixed_threshold"]
            )
            nested_result = nested_results.get(key)
            experiment_name = f"{suite.get('experiment_prefix', 'boosting')}_{model['name']}_{step}trees"
            oof_artifact, oof_sha256 = _save_oof_predictions(
                oof_dir,
                experiment_name,
                development_indices,
                fold_ids,
                development_labels,
                predictions[key],
                choice.threshold,
                nested_applied_thresholds.get(key),
            )
            config = {
                "experiment_name": experiment_name,
                "hypothesis": "Measure depth and tree-count learning curves before selecting one NumPy boosting candidate.",
                "data": {
                    "dataset_id": "project1_processed_training_arrays",
                    "features_path": data["features_path"],
                    "labels_path": data["labels_path"],
                },
                "features": {
                    "preprocessing_source": suite["preprocessing_config_path"],
                    "preprocessing_variant": variant["experiment_name"],
                    "output_feature_count": len(output_names),
                },
                "missing_values": {
                    "method": "Codebook non-responses become NaN; every tree split evaluates both missing-value directions.",
                    "fit_partition": "corresponding training rows only",
                },
                "split": {
                    "id": suite["split_id"],
                    "outer_partition": "saved development indices only",
                    "fold_count": suite["fold_count"],
                    "fold_seed": suite["seed"],
                },
                "model": {
                    "name": "numpy_histogram_gradient_boosting",
                    "parameters": {**model["parameters"], "n_estimators": step},
                },
                "threshold": {
                    "value": choice.threshold,
                    "selection_method": "pooled outer OOF F1; exploratory unless nested_threshold_evaluation is present",
                    "fixed_reference": suite["fixed_threshold"],
                },
                "seed": suite["seed"],
            }
            outcome = {
                "status": "completed",
                "evaluation_partition": "development_only_3_fold_oof",
                "metrics": exploratory_metrics,
                "confusion_counts": exploratory_counts,
                "fixed_threshold_metrics": fixed_metrics,
                "fixed_threshold_confusion_counts": fixed_counts,
                "folds": fold_details[key],
                "nested_threshold_evaluation": nested_result,
                "model_checkpoints": checkpoint_artifacts[key],
                "oof_predictions_artifact": {
                    "path": str(oof_artifact.relative_to(project_root)),
                    "sha256": oof_sha256,
                    "arrays": [
                        "development_indices",
                        "fold_ids",
                        "labels",
                        "probabilities",
                        "exploratory_threshold",
                        "nested_thresholds",
                        "nested_predictions",
                    ],
                },
                "selection_note": "Pooled OOF F1 uses the same labels to select and describe its threshold. Use nested_threshold_evaluation, when configured, for an unbiased threshold result.",
                "runtime_seconds": runtime,
                "runtime_basis": "Elapsed time for the complete configured boosting suite, including optional nested threshold fits.",
                "memory_estimate": {
                    "arrays_bytes": max_memory_bytes,
                    "arrays_mebibytes": max_memory_bytes / 1024**2,
                    "basis": "Largest retained float32 cleaned train/validation pair; excludes temporary histogram arrays and runtime overhead.",
                },
                "input_sha256": fingerprints,
            }
            records.append(save_experiment(config, outcome, output_dir))
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
        print(f"Saved experiment record: {record}")


if __name__ == "__main__":
    main()

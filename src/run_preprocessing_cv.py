"""Compare fold-safe Phase 3 preprocessing variants on development-only CV."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from implementations import logistic_regression
from src.baseline_preprocessing import (
    fit_baseline_preprocessor,
    transform_baseline_features,
)
from src.evaluation import (
    average_precision,
    binary_log_loss,
    classification_metrics,
    labels_from_scores,
)
from src.experiment_logger import save_experiment
from src.feature_preprocessing import FeaturePreprocessor, load_feature_metadata


def _resolve(root: Path, configured_path: str) -> Path:
    """Resolve and require a project-relative experiment input."""

    resolved = root / configured_path
    if not resolved.is_file():
        raise FileNotFoundError(f"Required experiment input does not exist: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    """Hash an input artifact in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_feature_names(x_train_csv: Path) -> list[str]:
    """Read the predictor order without loading the full CSV."""

    with x_train_csv.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    if len(header) < 3 or header[0] != "Id":
        raise ValueError("x_train.csv must start with Id followed by predictors.")
    return header[1:]


def _as_binary_labels(labels: np.ndarray) -> np.ndarray:
    """Map project -1/+1 labels into the internal zero/one convention."""

    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.all((labels == -1) | (labels == 1)):
        raise ValueError(
            "The cached project labels must be one-dimensional -1/+1 values."
        )
    return ((labels + 1) // 2).astype(np.int8)


def _stratified_folds(
    labels: np.ndarray, fold_count: int, seed: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield deterministic training/validation positions for stratified folds."""

    if fold_count < 2:
        raise ValueError("fold_count must be at least two.")
    rng = np.random.default_rng(seed)
    assignments = np.empty(labels.size, dtype=np.int16)
    for class_value in (0, 1):
        positions = np.flatnonzero(labels == class_value)
        rng.shuffle(positions)
        for fold, subset in enumerate(np.array_split(positions, fold_count)):
            assignments[subset] = fold
    for fold in range(fold_count):
        validation = np.flatnonzero(assignments == fold)
        training = np.flatnonzero(assignments != fold)
        yield training, validation


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    """Return finite probabilities from finite linear scores."""

    scores = np.asarray(scores, dtype=np.float64)
    probabilities = np.empty_like(scores)
    non_negative = scores >= 0.0
    probabilities[non_negative] = 1.0 / (1.0 + np.exp(-scores[non_negative]))
    exp_scores = np.exp(scores[~non_negative])
    probabilities[~non_negative] = exp_scores / (1.0 + exp_scores)
    return probabilities


def _raw_baseline_designs(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    feature_names: list[str],
    excluded_features: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fit the original numeric baseline after an explicit column exclusion."""

    excluded = set(excluded_features)
    unknown = excluded - set(feature_names)
    if unknown:
        raise ValueError(
            f"Raw preprocessing excludes unknown features: {', '.join(sorted(unknown))}"
        )
    kept_indices = [
        index for index, name in enumerate(feature_names) if name not in excluded
    ]
    fitted = fit_baseline_preprocessor(train_features[:, kept_indices])
    return (
        transform_baseline_features(train_features[:, kept_indices], fitted),
        transform_baseline_features(validation_features[:, kept_indices], fitted),
        [feature_names[index] for index in kept_indices],
    )


def _designs_for_variant(
    variant: Mapping[str, Any],
    train_features: np.ndarray,
    validation_features: np.ndarray,
    feature_names: list[str],
    metadata: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fit one configured representation and transform its train/validation rows."""

    preprocessing = variant["preprocessing"]
    name = preprocessing["name"]
    if name == "raw_median_scale":
        return _raw_baseline_designs(
            train_features,
            validation_features,
            feature_names,
            list(preprocessing.get("exclude_features", [])),
        )
    if name == "codebook_aware":
        processor = FeaturePreprocessor(feature_names, metadata, preprocessing)
        processor.fit(train_features)
        return (
            processor.transform(train_features),
            processor.transform(validation_features),
            list(processor.output_feature_names),
        )
    raise ValueError(f"Unsupported preprocessing name: {name}")


def _run_variant(
    variant: Mapping[str, Any],
    features: np.ndarray,
    labels: np.ndarray,
    development_indices: np.ndarray,
    feature_names: list[str],
    metadata: list[dict[str, Any]],
    fold_count: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate one preprocessing variant with deterministic CV inside development data."""

    model = variant["model"]
    parameters = model["parameters"]
    max_iters = int(parameters["max_iters"])
    gamma = float(parameters["gamma"])
    threshold = float(variant["threshold"]["value"])
    if max_iters < 0 or gamma <= 0.0 or not np.isfinite(gamma):
        raise ValueError(
            "Logistic parameters must have non-negative iterations and finite positive gamma."
        )
    if not np.isfinite(threshold):
        raise ValueError("The fixed threshold must be finite.")

    development_labels = labels[development_indices]
    probabilities = np.empty(development_indices.size, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    column_counts: list[int] = []
    memory_bytes: list[int] = []
    started = time.perf_counter()
    for fold, (train_positions, validation_positions) in enumerate(
        _stratified_folds(development_labels, fold_count, seed), start=1
    ):
        train_indices = development_indices[train_positions]
        validation_indices = development_indices[validation_positions]
        train_features = np.asarray(features[train_indices], dtype=np.float32)
        validation_features = np.asarray(features[validation_indices], dtype=np.float32)
        train_labels = development_labels[train_positions]
        validation_labels = development_labels[validation_positions]
        design_train, design_validation, output_names = _designs_for_variant(
            variant, train_features, validation_features, feature_names, metadata
        )
        weights, training_loss = logistic_regression(
            train_labels,
            design_train,
            np.zeros(design_train.shape[1], dtype=np.float64),
            max_iters,
            gamma,
        )
        fold_probabilities = _sigmoid(design_validation @ weights)
        fold_predictions = labels_from_scores(fold_probabilities, threshold)
        counts, metrics = classification_metrics(validation_labels, fold_predictions)
        metrics["log_loss"] = binary_log_loss(validation_labels, fold_probabilities)
        metrics["average_precision"] = average_precision(
            validation_labels, fold_probabilities
        )
        probabilities[validation_positions] = fold_probabilities
        column_counts.append(len(output_names))
        memory_bytes.append(design_train.nbytes + design_validation.nbytes)
        fold_rows.append(
            {
                "fold": fold,
                "training_rows": int(train_positions.size),
                "validation_rows": int(validation_positions.size),
                "output_features": int(len(output_names)),
                "training_loss": float(training_loss),
                "metrics": metrics,
                "confusion_counts": {
                    "true_positives": counts.true_positives,
                    "false_positives": counts.false_positives,
                    "true_negatives": counts.true_negatives,
                    "false_negatives": counts.false_negatives,
                },
            }
        )
        del (
            train_features,
            validation_features,
            design_train,
            design_validation,
            weights,
        )
        gc.collect()

    predictions = labels_from_scores(probabilities, threshold)
    counts, metrics = classification_metrics(development_labels, predictions)
    metrics["log_loss"] = binary_log_loss(development_labels, probabilities)
    metrics["average_precision"] = average_precision(development_labels, probabilities)
    elapsed = time.perf_counter() - started
    outcome = {
        "status": "completed",
        "evaluation_partition": "development_only_3_fold_cv",
        "confusion_counts": {
            "true_positives": counts.true_positives,
            "false_positives": counts.false_positives,
            "true_negatives": counts.true_negatives,
            "false_negatives": counts.false_negatives,
        },
        "metrics": metrics,
        "folds": fold_rows,
        "model_details": {
            "score_kind": "positive-class probability",
            "fold_count": fold_count,
            "output_features_per_fold": column_counts,
        },
        "runtime_seconds": elapsed,
        "memory_estimate": {
            "arrays_bytes": int(max(memory_bytes)),
            "arrays_mebibytes": max(memory_bytes) / 1024**2,
            "basis": "Largest retained pair of float32 transformed train/validation matrices; excludes raw source arrays, float64 optimizer copies, temporary gradients, and Python/NumPy runtime overhead.",
        },
    }
    return outcome, {
        "development_rows": int(development_indices.size),
        "fold_seed": seed,
        "fold_count": fold_count,
        "output_feature_names": output_names,
    }


def run_suite(
    config_path: Path,
    project_root: Path,
    output_dir: Path,
    selected_experiments: set[str] | None = None,
) -> list[Path]:
    """Run every configured preprocessing comparison and store one record each."""

    with config_path.open(encoding="utf-8") as handle:
        suite = json.load(handle)
    data = suite["data"]
    features_path = _resolve(project_root, data["features_path"])
    labels_path = _resolve(project_root, data["labels_path"])
    split_path = _resolve(project_root, data["split_indices_path"])
    metadata_path = _resolve(project_root, suite["feature_metadata_path"])
    header_path = _resolve(project_root, data["x_train_csv_path"])
    features = np.load(features_path, mmap_mode="r")
    labels = _as_binary_labels(np.load(labels_path))
    with np.load(split_path) as split:
        development_indices = np.asarray(split["development_indices"])
    if features.ndim != 2 or features.shape[0] != labels.size:
        raise ValueError("Feature and label caches must have matching row counts.")
    feature_names = _load_feature_names(header_path)
    if len(feature_names) != features.shape[1]:
        raise ValueError(
            "CSV header and cached feature matrix disagree on column count."
        )
    metadata = load_feature_metadata(metadata_path)
    fold_count = int(suite["fold_count"])
    seed = int(suite["seed"])
    records: list[Path] = []
    variants = [
        variant
        for variant in suite["variants"]
        if selected_experiments is None
        or variant["experiment_name"] in selected_experiments
    ]
    if selected_experiments is not None:
        configured = {variant["experiment_name"] for variant in suite["variants"]}
        unknown = selected_experiments - configured
        if unknown:
            raise ValueError(
                f"Requested unknown experiments: {', '.join(sorted(unknown))}"
            )
    if not variants:
        raise ValueError("No preprocessing variants were selected.")
    for variant in variants:
        outcome, cv_details = _run_variant(
            variant,
            features,
            labels,
            development_indices,
            feature_names,
            metadata,
            fold_count,
            seed,
        )
        ledger_config = {
            "experiment_name": variant["experiment_name"],
            "hypothesis": variant["hypothesis"],
            "data": {
                "dataset_id": "project1_processed_training_arrays",
                "features_path": data["features_path"],
                "labels_path": data["labels_path"],
                "feature_metadata_path": suite["feature_metadata_path"],
            },
            "features": {
                "preprocessing": variant["preprocessing"],
                "output_feature_count_per_fold": outcome["model_details"][
                    "output_features_per_fold"
                ],
            },
            "missing_values": variant["missing_values"],
            "split": {
                "id": suite["split_id"],
                "outer_partition": "saved development indices only",
                "inner_cv": cv_details,
            },
            "model": variant["model"],
            "threshold": variant["threshold"],
            "seed": seed,
        }
        outcome["input_sha256"] = {
            "features": _sha256(features_path),
            "labels": _sha256(labels_path),
            "split_indices": _sha256(split_path),
            "feature_metadata": _sha256(metadata_path),
        }
        records.append(save_experiment(ledger_config, outcome, output_dir))
    return records


def main() -> None:
    """Run a Phase 3 CV suite from a single JSON configuration file."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", type=Path, help="Path to the preprocessing-suite JSON file."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("results/experiments"))
    parser.add_argument(
        "--experiments",
        nargs="+",
        help="Optional experiment names from the suite; unlisted variants are skipped.",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    )
    selected = None if args.experiments is None else set(args.experiments)
    for record in run_suite(args.config, root, output_dir, selected):
        print(f"Saved experiment record: {record}")


if __name__ == "__main__":
    main()

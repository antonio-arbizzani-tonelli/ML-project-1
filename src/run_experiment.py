"""Run a configured baseline and store a reproducible experiment record."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from implementations import logistic_regression, ridge_regression
from src.baseline_preprocessing import (
    fit_baseline_preprocessor,
    transform_baseline_features,
)
from src.evaluation import binary_log_loss, classification_metrics, labels_from_scores
from src.experiment_logger import save_experiment, validate_config


def _load_config(path: Path) -> dict[str, Any]:
    """Load one JSON configuration and validate its common ledger fields."""

    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def _resolve(project_root: Path, relative_path: str) -> Path:
    """Resolve a configured project-relative file and require that it exists."""

    path = project_root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required experiment input does not exist: {path}")
    return path


def _sha256_file(path: Path) -> str:
    """Hash a small provenance artifact without loading it as text."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _all_negative_predictions(
    labels: np.ndarray, config: Mapping[str, Any]
) -> np.ndarray:
    """Return the reference model's validation predictions."""

    if config["model"].get("name") != "all_negative":
        raise ValueError(
            "This first runner supports model.name='all_negative' only. "
            "Add an explicit runner branch before registering another model."
        )
    return np.zeros(labels.size, dtype=np.int8)


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    """Return finite positive-class probabilities from arbitrary finite scores."""

    scores = np.asarray(scores, dtype=np.float64)
    probabilities = np.empty_like(scores)
    non_negative = scores >= 0.0
    probabilities[non_negative] = 1.0 / (1.0 + np.exp(-scores[non_negative]))
    exponentials = np.exp(scores[~non_negative])
    probabilities[~non_negative] = exponentials / (1.0 + exponentials)
    return probabilities


def _as_internal_labels(labels: np.ndarray) -> tuple[np.ndarray, str]:
    """Convert stored project labels to the 0/1 convention used by evaluation.

    The existing processed cache intentionally preserves the competition's
    ``-1/+1`` labels.  Conversion occurs here, at the model-evaluation boundary,
    rather than changing the cache or letting a metric silently infer a class.
    """

    if np.all((labels == 0) | (labels == 1)):
        return labels.astype(np.int8, copy=False), "stored 0/1 labels"
    if np.all((labels == -1) | (labels == 1)):
        return ((labels + 1) // 2).astype(np.int8), "stored -1/+1 labels mapped to 0/1"
    raise ValueError("The configured labels must contain either 0/1 or -1/+1 only.")


def _load_split_indices(
    indices_path: Path, label_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate non-overlapping saved development and validation indices."""

    with np.load(indices_path) as split:
        development_indices = split["development_indices"]
        validation_indices = split["validation_indices"]
    for name, indices in (
        ("development_indices", development_indices),
        ("validation_indices", validation_indices),
    ):
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError(
                f"The split must contain non-empty one-dimensional {name}."
            )
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"{name} must contain integer row indices.")
        if np.any(indices < 0) or np.any(indices >= label_count):
            raise ValueError(f"{name} contains an out-of-range row index.")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"{name} contains duplicate row indices.")
    if np.intersect1d(development_indices, validation_indices).size:
        raise ValueError("Development and validation indices must not overlap.")
    return development_indices, validation_indices


def _load_common_design_matrices(
    config: Mapping[str, Any],
    project_root: Path,
    development_indices: np.ndarray,
    validation_indices: np.ndarray,
    label_count: int,
) -> tuple[np.ndarray, np.ndarray, Path]:
    """Fit the common baseline feature treatment then transform both partitions."""

    try:
        features_relative_path = config["data"]["features_path"]
    except KeyError as error:
        raise ValueError("A learned baseline requires data.features_path.") from error
    features_path = _resolve(project_root, features_relative_path)
    features = np.load(features_path, mmap_mode="r")
    if features.ndim != 2 or features.shape[0] != label_count:
        raise ValueError("Configured feature rows must match the configured labels.")
    development_features = features[development_indices]
    validation_features = features[validation_indices]
    preprocessor = fit_baseline_preprocessor(development_features)
    return (
        transform_baseline_features(development_features, preprocessor),
        transform_baseline_features(validation_features, preprocessor),
        features_path,
    )


def _fit_and_predict(
    config: Mapping[str, Any],
    development_labels: np.ndarray,
    validation_labels: np.ndarray,
    development_design: np.ndarray,
    validation_design: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Fit one supported learned baseline and return labels, probabilities, details."""

    model = config["model"]
    parameters = model.get("parameters", {})
    model_name = model.get("name")
    threshold = config["threshold"].get("value")
    if model_name == "ridge_classifier":
        if "lambda" not in parameters:
            raise ValueError("ridge_classifier requires model.parameters.lambda.")
        lambda_ = float(parameters["lambda"])
        if lambda_ < 0.0 or not np.isfinite(lambda_):
            raise ValueError("ridge lambda must be a finite non-negative value.")
        signed_labels = 2.0 * development_labels.astype(np.float64) - 1.0
        weights, training_loss = ridge_regression(
            signed_labels, development_design, lambda_
        )
        scores = validation_design @ weights
        predictions = (scores >= float(threshold)).astype(np.int8)
        return (
            predictions,
            None,
            {
                "training_loss": float(training_loss),
                "score_kind": "signed linear ridge score; it is not a probability",
                "weights": int(weights.size),
            },
        )
    if model_name == "logistic_classifier":
        required = ("max_iters", "gamma")
        missing = [key for key in required if key not in parameters]
        if missing:
            raise ValueError(
                "logistic_classifier requires model.parameters." + " and ".join(missing)
            )
        max_iters = int(parameters["max_iters"])
        gamma = float(parameters["gamma"])
        if max_iters < 0 or gamma <= 0.0 or not np.isfinite(gamma):
            raise ValueError(
                "Logistic max_iters must be non-negative and gamma finite positive."
            )
        initial_weights = np.zeros(development_design.shape[1], dtype=np.float64)
        weights, training_loss = logistic_regression(
            development_labels, development_design, initial_weights, max_iters, gamma
        )
        probabilities = _sigmoid(validation_design @ weights)
        predictions = labels_from_scores(probabilities, threshold=float(threshold))
        return (
            predictions,
            probabilities,
            {
                "training_loss": float(training_loss),
                "score_kind": "positive-class probability",
                "weights": int(weights.size),
            },
        )
    raise ValueError(
        "Supported learned baselines are model.name='ridge_classifier' and "
        "model.name='logistic_classifier'."
    )


def run_config(config_path: Path, project_root: Path, output_dir: Path) -> Path:
    """Execute one config on the frozen validation indices and save its record."""

    config = _load_config(config_path)
    labels_path = _resolve(project_root, config["data"]["labels_path"])
    indices_path = _resolve(project_root, config["split"]["indices_path"])
    started = time.perf_counter()
    labels = np.load(labels_path)
    if labels.ndim != 1:
        raise ValueError("The configured label array must be one-dimensional.")
    development_indices, validation_indices = _load_split_indices(
        indices_path, labels.size
    )
    internal_labels, label_conversion = _as_internal_labels(labels)
    development_labels = internal_labels[development_indices]
    validation_labels = internal_labels[validation_indices]
    model_name = config["model"].get("name")
    probabilities = None
    model_details: dict[str, Any] = {}
    features_path = None
    design_memory_bytes = 0
    if model_name == "all_negative":
        predictions = _all_negative_predictions(validation_labels, config)
    else:
        development_design, validation_design, features_path = (
            _load_common_design_matrices(
                config,
                project_root,
                development_indices,
                validation_indices,
                labels.size,
            )
        )
        predictions, probabilities, model_details = _fit_and_predict(
            config,
            development_labels,
            validation_labels,
            development_design,
            validation_design,
        )
        design_memory_bytes = development_design.nbytes + validation_design.nbytes
    counts, metrics = classification_metrics(validation_labels, predictions)
    if probabilities is not None:
        metrics["log_loss"] = binary_log_loss(validation_labels, probabilities)
    elapsed = time.perf_counter() - started
    arrays_bytes = (
        labels.nbytes
        + development_indices.nbytes
        + validation_indices.nbytes
        + development_labels.nbytes
        + validation_labels.nbytes
        + predictions.nbytes
        + design_memory_bytes
        + (0 if probabilities is None else probabilities.nbytes)
    )
    outcome = {
        "status": "completed",
        "evaluation_partition": config["split"].get(
            "evaluation_partition", "validation"
        ),
        "confusion_counts": {
            "true_positives": counts.true_positives,
            "false_positives": counts.false_positives,
            "true_negatives": counts.true_negatives,
            "false_negatives": counts.false_negatives,
        },
        "metrics": metrics,
        "model_details": model_details,
        "runtime_seconds": elapsed,
        "memory_estimate": {
            "arrays_bytes": int(arrays_bytes),
            "arrays_mebibytes": arrays_bytes / 1024**2,
            "basis": "Retained labels, split indices, labels, predictions, probabilities where present, and transformed design matrices; excludes temporary optimizer arrays and Python/NumPy runtime overhead.",
        },
        "input_sha256": {
            "labels": _sha256_file(labels_path),
            "split_indices": _sha256_file(indices_path),
        },
        "label_conversion": label_conversion,
    }
    if features_path is not None:
        outcome["input_sha256"]["features"] = _sha256_file(features_path)
    return save_experiment(config, outcome, output_dir)


def main() -> None:
    """Parse CLI arguments and print the record produced by the run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to an experiment JSON config.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root used for config-relative inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/experiments"),
        help="Directory for JSON records and index.csv.",
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else project_root / args.output_dir
    )
    record_path = run_config(args.config, project_root, output_dir)
    print(f"Saved experiment record: {record_path}")


if __name__ == "__main__":
    main()

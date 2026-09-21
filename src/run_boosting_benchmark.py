"""Benchmark one real development fold for the NumPy histogram booster.

This runner is deliberately a feasibility measurement, not model selection.
It uses a fixed 0.5 threshold for comparable per-fold diagnostics and records
the best validation F1 threshold only as a descriptive quantity. A later
three-fold experiment must select thresholds within each outer training fold.
"""

from __future__ import annotations

import argparse
import gc
import json
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
    """Return classification, calibration, and ranking metrics at one threshold."""

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


def _benchmark_memory_bytes(train: np.ndarray, validation: np.ndarray) -> int:
    """Estimate the main retained arrays used by one fit and validation pass."""

    raw_bytes = train.nbytes + validation.nbytes
    binned_bytes = train.size + validation.size
    training_derivatives = train.shape[0] * 8 * 3
    validation_scores = validation.shape[0] * 8
    return int(raw_bytes + binned_bytes + training_derivatives + validation_scores)


def _load_variant(suite: dict, name: str) -> dict:
    """Return one named preprocessing variant or raise a clear configuration error."""

    matching = [
        variant for variant in suite["variants"] if variant["experiment_name"] == name
    ]
    if len(matching) != 1:
        raise ValueError(f"Expected exactly one preprocessing variant named {name}.")
    return matching[0]


def _validate_config(config: dict) -> None:
    """Reject a benchmark configuration that cannot make a fair measurement."""

    required = {
        "data",
        "feature_metadata_path",
        "preprocessing_config_path",
        "preprocessing_variant",
        "split_id",
        "fold_count",
        "fold_index",
        "seed",
        "fixed_threshold",
        "models",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(
            f"Benchmark configuration is missing: {', '.join(sorted(missing))}"
        )
    if type(config["fold_count"]) is not int or config["fold_count"] < 2:
        raise ValueError("fold_count must be an integer of at least two.")
    if (
        type(config["fold_index"]) is not int
        or not 1 <= config["fold_index"] <= config["fold_count"]
    ):
        raise ValueError(
            "fold_index must identify one configured fold, starting at one."
        )
    if type(config["seed"]) is not int:
        raise ValueError("seed must be an integer.")
    if not np.isfinite(config["fixed_threshold"]):
        raise ValueError("fixed_threshold must be finite.")
    if not isinstance(config["models"], list) or not config["models"]:
        raise ValueError("models must be a non-empty list.")
    names = []
    for model in config["models"]:
        if not isinstance(model, dict) or not isinstance(model.get("name"), str):
            raise ValueError("Each benchmark model requires a string name.")
        if not isinstance(model.get("parameters"), dict):
            raise ValueError("Each benchmark model requires a parameter mapping.")
        names.append(model["name"])
    if len(set(names)) != len(names):
        raise ValueError("Benchmark model names must be unique.")


def run_benchmark(
    config_path: Path,
    project_root: Path,
    output_dir: Path,
    selected_models: set[str] | None = None,
) -> list[Path]:
    """Run every configured booster on one fixed development fold and log it."""

    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    _validate_config(config)
    with _resolve(project_root, config["preprocessing_config_path"]).open(
        encoding="utf-8"
    ) as handle:
        preprocessing_suite = json.load(handle)
    variant = _load_variant(preprocessing_suite, config["preprocessing_variant"])
    data = config["data"]
    features_path = _resolve(project_root, data["features_path"])
    labels_path = _resolve(project_root, data["labels_path"])
    split_path = _resolve(project_root, data["split_indices_path"])
    metadata_path = _resolve(project_root, config["feature_metadata_path"])
    header_path = _resolve(project_root, data["x_train_csv_path"])
    features = np.load(features_path, mmap_mode="r")
    labels = _as_binary_labels(np.load(labels_path))
    with np.load(split_path) as split:
        development_indices = np.asarray(split["development_indices"], dtype=np.int64)
    if features.ndim != 2 or features.shape[0] != labels.size:
        raise ValueError("Feature and label caches must have matching row counts.")
    feature_names = _load_feature_names(header_path)
    if len(feature_names) != features.shape[1]:
        raise ValueError(
            "CSV header and cached feature matrix disagree on column count."
        )
    metadata = load_feature_metadata(metadata_path)
    development_labels = labels[development_indices]
    folds = list(
        _stratified_folds(development_labels, config["fold_count"], config["seed"])
    )
    train_positions, validation_positions = folds[config["fold_index"] - 1]
    train_rows = development_indices[train_positions]
    validation_rows = development_indices[validation_positions]
    train_source = np.asarray(features[train_rows], dtype=np.float32)
    validation_source = np.asarray(features[validation_rows], dtype=np.float32)
    train_features, validation_features, output_names = tree_feature_matrices(
        train_source,
        validation_source,
        feature_names,
        metadata,
        variant["preprocessing"],
    )
    train_labels = development_labels[train_positions]
    validation_labels = development_labels[validation_positions]
    input_hashes = {
        "benchmark_config": _sha256(config_path),
        "preprocessing_config": _sha256(
            _resolve(project_root, config["preprocessing_config_path"])
        ),
        "features": _sha256(features_path),
        "labels": _sha256(labels_path),
        "split_indices": _sha256(split_path),
        "feature_metadata": _sha256(metadata_path),
    }
    models = config["models"]
    if selected_models is not None:
        configured_names = {model["name"] for model in models}
        unknown = selected_models - configured_names
        if unknown:
            raise ValueError(
                f"Requested unknown benchmark models: {', '.join(sorted(unknown))}"
            )
        models = [model for model in models if model["name"] in selected_models]
    if not models:
        raise ValueError("No benchmark models were selected.")
    records = []
    for model_config in models:
        parameters = dict(model_config["parameters"])
        started = time.perf_counter()
        model = HistogramGradientBoostingClassifier(**parameters).fit(
            train_features, train_labels
        )
        probabilities = model.predict_proba(validation_features)[:, 1]
        elapsed = time.perf_counter() - started
        fixed_metrics, fixed_counts = _metrics(
            validation_labels, probabilities, float(config["fixed_threshold"])
        )
        descriptive_choice = best_f1_threshold(validation_labels, probabilities)
        descriptive_metrics, descriptive_counts = _metrics(
            validation_labels, probabilities, descriptive_choice.threshold
        )
        bin_counts = model.binner_.bin_counts_
        benchmark_name = (
            f"boosting_benchmark_{model_config['name']}_fold_{config['fold_index']}"
        )
        ledger_config = {
            "experiment_name": benchmark_name,
            "hypothesis": "Measure real-fold runtime, memory, and score behavior before selecting a full boosting experiment grid.",
            "data": {
                "dataset_id": "project1_processed_training_arrays",
                "features_path": data["features_path"],
                "labels_path": data["labels_path"],
            },
            "features": {
                "preprocessing_source": config["preprocessing_config_path"],
                "preprocessing_variant": variant["experiment_name"],
                "preprocessing": variant["preprocessing"],
                "output_feature_count": len(output_names),
                "output_feature_names": output_names,
                "native_missing_values": True,
            },
            "missing_values": {
                "method": "Codebook non-response codes become NaN; histogram trees select a missing-value branch per split.",
                "fit_partition": "outer fold training rows only for binary mappings and bin edges",
            },
            "split": {
                "id": config["split_id"],
                "outer_partition": "saved development indices only",
                "fold_count": config["fold_count"],
                "fold_index": config["fold_index"],
                "fold_seed": config["seed"],
                "training_rows": int(train_positions.size),
                "validation_rows": int(validation_positions.size),
            },
            "model": {
                "name": "numpy_histogram_gradient_boosting",
                "parameters": parameters,
            },
            "threshold": {
                "value": float(config["fixed_threshold"]),
                "selection_method": "Fixed feasibility-benchmark threshold; no threshold is selected for a claimed validation result.",
            },
            "seed": parameters.get("random_seed"),
        }
        outcome = {
            "status": "completed",
            "evaluation_partition": "one_development_fold_feasibility_benchmark",
            "metrics": fixed_metrics,
            "confusion_counts": fixed_counts,
            "descriptive_validation_threshold": {
                "threshold": descriptive_choice.threshold,
                "metrics": descriptive_metrics,
                "confusion_counts": descriptive_counts,
                "note": "Selected and evaluated on the same validation fold; retained only to inspect score separation.",
            },
            "model_details": {
                "tree_count": len(model.trees_),
                "base_score": model.base_score_,
                "bin_count_minimum": int(np.min(bin_counts)),
                "bin_count_maximum": int(np.max(bin_counts)),
            },
            "runtime_seconds": elapsed,
            "memory_estimate": {
                "arrays_bytes": _benchmark_memory_bytes(
                    train_features, validation_features
                ),
                "arrays_mebibytes": _benchmark_memory_bytes(
                    train_features, validation_features
                )
                / 1024**2,
                "basis": "Retained float32 cleaned matrices, uint8 binned matrices, three float64 training derivative/score arrays, and one float64 validation score array; excludes temporary node arrays and Python/NumPy runtime overhead.",
            },
            "input_sha256": input_hashes,
        }
        records.append(save_experiment(ledger_config, outcome, output_dir))
        print(
            f"{benchmark_name}: {elapsed:.1f}s, fixed-F1={fixed_metrics['f1']:.5f}, "
            f"AP={fixed_metrics['average_precision']:.5f}",
            flush=True,
        )
        del model, probabilities
        gc.collect()
    del train_source, validation_source, train_features, validation_features
    gc.collect()
    return records


def main() -> None:
    """Run the configured one-fold NumPy boosting benchmark."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", type=Path, help="Path to the benchmark JSON configuration."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("results/experiments"))
    parser.add_argument(
        "--models", nargs="+", help="Optional configured model names to run."
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    )
    selected = None if args.models is None else set(args.models)
    for record in run_benchmark(args.config, root, output_dir, selected):
        print(f"Saved experiment record: {record}")


if __name__ == "__main__":
    main()

"""Train the selected NumPy booster and create an AIcrowd submission.

The default configuration reproduces the current reference pipeline. Raw CSV
files are converted to local NumPy caches on the first run; later runs reuse
them only when receipts match the CSV contents, cached arrays and feature order.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from helpers import create_csv_submission
from src.feature_preprocessing import load_feature_metadata, tree_feature_matrices
from src.numpy_boosting import HistogramGradientBoostingClassifier
from src.boosting_implementation import get_booster_class

DEFAULT_CONFIG = Path("configs/final_model.json")


def _count_rows(path: Path) -> int:
    """Count data rows in a CSV file without retaining them in memory."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        row_count = sum(1 for _ in handle) - 1
    if row_count <= 0:
        raise ValueError(f"{path} contains no data rows.")
    return row_count


def _feature_header(path: Path) -> list[str]:
    """Return feature names from an official feature CSV."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    if len(header) < 2 or header[0] != "Id":
        raise ValueError(f"{path.name} must start with Id followed by features.")
    return header[1:]


def _read_feature_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load one official feature CSV into ID and float32 arrays."""

    feature_names = _feature_header(path)
    row_count = _count_rows(path)
    ids = np.empty(row_count, dtype=np.int64)
    features = np.empty((row_count, len(feature_names)), dtype=np.float32)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for index, row in enumerate(reader):
            if len(row) != len(feature_names) + 1:
                raise ValueError(
                    f"{path.name}:{index + 2} has {len(row)} fields; "
                    f"expected {len(feature_names) + 1}."
                )
            ids[index] = int(row[0])
            features[index] = [float(value) if value else np.nan for value in row[1:]]
    if np.unique(ids).size != ids.size:
        raise ValueError(f"{path.name} contains duplicate IDs.")
    return ids, features, feature_names


def _read_labels_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate the official signed training labels."""

    row_count = _count_rows(path)
    ids = np.empty(row_count, dtype=np.int64)
    labels = np.empty(row_count, dtype=np.int8)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader) != ["Id", "_MICHD"]:
            raise ValueError("y_train.csv must have header Id,_MICHD.")
        for index, row in enumerate(reader):
            if len(row) != 2:
                raise ValueError(f"y_train.csv:{index + 2} must contain two fields.")
            ids[index] = int(row[0])
            labels[index] = int(row[1])
    if not np.all((labels == -1) | (labels == 1)):
        raise ValueError("Training labels must use the {-1, +1} domain.")
    return ids, labels


def _cache_paths(cache_dir: Path, split: str) -> tuple[Path, Path]:
    """Return feature and ID cache paths for a data split."""

    return cache_dir / f"x_{split}_float32.npy", cache_dir / f"id_{split}_int64.npy"


def _file_sha256(path: Path) -> str:
    """Hash large CSVs and caches without loading them into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_cache_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    """Publish the receipt only after both cache files have been written."""

    candidate = path.with_name(path.name + "." + os.urandom(8).hex() + ".tmp")
    temporary = None
    try:
        with candidate.open("x", encoding="utf-8", newline="\n") as handle:
            temporary = candidate
            json.dump(receipt, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _ensure_feature_cache(
    csv_path: Path,
    cache_dir: Path,
    split: str,
    expected_names: Sequence[str] | None = None,
) -> tuple[Path, Path, list[str]]:
    """Build new caches, or reject caches without matching source/file receipts."""

    feature_path, id_path = _cache_paths(cache_dir, split)
    receipt_path = cache_dir / f"x_{split}_cache.json"
    feature_names = _feature_header(csv_path)
    if expected_names is not None and list(expected_names) != feature_names:
        raise ValueError(
            f"{csv_path.name} feature names or order differ from x_train.csv."
        )
    source_sha256 = _file_sha256(csv_path)

    def invalid_cache(reason: str) -> ValueError:
        return ValueError(
            f"Cannot reuse {split} feature cache in {cache_dir}: {reason}. "
            "Use a new cache directory (--cache-dir, or --output-dir for the report runner) "
            "to rebuild it from the current CSVs. "
            "Existing cache files have not been removed."
        )

    paths = (feature_path, id_path, receipt_path)
    if any(path.exists() for path in paths):
        if not all(path.is_file() for path in paths):
            raise invalid_cache("incomplete cache or missing source receipt")
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise invalid_cache("invalid source receipt") from exc
        if not isinstance(receipt, dict) or receipt.get("schema_version") != 1:
            raise invalid_cache("unsupported source receipt")
        if receipt.get("csv_sha256") != source_sha256:
            raise invalid_cache("source CSV contents changed")
        if receipt.get("feature_names") != feature_names:
            raise invalid_cache("feature names or order changed")
        expected_files = {
            feature_path.name: _file_sha256(feature_path),
            id_path.name: _file_sha256(id_path),
        }
        if receipt.get("files") != expected_files:
            raise invalid_cache("cached array contents changed")
        features = np.load(feature_path, mmap_mode="r")
        ids = np.load(id_path, mmap_mode="r")
        if (ids.ndim != 1 or ids.dtype != np.dtype("int64")
                or features.dtype != np.dtype("float32")
                or features.shape != (ids.size, len(feature_names))
                or receipt.get("shape") != list(features.shape)):
            raise invalid_cache("cached array schema differs from the receipt")
        return feature_path, id_path, feature_names

    cache_dir.mkdir(parents=True, exist_ok=True)
    ids, features, parsed_names = _read_feature_csv(csv_path)
    if parsed_names != feature_names or _file_sha256(csv_path) != source_sha256:
        raise ValueError(f"{csv_path.name} changed while preparing its cache; retry with stable input files.")
    np.save(feature_path, features)
    np.save(id_path, ids)
    _write_cache_receipt(receipt_path, {
        "schema_version": 1,
        "csv_sha256": source_sha256,
        "feature_names": parsed_names,
        "shape": list(features.shape),
        "files": {
            feature_path.name: _file_sha256(feature_path),
            id_path.name: _file_sha256(id_path),
        },
    })
    return feature_path, id_path, parsed_names


def _ensure_label_cache(
    labels_path: Path, cache_dir: Path, expected_ids: np.ndarray
) -> Path:
    """Build or validate the signed-label cache."""

    cache_path = cache_dir / "y_train_int8.npy"
    label_ids, labels = _read_labels_csv(labels_path)
    if not np.array_equal(label_ids, expected_ids):
        raise ValueError("x_train.csv and y_train.csv IDs do not align.")
    if cache_path.is_file():
        cached = np.load(cache_path, mmap_mode="r")
        if cached.shape != labels.shape or not np.array_equal(cached, labels):
            raise ValueError("Cached labels do not align with training IDs.")
        return cache_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, labels)
    return cache_path


def _load_variant(path: Path, name: str) -> Mapping[str, Any]:
    """Load one named preprocessing variant."""

    with path.open(encoding="utf-8") as handle:
        suite = json.load(handle)
    for variant in suite.get("variants", []):
        if variant.get("experiment_name") == name:
            return variant["preprocessing"]
    raise ValueError(f"Preprocessing variant {name!r} was not found in {path}.")


def _write_submission(path: Path, ids: np.ndarray, predictions: np.ndarray) -> None:
    """Validate vectors, then save them with the course submission helper."""

    ids = np.asarray(ids)
    predictions = np.asarray(predictions)
    if ids.ndim != 1 or predictions.shape != ids.shape:
        raise ValueError("Submission IDs and predictions must be aligned vectors.")
    if not np.all((predictions == -1) | (predictions == 1)):
        raise ValueError("Submission predictions must use the {-1, +1} domain.")
    path.parent.mkdir(parents=True, exist_ok=True)
    create_csv_submission(ids.astype(np.int64), predictions.astype(np.int8), path)


def _read_sample_submission_ids(path: Path) -> np.ndarray:
    """Read the official submission IDs and validate its two-column schema."""

    row_count = _count_rows(path)
    ids = np.empty(row_count, dtype=np.int64)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader) != ["Id", "Prediction"]:
            raise ValueError("sample_submission.csv must have header Id,Prediction.")
        for index, row in enumerate(reader):
            if len(row) != 2:
                raise ValueError(
                    f"sample_submission.csv:{index + 2} must contain two fields."
                )
            ids[index] = int(row[0])
    return ids


def run_pipeline(
    project_root: Path,
    config_path: Path,
    data_dir: Path,
    cache_dir: Path,
    output_path: Path,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Fit the configured model on all training rows and create predictions."""

    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    train_csv = data_dir / config["data"]["train_features"]
    labels_csv = data_dir / config["data"]["train_labels"]
    test_csv = data_dir / config["data"]["test_features"]
    sample_submission = data_dir / config["data"]["sample_submission"]
    for path in (train_csv, labels_csv, test_csv, sample_submission):
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset file is missing: {path}")

    train_feature_path, train_id_path, feature_names = _ensure_feature_cache(
        train_csv, cache_dir, "train"
    )
    test_feature_path, test_id_path, _ = _ensure_feature_cache(
        test_csv, cache_dir, "test", feature_names
    )
    train_ids = np.load(train_id_path, mmap_mode="r")
    test_ids = np.load(test_id_path, mmap_mode="r")
    if not np.array_equal(_read_sample_submission_ids(sample_submission), test_ids):
        raise ValueError("x_test.csv IDs do not match sample_submission.csv.")
    label_path = _ensure_label_cache(labels_csv, cache_dir, train_ids)
    signed_labels = np.load(label_path, mmap_mode="r")
    labels = ((signed_labels + 1) // 2).astype(np.int8)

    preprocessing_path = project_root / config["preprocessing"]["config"]
    metadata_path = project_root / config["preprocessing"]["metadata"]
    plan = _load_variant(preprocessing_path, config["preprocessing"]["variant"])
    metadata = load_feature_metadata(metadata_path)
    train_source = np.load(train_feature_path, mmap_mode="r")
    test_source = np.load(test_feature_path, mmap_mode="r")
    train_matrix, test_matrix, output_names = tree_feature_matrices(
        train_source, test_source, feature_names, metadata, plan
    )

    started = time.perf_counter()
    model_class = (get_booster_class(config["implementation"])
                   if "implementation" in config else HistogramGradientBoostingClassifier)
    model = model_class(**config["model"]).fit(
        train_matrix, labels
    )
    probabilities = model.predict_proba(test_matrix)[:, 1]
    threshold = float(config["decision_threshold"])
    predictions = np.where(probabilities >= threshold, 1, -1).astype(np.int8)
    _write_submission(output_path, test_ids, predictions)
    if checkpoint_path is not None:
        model.save_checkpoint(
            checkpoint_path,
            {
                "artifact_type": "trusted_local_final_model",
                "feature_names": output_names,
                "decision_threshold": threshold,
                "training_rows": int(train_matrix.shape[0]),
            },
        )
    return {
        "training_rows": int(train_matrix.shape[0]),
        "test_rows": int(test_matrix.shape[0]),
        "features": len(output_names),
        "positive_predictions": int(np.count_nonzero(predictions == 1)),
        "negative_predictions": int(np.count_nonzero(predictions == -1)),
        "threshold": threshold,
        "implementation": config.get("implementation", "phase34"),
        "fit_and_prediction_seconds": time.perf_counter() - started,
        "output": str(output_path),
    }


def main() -> None:
    """Parse command-line arguments and execute the reference pipeline."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/dataset"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else project_root / path

    summary = run_pipeline(
        project_root,
        resolved(args.config),
        resolved(args.data_dir),
        resolved(args.cache_dir),
        resolved(args.output),
        None if args.checkpoint is None else resolved(args.checkpoint),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

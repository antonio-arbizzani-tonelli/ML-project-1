"""Train the experimental NumPy MLP on all labels and write signed predictions.

    python run_mlp.py --model mlp128_64 --threshold VALIDATED_THRESHOLD

The threshold must be supplied from development validation. This command does
not select a threshold using the competition test set or replace run.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from run import _ensure_feature_cache, _ensure_label_cache, _load_variant, _read_sample_submission_ids, _write_submission
from src.feature_preprocessing import load_feature_metadata
from src.run_mlp_cv import fit_partition, load_pipeline, save_pipeline, validate_suite
from src.run_preprocessing_cv import _as_binary_labels


def run_pipeline(project_root, config_path, model_name, threshold, data_dir,
                 cache_dir, artifact_dir, output_path, quiet=False):
    """Fit epochs on a training holdout, refit on all labels, then predict test rows."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("A validated probability threshold between zero and one is required.")
    suite = json.loads(Path(config_path).read_text(encoding="utf-8"))
    validate_suite(suite)
    models = [model for model in suite["models"] if model["name"] == model_name]
    if len(models) != 1:
        raise ValueError("model_name must identify exactly one configured MLP.")
    project_root, data_dir, cache_dir = map(Path, (project_root, data_dir, cache_dir))
    train_path, train_ids_path, names = _ensure_feature_cache(data_dir / "x_train.csv", cache_dir, "train")
    test_path, test_ids_path, _ = _ensure_feature_cache(data_dir / "x_test.csv", cache_dir, "test", expected_names=names)
    train_ids = np.load(train_ids_path)
    labels_path = _ensure_label_cache(data_dir / "y_train.csv", cache_dir, train_ids)
    labels = _as_binary_labels(np.load(labels_path))
    test_ids = np.load(test_ids_path)
    sample = data_dir / "sample_submission.csv"
    if sample.is_file() and not np.array_equal(test_ids, _read_sample_submission_ids(sample)):
        raise ValueError("Test IDs differ from the sample submission order.")
    training = np.load(train_path, mmap_mode="r")
    testing = np.load(test_path, mmap_mode="r")
    source = np.concatenate((training, testing), axis=0)
    training_rows = np.arange(labels.size, dtype=np.int64)
    testing_rows = np.arange(labels.size, source.shape[0], dtype=np.int64)
    # No test targets exist: placeholders are never read by fit_partition.
    aligned_labels = np.r_[labels, np.zeros(test_ids.size, dtype=np.int8)]
    metadata = load_feature_metadata(project_root / suite["feature_metadata_path"])
    cleaning = _load_variant(project_root / suite["cleaning_config_path"], suite["cleaning_variant"])
    probabilities, report = fit_partition(source, aligned_labels, training_rows, testing_rows,
        names, metadata, cleaning, suite["preprocessing"], models, Path(artifact_dir),
        "full_training", suite["preprocessing"]["early_stopping_seed"], quiet=quiet)
    details = report["models"][model_name]
    original_checkpoint = Path(details["checkpoint"]["path"])
    payload = load_pipeline(original_checkpoint)
    payload["metadata"]["threshold"] = float(threshold)
    checkpoint = original_checkpoint.with_name(f"full_training_{model_name}_with_threshold.pkl")
    save_pipeline(checkpoint, payload["model"], payload["preprocessor"], payload["metadata"])
    signed_predictions = np.where(probabilities[model_name] >= threshold, 1, -1).astype(np.int8)
    _write_submission(Path(output_path), test_ids, signed_predictions)
    return {"model": model_name, "threshold": float(threshold), "output": str(output_path),
            "checkpoint": str(checkpoint), "training_rows": labels.size,
            "predictions": test_ids.size, "selected_epochs": details["selected_epochs"]}


def main():
    """Expose an explicit experimental full-training entry point."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/phase32_mlp.json"))
    parser.add_argument("--model", default="mlp128_64")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/dataset"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("results/mlp_artifacts/full_training"))
    parser.add_argument("--output", type=Path, default=Path("results/submission_mlp.csv"))
    args = parser.parse_args()
    resolved = lambda path: path if path.is_absolute() else root / path
    result = run_pipeline(root, resolved(args.config), args.model, args.threshold,
                          resolved(args.data_dir), resolved(args.cache_dir),
                          resolved(args.artifact_dir), resolved(args.output))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

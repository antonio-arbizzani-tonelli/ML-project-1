"""Reproducible nested development CV for the separate NumPy MLP pipeline."""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import platform
import time
from pathlib import Path

import numpy as np

from src.evaluation import average_precision, best_f1_threshold, binary_log_loss, classification_metrics, labels_from_scores
from src.experiment_logger import _write_json_atomically, save_experiment
from src.feature_preprocessing import load_feature_metadata
from src.mlp_preprocessing import MLPPreprocessor
from src.numpy_mlp import MLPClassifier
from src.run_boosting_cv import _metrics, _save_inner_oof_predictions, _save_oof_predictions
from src.run_preprocessing_cv import _as_binary_labels, _load_feature_names, _resolve, _sha256, _stratified_folds


def stratified_holdout(labels, fraction, seed):
    """Split only the supplied training labels for independent early stopping."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Holdout labels must be one-dimensional 0/1 values.")
    if not 0 < fraction < 0.5:
        raise ValueError("Early-stopping fraction must be between zero and 0.5.")
    rng = np.random.default_rng(seed)
    held_out = []
    for value in (0, 1):
        positions = np.flatnonzero(labels == value)
        if positions.size < 2:
            raise ValueError("Each class needs at least two training rows.")
        rng.shuffle(positions)
        count = min(positions.size - 1, max(1, int(round(fraction * positions.size))))
        held_out.extend(positions[:count])
    validation = np.sort(np.asarray(held_out, dtype=np.int64))
    mask = np.ones(labels.size, dtype=bool)
    mask[validation] = False
    return np.flatnonzero(mask), validation


def save_pipeline(path, model, preprocessor, metadata):
    """Save the fitted network and preprocessing together as a trusted local pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump({"format": "numpy_mlp_pipeline_v1", "model": model,
                     "preprocessor": preprocessor, "metadata": metadata},
                    handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def load_pipeline(path):
    """Load only a trusted local pipeline; pickle is not safe for untrusted files."""
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if (not isinstance(payload, dict) or payload.get("format") != "numpy_mlp_pipeline_v1"
            or not isinstance(payload.get("model"), MLPClassifier)
            or not isinstance(payload.get("preprocessor"), MLPPreprocessor)):
        raise ValueError("The checkpoint is not a fitted NumPy MLP pipeline.")
    return payload


def predict_pipeline(payload, source_features, batch_size=4096):
    """Predict raw survey rows in bounded batches using the frozen representation."""
    source_features = np.asarray(source_features)
    if source_features.ndim != 2 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Source features must be a matrix and batch_size positive.")
    predictions = np.empty(source_features.shape[0], dtype=np.float64)
    for start in range(0, predictions.size, batch_size):
        transformed = payload["preprocessor"].transform(source_features[start:start + batch_size])
        predictions[start:start + batch_size] = payload["model"].predict_proba(transformed)
    return predictions


def fit_partition(source, labels, training_rows, evaluation_rows, feature_names,
                  metadata, cleaning_plan, settings, models, artifact_dir, tag,
                  selection_seed, quiet=False):
    """Select epochs within training, then refit both preprocessing and model.

    Evaluation labels are deliberately not accepted. Only training rows enter
    preprocessing, early stopping, initialization and Adam updates.
    """
    training_rows = np.asarray(training_rows, dtype=np.int64)
    evaluation_rows = np.asarray(evaluation_rows, dtype=np.int64)
    if (np.unique(training_rows).size != training_rows.size or
            np.unique(evaluation_rows).size != evaluation_rows.size or
            np.intersect1d(training_rows, evaluation_rows).size):
        raise ValueError("Fit and evaluation rows must be unique and disjoint.")
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    fit_positions, stopping_positions = stratified_holdout(
        labels[training_rows], settings["early_stopping_fraction"], selection_seed)
    fit_rows, stopping_rows = training_rows[fit_positions], training_rows[stopping_positions]
    boundary_path = artifact_dir / f"{tag}_boundaries.npz"
    np.savez_compressed(boundary_path, training_rows=training_rows, evaluation_rows=evaluation_rows,
                        gradient_fit_rows=fit_rows, early_stopping_rows=stopping_rows)
    started = time.perf_counter()

    def processor():
        return MLPPreprocessor(feature_names, metadata, cleaning_plan,
                               settings.get("frequency_features", []))

    def progress(name, stage):
        def callback(entry):
            if not quiet and (entry["epoch"] == 1 or entry["epoch"] % 5 == 0):
                print(json.dumps({"fit": tag, "model": name, "stage": stage, **entry}), flush=True)
        return callback

    selection_processor = processor().fit(source[fit_rows])
    fit_design = selection_processor.transform(source[fit_rows])
    stopping_design = selection_processor.transform(source[stopping_rows])
    selection_audit = selection_processor.audit()
    chosen = {}
    for specification in models:
        model_started = time.perf_counter()
        model = MLPClassifier(**specification["parameters"])
        model.fit(fit_design, labels[fit_rows], stopping_design, labels[stopping_rows],
                  progress_callback=progress(specification["name"], "select_epochs"))
        chosen[specification["name"]] = {
            "selected_epochs": model.best_epoch_, "epochs_attempted": model.epochs_trained_,
            "selection_history": model.history_,
            "selection_seconds": time.perf_counter() - model_started,
            "epoch_limit_reached": model.epochs_trained_ == model.max_epochs,
        }
        _write_json_atomically(artifact_dir / f"{tag}_progress.json", chosen)
    del fit_design, stopping_design, selection_processor, model
    gc.collect()
    full_processor = processor().fit(source[training_rows])
    training_design = full_processor.transform(source[training_rows])
    evaluation_design = full_processor.transform(source[evaluation_rows])
    audit = full_processor.audit()
    probabilities, details = {}, {}
    for specification in models:
        name = specification["name"]
        parameters = dict(specification["parameters"])
        parameters["max_epochs"] = chosen[name]["selected_epochs"]
        model_started = time.perf_counter()
        fitted = MLPClassifier(**parameters).fit(
            training_design, labels[training_rows],
            progress_callback=progress(name, "refit_full_training"))
        probabilities[name] = fitted.predict_proba(evaluation_design)
        checkpoint = artifact_dir / f"{tag}_{name}.pkl"
        save_pipeline(checkpoint, fitted, full_processor,
                      {"tag": tag, "parameters": parameters,
                       "boundaries_path": str(boundary_path), "selection_seed": selection_seed})
        # Verify persisted inference, including transformations, on a small
        # evaluation batch without looking at evaluation labels.
        reloaded = load_pipeline(checkpoint)
        count = min(64, evaluation_rows.size)
        np.testing.assert_array_equal(
            reloaded["model"].predict_proba(evaluation_design[:count]),
            fitted.predict_proba(evaluation_design[:count]))
        np.testing.assert_array_equal(reloaded["preprocessor"].transform(source[evaluation_rows[:count]]),
                                      evaluation_design[:count])
        details[name] = {**chosen[name], "refit_history": fitted.history_,
                         "refit_seconds": time.perf_counter() - model_started,
                         "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
                         "parameter_count": sum(p.size for p in fitted.weights_ + fitted.biases_)}
        _write_json_atomically(artifact_dir / f"{tag}_progress.json", details)
    report = {"tag": tag, "selection_seed": selection_seed,
              "training_rows": int(training_rows.size), "evaluation_rows": int(evaluation_rows.size),
              "early_stopping_rows": int(stopping_rows.size),
              "boundaries": {"path": str(boundary_path), "sha256": _sha256(boundary_path)},
              "selection_preprocessing": selection_audit, "refit_preprocessing": audit,
              "models": details, "runtime_seconds": time.perf_counter() - started,
              "design_memory_mebibytes": (training_design.nbytes + evaluation_design.nbytes) / 2**20}
    _write_json_atomically(artifact_dir / f"{tag}_fit.json", report)
    return probabilities, report


def validate_suite(suite):
    """Require deterministic nested CV settings and valid unique model names."""
    for key in ("experiment_prefix", "data", "feature_metadata_path", "cleaning_config_path",
                "cleaning_variant", "split_id", "fold_count", "seed", "inner_fold_count",
                "inner_seed", "preprocessing", "models", "artifact_dir"):
        if key not in suite:
            raise ValueError(f"MLP suite is missing {key}.")
    for key in ("fold_count", "inner_fold_count"):
        if type(suite[key]) is not int or suite[key] < 2:
            raise ValueError(f"{key} must be an integer of at least two.")
    names = []
    for model in suite["models"]:
        name = model["name"]
        if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name):
            raise ValueError("Model names must be lowercase letters, digits and underscores.")
        MLPClassifier(**model["parameters"])
        names.append(name)
    if not names or len(names) != len(set(names)):
        raise ValueError("Model names must be nonempty and unique.")
    fraction = suite["preprocessing"].get("early_stopping_fraction")
    if not isinstance(fraction, (int, float)) or not 0 < fraction < 0.5:
        raise ValueError("Configure an early-stopping fraction between zero and 0.5.")


def run_suite(config_path, project_root, output_dir, quiet=False):
    """Evaluate candidates on every development row with nested thresholds."""
    config_path, project_root, output_dir = map(Path, (config_path, project_root, output_dir))
    suite = json.loads(config_path.read_text(encoding="utf-8"))
    validate_suite(suite)
    data = suite["data"]
    paths = {key: _resolve(project_root, value) for key, value in data.items()}
    metadata_path = _resolve(project_root, suite["feature_metadata_path"])
    cleaning_path = _resolve(project_root, suite["cleaning_config_path"])
    cleaning_suite = json.loads(cleaning_path.read_text(encoding="utf-8"))
    variants = [v for v in cleaning_suite["variants"] if v["experiment_name"] == suite["cleaning_variant"]]
    if len(variants) != 1:
        raise ValueError("Expected exactly one configured cleaning variant.")
    plan = variants[0]["preprocessing"]
    source = np.load(paths["features_path"], mmap_mode="r")
    labels = _as_binary_labels(np.load(paths["labels_path"]))
    feature_names = _load_feature_names(paths["x_train_csv_path"])
    metadata = load_feature_metadata(metadata_path)
    with np.load(paths["split_indices_path"]) as split:
        rows = np.asarray(split["development_indices"], dtype=np.int64)
    if (source.shape != (labels.size, len(feature_names)) or rows.ndim != 1
            or np.unique(rows).size != rows.size or np.any(rows < 0) or np.any(rows >= labels.size)):
        raise ValueError("Data, schema or development rows do not align.")
    development_labels = labels[rows]
    folds = list(_stratified_folds(development_labels, suite["fold_count"], suite["seed"]))
    artifact_dir = project_root / suite["artifact_dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__), Path(__file__).with_name("numpy_mlp.py"),
                    Path(__file__).with_name("mlp_preprocessing.py"),
                    Path(__file__).with_name("feature_preprocessing.py"),
                    Path(__file__).with_name("evaluation.py"),
                    Path(__file__).with_name("run_preprocessing_cv.py"),
                    Path(__file__).with_name("run_boosting_cv.py")]
    fingerprints = {**{key: _sha256(path) for key, path in paths.items()},
                    "suite_config": _sha256(config_path), "cleaning_config": _sha256(cleaning_path),
                    "feature_metadata": _sha256(metadata_path),
                    **{path.name: _sha256(path) for path in source_paths}}
    _write_json_atomically(artifact_dir / "resolved_config.json", suite)
    for path in source_paths:
        (artifact_dir / f"executed_{path.name}").write_bytes(path.read_bytes())
    names = [m["name"] for m in suite["models"]]
    scores = {n: np.full(rows.size, np.nan) for n in names}
    thresholds = {n: np.full(rows.size, np.nan) for n in names}
    outer_ids = np.zeros(rows.size, dtype=np.int8)
    details = {n: [] for n in names}
    started = time.perf_counter()
    for outer_fold, (train_positions, validation_positions) in enumerate(folds, 1):
        training_rows, validation_rows = rows[train_positions], rows[validation_positions]
        inner_scores = {n: np.full(train_positions.size, np.nan) for n in names}
        inner_ids = np.zeros(train_positions.size, dtype=np.int8)
        inner_fits = []
        for inner_fold, (inner_train, inner_valid) in enumerate(_stratified_folds(
                development_labels[train_positions], suite["inner_fold_count"], suite["inner_seed"] + outer_fold), 1):
            predicted, report = fit_partition(
                source, labels, training_rows[inner_train], training_rows[inner_valid], feature_names,
                metadata, plan, suite["preprocessing"], suite["models"], artifact_dir,
                f"outer{outer_fold}_inner{inner_fold}", suite["preprocessing"]["early_stopping_seed"] + outer_fold * 10 + inner_fold,
                quiet=quiet)
            inner_fits.append({"tag": report["tag"], "runtime_seconds": report["runtime_seconds"],
                               "models": report["models"]})
            inner_ids[inner_valid] = inner_fold
            for name in names:
                inner_scores[name][inner_valid] = predicted[name]
            gc.collect()
        outer_predicted, outer_report = fit_partition(
            source, labels, training_rows, validation_rows, feature_names, metadata, plan,
            suite["preprocessing"], suite["models"], artifact_dir, f"outer{outer_fold}",
            suite["preprocessing"]["early_stopping_seed"] + outer_fold * 10, quiet=quiet)
        outer_ids[validation_positions] = outer_fold
        for name in names:
            choice = best_f1_threshold(labels[training_rows], inner_scores[name])
            inner_artifact = _save_inner_oof_predictions(
                artifact_dir, f"{suite['experiment_prefix']}_{name}", outer_fold,
                suite["inner_seed"] + outer_fold, training_rows, validation_rows, inner_ids,
                labels[training_rows], inner_scores[name])
            scores[name][validation_positions] = outer_predicted[name]
            thresholds[name][validation_positions] = choice.threshold
            metrics, counts = _metrics(labels[validation_rows], outer_predicted[name], choice.threshold)
            details[name].append({"outer_fold": outer_fold, "inner_threshold": choice.threshold,
                                  "inner_selection_metrics": _metrics(labels[training_rows], inner_scores[name], choice.threshold)[0],
                                  "outer_evaluation_metrics": metrics,
                                  "outer_evaluation_confusion_counts": counts,
                                  "inner_oof_predictions_artifact": inner_artifact,
                                  "inner_fits": inner_fits,
                                  "outer_fit": outer_report["models"][name],
                                  "output_features": outer_report["refit_preprocessing"]["output_count"]})
            if not quiet:
                print(json.dumps({"outer_fold_completed": outer_fold, "model": name,
                                  "threshold": choice.threshold, "metrics": metrics}), flush=True)
        _write_json_atomically(artifact_dir / "status.json", {"completed_outer_folds": outer_fold,
            "total_outer_folds": suite["fold_count"], "status": "running",
            "elapsed_seconds": time.perf_counter() - started, "folds": details})
    runtime = time.perf_counter() - started
    # Detect changes to executed dependencies/configs during the experiment.
    if any(_sha256(path) != fingerprints[path.name] for path in source_paths):
        raise RuntimeError("A training dependency changed during the experiment.")
    records = []
    for name in names:
        if not np.all(np.isfinite(scores[name])) or not np.all(np.isfinite(thresholds[name])) or np.any(outer_ids == 0):
            raise RuntimeError("The nested experiment does not cover all development rows.")
        predictions = labels_from_scores(scores[name] - thresholds[name], 0)
        counts, metrics = classification_metrics(development_labels, predictions)
        metrics["average_precision"] = average_precision(development_labels, scores[name])
        metrics["log_loss"] = binary_log_loss(development_labels, scores[name])
        # 0.5 is saved solely as an unused diagnostic threshold in this shared
        # artifact format. Final metrics always use the nested per-row threshold.
        oof_path, oof_hash = _save_oof_predictions(artifact_dir, f"{suite['experiment_prefix']}_{name}",
            rows, outer_ids, development_labels, scores[name], 0.5, thresholds[name])
        model_specification = next(m for m in suite["models"] if m["name"] == name)
        config = {"experiment_name": f"{suite['experiment_prefix']}_{name}",
                  "hypothesis": "A fold-safe NumPy ReLU MLP with model-specific finite inputs improves the Phase 14 binary F1 baseline.",
                  "data": data, "features": {"cleaning_plan": plan, "mlp_preprocessing": suite["preprocessing"]},
                  "missing_values": {"numeric": "training median/mode plus missing flag",
                                     "nominal": "full one-hot plus missing and unseen flags"},
                  "split": {"id": suite["split_id"], "outer_folds": suite["fold_count"], "outer_seed": suite["seed"],
                            "inner_folds": suite["inner_fold_count"], "inner_seed": suite["inner_seed"]},
                  "model": {"name": "numpy_mlp", "parameters": model_specification["parameters"]},
                  "threshold": {"selection_method": "inner OOF F1 threshold separately for each outer training fold"}}
        outcome = {"status": "completed", "metrics": metrics,
                   "confusion_counts": {key: int(getattr(counts, key)) for key in counts.__dataclass_fields__},
                   "nested_threshold_evaluation": {"folds": details[name], "pooled_metrics": metrics},
                   "oof_predictions_artifact": {"path": str(oof_path.relative_to(project_root)), "sha256": oof_hash},
                   "fingerprints": fingerprints, "development_rows": int(rows.size),
                   "runtime_seconds": runtime, "runtime_basis": "Complete shared suite, including preprocessing, epoch selection and full refits for both candidates.",
                   "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                   "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS")},
                   "evaluation_limitations": "Many configurations have already been compared on these development folds. This is development validation, not an untouched final test."}
        records.append(save_experiment(config, outcome, output_dir))
    _write_json_atomically(artifact_dir / "status.json", {"status": "completed", "runtime_seconds": runtime,
                                                          "records": [str(p) for p in records]})
    return records


def main():
    """Run the configured experiment without changing the selected booster."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("results/experiments/mlp"))
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else args.project_root / args.output_dir
    for record in run_suite(args.config, args.project_root, output):
        print(f"Completed: {record}", flush=True)


if __name__ == "__main__":
    main()

"""Calibrate both fixed boosters on all labeled rows, then create two submissions.

Usage: python -B -m tools.final_phase14_phase34 prepare
       python -B -m tools.final_phase14_phase34 run --workers 1
       python -B -m tools.final_phase14_phase34 status

Only this run's artifact directory is written. OOF F1 selects each threshold;
it is a calibration statistic, not an unbiased test score. Official AIcrowd
scores remain empty until the two CSVs are submitted by the user.
"""

from __future__ import annotations

import os
for _thread_name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_name] = "1"

import argparse
import concurrent.futures
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from run import _feature_header, _read_labels_csv, _read_sample_submission_ids, _write_submission
from src.run_preprocessing_cv import _as_binary_labels, _stratified_folds

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/boosting_artifacts/final_phase14_phase34"
REFERENCE = "configs/experiments/phase14_boosting_codebook_corrections_only.json"
HISTORICAL = "src/numpy_boosting_phase14.py"
HISTORICAL_SHA256 = "6fede57915b07a6ea7675292833f8ad79269e665f09977bffd2a9562ce1aaf98"
EXPECTED_SHAPE = (328135, 109379, 321)
EXPECTED_FEATURES = 295
VARIANTS = ("phase14", "phase34")
MODULE_NAMES = {"phase14": "_final_phase14_historical_booster",
                "phase34": "_final_phase34_corrected_booster",
                "preprocessing": "_final_phase14_phase34_preprocessing",
                "evaluation": "_final_phase14_phase34_evaluation"}
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_checksum(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(values.shape).encode())
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_array(path, values):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def environment():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "threads": {name: os.environ.get(name) for name in THREADS}}


def full_train_folds(labels, seed):
    """Require disjoint train/validation rows and exactly one OOF prediction per row."""
    folds = list(_stratified_folds(labels, 3, seed))
    coverage = np.zeros(len(labels), dtype=np.int8)
    all_rows = np.arange(len(labels), dtype=np.int64)
    for train, valid in folds:
        if np.intersect1d(train, valid).size:
            raise ValueError("A calibration fold leaks validation rows into training.")
        if not np.array_equal(np.sort(np.r_[train, valid]), all_rows):
            raise ValueError("A calibration fold does not partition all labeled rows.")
        coverage[valid] += 1
    if not np.all(coverage == 1):
        raise ValueError("OOF calibration must cover every labeled row exactly once.")
    return folds


def _csv_ids_match(path, expected):
    count = 0
    with Path(path).open(encoding="utf-8", newline="") as handle:
        next(handle)
        for count, line in enumerate(handle, start=1):
            if count > len(expected) or int(line.split(",", 1)[0]) != expected[count - 1]:
                raise ValueError(f"Cached IDs differ from {path} at data row {count}.")
    if count != len(expected) or np.unique(expected).size != len(expected):
        raise ValueError(f"Row count or unique IDs are invalid: {path}.")


def _dataset(root, suite):
    data = suite["data"]
    train_csv = root / data["x_train_csv_path"]
    raw_dir = train_csv.parent
    cache_dir = (root / data["features_path"]).parent
    paths = {"train_features": root / data["features_path"],
             "train_labels": root / data["labels_path"],
             "test_features": cache_dir / "x_test_float32.npy",
             "train_ids": cache_dir / "id_train_int64.npy",
             "test_ids": cache_dir / "id_test_int64.npy",
             "train_csv": train_csv, "test_csv": raw_dir / "x_test.csv",
             "labels_csv": raw_dir / "y_train.csv", "sample_submission": raw_dir / "sample_submission.csv"}
    names = _feature_header(train_csv)
    if names != _feature_header(paths["test_csv"]):
        raise ValueError("Train and test feature headers differ.")
    x = np.load(paths["train_features"], mmap_mode="r")
    test = np.load(paths["test_features"], mmap_mode="r")
    train_ids = np.load(paths["train_ids"], mmap_mode="r")
    test_ids = np.load(paths["test_ids"], mmap_mode="r")
    signed = np.load(paths["train_labels"])
    expected_train, expected_test, expected_columns = EXPECTED_SHAPE
    if (x.shape != (expected_train, expected_columns) or test.shape != (expected_test, expected_columns)
            or len(names) != expected_columns or train_ids.shape != (expected_train,)
            or test_ids.shape != (expected_test,) or signed.shape != (expected_train,)):
        raise ValueError("Official raw/cache dataset dimensions do not align.")
    _csv_ids_match(train_csv, train_ids)
    _csv_ids_match(paths["test_csv"], test_ids)
    label_ids, raw_labels = _read_labels_csv(paths["labels_csv"])
    if not np.array_equal(label_ids, train_ids) or not np.array_equal(raw_labels, signed):
        raise ValueError("Training IDs, raw labels and cached labels differ.")
    if not np.array_equal(_read_sample_submission_ids(paths["sample_submission"]), test_ids):
        raise ValueError("Test IDs differ from official sample submission order.")
    if np.intersect1d(train_ids, test_ids).size:
        raise ValueError("Labeled and test respondent IDs overlap.")
    labels = _as_binary_labels(signed)
    if min(np.bincount(labels, minlength=2)) < 3:
        raise ValueError("Each label class needs at least three rows.")
    return paths, names, labels


def load_source(name, path, expected_hash):
    """Register before execution so dataclasses and pickle use stable module names."""
    if checksum(path) != expected_hash:
        raise ValueError(f"Frozen source changed: {path}.")
    previous = sys.modules.get(name)
    if previous is not None:
        if getattr(previous, "_frozen_sha256", None) != expected_hash:
            raise ValueError(f"Another source version is already loaded as {name}.")
        return previous
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    module._frozen_sha256 = expected_hash
    return module


def modules(manifest):
    return {key: load_source(spec["module"], spec["path"], spec["sha256"])
            for key, spec in manifest["sources"].items()}


def frozen(manifest, inputs=False):
    if environment() != manifest["environment"]:
        raise ValueError("Frozen Python, NumPy or numerical thread settings changed.")
    checks = dict(manifest["control_sha256"])
    checks.update({entry["path"]: entry["sha256"] for entry in manifest["sources"].values()})
    checks[manifest["folds_path"]] = manifest["folds_sha256"]
    if inputs:
        checks.update(manifest["input_sha256"])
    for path, expected in checks.items():
        if checksum(path) != expected:
            raise ValueError(f"A frozen input, configuration or source changed: {path}.")


def read_manifest(path):
    path = Path(path).resolve()
    if checksum(path) != path.with_suffix(".sha256").read_text(encoding="ascii").strip():
        raise ValueError("The prepared manifest was changed.")
    return read_json(path)


def prepare(root=ROOT, output=DEFAULT_OUTPUT):
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = read_manifest(manifest_path)
        frozen(manifest, inputs=True)
        print(f"PREPARED (reused): {manifest_path}", flush=True)
        return manifest
    suite_path = root / REFERENCE
    suite = read_json(suite_path)
    if suite["fold_count"] != 3 or len(suite["models"]) != 1 or len(suite["models"][0]["checkpoints"]) != 1:
        raise ValueError("Expected the selected single Phase 14 model with three folds.")
    paths, names, labels = _dataset(root, suite)
    preprocessing_path = root / suite["preprocessing_config_path"]
    plans = read_json(preprocessing_path)["variants"]
    plan = next(item["preprocessing"] for item in plans if item["experiment_name"] == suite["preprocessing_variant"])
    metadata_path = root / suite["feature_metadata_path"]
    historical = root / HISTORICAL
    if checksum(historical) != HISTORICAL_SHA256:
        raise ValueError("Historical Phase 14 implementation differs from the trusted snapshot.")
    if checksum(historical) == checksum(root / "src/numpy_boosting.py"):
        raise ValueError("Historical and corrected implementations must be distinct.")
    output.mkdir(parents=True, exist_ok=True)
    snapshots = output / "source_snapshot"
    snapshots.mkdir(exist_ok=True)
    source_paths = {"phase14": historical, "phase34": root / "src/numpy_boosting.py",
                    "preprocessing": root / "src/feature_preprocessing.py", "evaluation": root / "src/evaluation.py"}
    sources = {}
    for key, source in source_paths.items():
        destination = snapshots / f"{key}.py"
        shutil.copyfile(source, destination)
        sources[key] = {"path": str(destination), "module": MODULE_NAMES[key], "sha256": checksum(destination),
                        "original_path": str(source)}
    metadata = read_json(metadata_path)
    folds = full_train_folds(labels, suite["seed"])
    fold_ids = np.zeros(len(labels), dtype=np.int8)
    for index, (_, valid) in enumerate(folds, start=1):
        fold_ids[valid] = index
    folds_path = output / "full_train_fold_ids.npy"
    save_array(folds_path, fold_ids)
    parameter_source = suite["models"][0]
    parameters = {**parameter_source["parameters"], "n_estimators": parameter_source["checkpoints"][0]}
    controls = [suite_path, preprocessing_path, metadata_path, historical, root / "src/numpy_boosting.py",
                root / "src/feature_preprocessing.py", root / "src/evaluation.py", root / "src/run_preprocessing_cv.py",
                root / "run.py", root / "helpers.py", Path(__file__).resolve()]
    manifest = {"schema_version": 1, "root": str(root), "output": str(output),
                "prepared_at_utc": datetime.now(timezone.utc).isoformat(), "environment": environment(),
                "sources": sources, "control_sha256": {str(p): checksum(p) for p in controls},
                "input_sha256": {str(p): checksum(p) for p in paths.values()},
                "data": {key: str(path) for key, path in paths.items()}, "feature_names": names,
                "metadata": metadata, "preprocessing": plan, "preprocessing_variant": suite["preprocessing_variant"],
                "model_parameters": parameters, "fold_seed": suite["seed"], "fold_count": 3,
                "folds_path": str(folds_path), "folds_sha256": checksum(folds_path),
                "training_rows": len(labels), "test_rows": EXPECTED_SHAPE[1], "expected_features": EXPECTED_FEATURES,
                "positive_training_rows": int(labels.sum()), "checkpoint_interval": 25,
                "threshold_protocol": "Maximize F1 on pooled three-fold OOF probabilities covering every labeled row; break F1 ties by accuracy.",
                "calibration_warning": "OOF F1 uses the same labels to select and describe its threshold. It is not an unbiased test score.",
                "tasks": [{"variant": variant, "partition": f"fold_{fold}"} for variant in VARIANTS for fold in (1, 2, 3)]
                         + [{"variant": variant, "partition": "full"} for variant in VARIANTS]}
    write_json(output / "frozen_configuration.json", {"suite": suite, "preprocessing": plan, "sources": sources})
    manifest["control_sha256"][str(output / "frozen_configuration.json")] = checksum(output / "frozen_configuration.json")
    write_json(manifest_path, manifest)
    manifest_path.with_suffix(".sha256").write_text(checksum(manifest_path) + "\n", encoding="ascii")
    frozen(manifest, inputs=True)
    print(f"PREPARED six OOF fits and two full fits; no training: {manifest_path}", flush=True)
    return manifest


def task_directory(manifest, variant, partition):
    if variant not in VARIANTS or partition not in ("fold_1", "fold_2", "fold_3", "full"):
        raise ValueError("Unknown variant or partition.")
    return Path(manifest["output"]) / variant / partition


@contextmanager
def file_lock(path):
    """OS locks release automatically when an interrupted worker exits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def completed(manifest, variant, partition):
    directory = task_directory(manifest, variant, partition)
    path = directory / "completed.json"
    if not path.exists():
        return None
    result = read_json(path)
    if result["variant"] != variant or result["partition"] != partition:
        raise ValueError("Completed fit identifies another task.")
    for key in ("model", "probabilities"):
        if checksum(directory / result[key]["file"]) != result[key]["sha256"]:
            raise ValueError(f"Completed {key} artifact changed for {variant}/{partition}.")
    return result


def _matrices(manifest, partition, preprocessing):
    data = manifest["data"]
    x = np.load(data["train_features"], mmap_mode="r")
    labels = _as_binary_labels(np.load(data["train_labels"]))
    if partition == "full":
        train_rows = np.arange(len(labels), dtype=np.int64)
        valid_rows = np.arange(manifest["test_rows"], dtype=np.int64)
        train_source, valid_source = x, np.load(data["test_features"], mmap_mode="r")
    else:
        ids = np.load(manifest["folds_path"])
        fold = int(partition[-1])
        train_rows, valid_rows = np.flatnonzero(ids != fold), np.flatnonzero(ids == fold)
        train_source, valid_source = x[train_rows], x[valid_rows]
    train, valid, output_names = preprocessing.tree_feature_matrices(
        train_source, valid_source, manifest["feature_names"], manifest["metadata"], manifest["preprocessing"])
    if len(output_names) != manifest["expected_features"]:
        raise ValueError("The selected Phase 14 feature count changed.")
    signature = {"training_rows_sha256": array_checksum(train_rows), "validation_rows_sha256": array_checksum(valid_rows),
                 "training_matrix_sha256": array_checksum(train), "validation_matrix_sha256": array_checksum(valid),
                 "training_labels_sha256": array_checksum(labels[train_rows]), "output_feature_names": output_names,
                 "training_shape": list(train.shape), "validation_shape": list(valid.shape)}
    return train, labels[train_rows], valid, signature


def worker(manifest_path, variant, partition):
    manifest = read_manifest(manifest_path)
    frozen(manifest, inputs=True)
    directory = task_directory(manifest, variant, partition)
    with file_lock(directory / "worker.lock"):
        if completed(manifest, variant, partition) is not None:
            print(f"CACHED {variant}/{partition}", flush=True)
            return
        loaded = modules(manifest)
        train, labels, valid, signature = _matrices(manifest, partition, loaded["preprocessing"])
        model_class = loaded[variant].HistogramGradientBoostingClassifier
        parameters = manifest["model_parameters"]
        target = parameters["n_estimators"]
        metadata = {"variant": variant, "partition": partition, "signature": signature,
                    "parameters": parameters, "source_sha256": {key: value["sha256"] for key, value in manifest["sources"].items()},
                    "environment": manifest["environment"], "input_sha256": manifest["input_sha256"],
                    "folds_sha256": manifest["folds_sha256"]}
        checkpoint = directory / "model.pkl"
        started = time.perf_counter()
        steps = sorted(set(range(manifest["checkpoint_interval"], target + 1, manifest["checkpoint_interval"])) | {target})
        def progress(step, model):
            model.save_checkpoint(checkpoint, metadata)
            write_json(directory / "progress.json", {"variant": variant, "partition": partition, "trees": step,
                       "target": target, "elapsed_seconds": time.perf_counter() - started, "pid": os.getpid()})
            print(f"PROGRESS {variant}/{partition}: {step}/{target} trees, elapsed={time.perf_counter()-started:.1f}s", flush=True)
        prior = 0
        if checkpoint.exists():
            model, saved = model_class.load_checkpoint(checkpoint)
            if saved != metadata:
                raise ValueError("Resume checkpoint differs from frozen data, source or parameters.")
            prior = len(model.trees_)
            if not 0 < prior <= target or model.n_features_in_ != train.shape[1]:
                raise ValueError("Invalid resumed model dimensions.")
            for key, value in parameters.items():
                if getattr(model, key) != value:
                    raise ValueError(f"Resumed model parameter differs: {key}.")
            if prior < target:
                model.continue_fit(train, labels, target - prior, [step for step in steps if step > prior], progress)
        else:
            model = model_class(**parameters).fit(train, labels, steps, progress)
        probabilities = model.predict_proba(valid)[:, 1]
        if probabilities.shape != (valid.shape[0],) or not np.all(np.isfinite(probabilities)):
            raise ValueError("Model probabilities are not finite aligned vectors.")
        save_array(directory / "probabilities.npy", probabilities)
        frozen(manifest)
        result = {"variant": variant, "partition": partition, "signature": signature, "trees": len(model.trees_),
                  "resumed_from_trees": prior, "fit_and_prediction_seconds": time.perf_counter() - started,
                  "total_tree_seconds": float(np.sum(model.tree_fit_seconds_)),
                  "model": {"file": "model.pkl", "sha256": checksum(checkpoint)},
                  "probabilities": {"file": "probabilities.npy", "sha256": checksum(directory / "probabilities.npy")}}
        write_json(directory / "completed.json", result)
        print(f"DONE {variant}/{partition}", flush=True)


def calibrate(manifest, variant):
    loaded = modules(manifest)
    labels = _as_binary_labels(np.load(manifest["data"]["train_labels"]))
    fold_ids = np.load(manifest["folds_path"])
    probabilities = np.full(len(labels), np.nan)
    checkpoints = []
    for fold in (1, 2, 3):
        partition = f"fold_{fold}"
        result = completed(manifest, variant, partition)
        if result is None:
            raise ValueError(f"OOF fit is incomplete: {variant}/{partition}.")
        valid = np.flatnonzero(fold_ids == fold)
        scores = np.load(task_directory(manifest, variant, partition) / "probabilities.npy")
        if scores.shape != valid.shape:
            raise ValueError("OOF probabilities do not align with their held-out rows.")
        probabilities[valid] = scores
        checkpoints.append({"partition": partition, **result["model"]})
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("OOF probabilities do not cover all labeled rows.")
    evaluation = loaded["evaluation"]
    choice = evaluation.best_f1_threshold(labels, probabilities)
    counts, metrics = evaluation.classification_metrics(labels, evaluation.labels_from_scores(probabilities, choice.threshold))
    directory = Path(manifest["output"]) / variant
    oof_path = directory / "full_train_oof.npz"
    temporary = oof_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, row_indices=np.arange(len(labels)), train_ids=np.load(manifest["data"]["train_ids"]),
                            labels=labels, fold_ids=fold_ids, probabilities=probabilities,
                            decision_threshold=np.asarray([choice.threshold]))
    os.replace(temporary, oof_path)
    config = {"artifact_type": "frozen_full_train_calibrated_booster_configuration", "variant": variant,
              "reproduction_command": f'"{sys.executable}" -B -m tools.final_phase14_phase34 run --manifest "{Path(manifest["output"]) / "manifest.json"}" --workers 1',
              "model": manifest["model_parameters"], "implementation": manifest["sources"][variant],
              "preprocessing": {"plan": manifest["preprocessing"], "source": manifest["sources"]["preprocessing"],
                                "metadata": manifest["metadata"], "feature_names": manifest["feature_names"]},
              "decision_threshold": choice.threshold, "threshold_source": manifest["threshold_protocol"],
              "threshold_training_rows": len(labels), "folds_sha256": manifest["folds_sha256"],
              "oof_sha256": checksum(oof_path), "input_sha256": manifest["input_sha256"],
              "environment": manifest["environment"]}
    write_json(directory / "calibrated_final_config.json", config)
    calibration = {"variant": variant, "decision_threshold": choice.threshold, "training_rows": len(labels),
                   "protocol": manifest["threshold_protocol"], "calibration_metrics": metrics,
                   "calibration_confusion_counts": vars(counts), "interpretation": manifest["calibration_warning"],
                   "oof_artifact": {"path": str(oof_path), "sha256": checksum(oof_path)},
                   "configuration_sha256": checksum(directory / "calibrated_final_config.json"),
                   "fold_models": checkpoints, "official_test_f1": None}
    write_json(directory / "calibration.json", calibration)
    print(f"CALIBRATED {variant}: threshold={choice.threshold:.12g}, calibration F1={metrics['f1']:.6f}", flush=True)
    return calibration


def publish_submission(manifest, variant):
    directory = Path(manifest["output"]) / variant
    calibration = read_json(directory / "calibration.json")
    if checksum(directory / "calibrated_final_config.json") != calibration["configuration_sha256"]:
        raise ValueError("The calibrated final configuration changed.")
    config = read_json(directory / "calibrated_final_config.json")
    if (calibration["variant"] != variant or config["variant"] != variant
            or calibration["decision_threshold"] != config["decision_threshold"]
            or calibration["oof_artifact"]["sha256"] != config["oof_sha256"]
            or calibration["protocol"] != config["threshold_source"]
            or config["folds_sha256"] != manifest["folds_sha256"]
            or config["implementation"] != manifest["sources"][variant]
            or not np.isfinite(config["decision_threshold"])):
        raise ValueError("Calibration threshold or provenance differs from its saved configuration.")
    if checksum(calibration["oof_artifact"]["path"]) != calibration["oof_artifact"]["sha256"]:
        raise ValueError("The threshold-selection OOF artifact changed.")
    fit = completed(manifest, variant, "full")
    if fit is None:
        raise ValueError("Full-data fit is incomplete.")
    probabilities = np.load(task_directory(manifest, variant, "full") / "probabilities.npy")
    test_ids = np.load(manifest["data"]["test_ids"])
    if probabilities.shape != test_ids.shape or not np.all(np.isfinite(probabilities)):
        raise ValueError("Test probabilities must be finite and align with test IDs.")
    if not np.array_equal(test_ids, _read_sample_submission_ids(Path(manifest["data"]["sample_submission"]))):
        raise ValueError("Submission IDs no longer match the sample.")
    threshold = config["decision_threshold"]
    predictions = np.where(probabilities >= threshold, 1, -1).astype(np.int8)
    submission_path = directory / f"submission_{variant}_full_train_oof.csv"
    temporary = submission_path.with_suffix(".tmp")
    _write_submission(temporary, test_ids, predictions)
    os.replace(temporary, submission_path)
    result = {"variant": variant, "training_rows": manifest["training_rows"], "test_rows": len(test_ids),
              "decision_threshold": threshold, "positive_predictions": int(np.sum(predictions == 1)),
              "negative_predictions": int(np.sum(predictions == -1)),
              "submission": {"path": str(submission_path), "sha256": checksum(submission_path)},
              "full_model": fit["model"], "test_probabilities": fit["probabilities"], "official_test_f1": None}
    write_json(directory / "submission_summary.json", result)
    return {**calibration, **result}


def _launch(manifest_path, variant, partition):
    manifest = read_manifest(manifest_path)
    directory = task_directory(manifest, variant, partition)
    directory.mkdir(parents=True, exist_ok=True)
    print(f"START {variant}/{partition}", flush=True)
    with (directory / "worker.log").open("a", encoding="utf-8") as log:
        process = subprocess.run([sys.executable, "-B", "-m", "tools.final_phase14_phase34", "worker",
                                  "--manifest", str(manifest_path), "--variant", variant, "--partition", partition],
                                 cwd=manifest["root"], stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if process.returncode:
        raise RuntimeError(f"Worker failed: {variant}/{partition}; inspect {directory / 'worker.log'}.")
    print(f"COMPLETE {variant}/{partition}", flush=True)


def run(manifest_path, workers=1):
    if workers not in (1, 2):
        raise ValueError("Use one or at most two isolated single-thread workers.")
    manifest_path = Path(manifest_path).resolve()
    manifest = read_manifest(manifest_path)
    frozen(manifest, inputs=True)
    with file_lock(Path(manifest["output"]) / "run.lock"):
        for stage in ("oof", "full"):
            selected = [task for task in manifest["tasks"] if (task["partition"] == "full") == (stage == "full")]
            missing = [task for task in selected if completed(manifest, task["variant"], task["partition"]) is None]
            if workers == 1:
                for task in missing:
                    _launch(manifest_path, task["variant"], task["partition"])
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(_launch, manifest_path, task["variant"], task["partition"]) for task in missing]
                    try:
                        for future in concurrent.futures.as_completed(futures):
                            future.result()
                    except BaseException:
                        for future in futures:
                            future.cancel()
                        raise
            if stage == "oof":
                for variant in VARIANTS:
                    calibrate(manifest, variant)
        frozen(manifest, inputs=True)
        variants = {variant: publish_submission(manifest, variant) for variant in VARIANTS}
        a = np.load(task_directory(manifest, "phase14", "full") / "probabilities.npy")
        b = np.load(task_directory(manifest, "phase34", "full") / "probabilities.npy")
        comparison = {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
                      "scope": "Two fixed implementations; identical full-training folds, preprocessing and model parameters. Each threshold is selected from that implementation's pooled full-training OOF predictions.",
                      "calibration_warning": manifest["calibration_warning"], "variants": variants,
                      "test_prediction_disagreements": int(np.sum((a >= variants['phase14']['decision_threshold']) !=
                                                                 (b >= variants['phase34']['decision_threshold']))),
                      "official_test_f1_comparison": None,
                      "next_action": "User uploads both distinct CSVs to AIcrowd and records the official test F1 for each."}
        write_json(Path(manifest["output"]) / "comparison.json", comparison)
        print(json.dumps(comparison, indent=2), flush=True)
        return comparison


def status(manifest_path):
    manifest = read_manifest(manifest_path)
    tasks = []
    for task in manifest["tasks"]:
        directory = task_directory(manifest, task["variant"], task["partition"])
        finished = directory / "completed.json"
        progress = directory / "progress.json"
        state = {**task, "status": "completed" if finished.exists() else "pending"}
        if progress.exists():
            state.update(read_json(progress))
            if not finished.exists():
                state["status"] = "checkpoint_saved"
        tasks.append(state)
    return {"manifest": str(manifest_path), "tasks": tasks,
            "comparison_ready": (Path(manifest["output"]) / "comparison.json").exists()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status", "worker"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--partition", choices=("fold_1", "fold_2", "fold_3", "full"))
    args = parser.parse_args()
    manifest_path = args.manifest or args.output_dir / "manifest.json"
    if args.command == "prepare":
        prepare(ROOT, args.output_dir)
    elif args.command == "run":
        run(manifest_path, args.workers)
    elif args.command == "status":
        print(json.dumps(status(manifest_path), indent=2))
    else:
        if args.variant is None or args.partition is None:
            parser.error("worker requires --variant and --partition")
        worker(manifest_path, args.variant, args.partition)


if __name__ == "__main__":
    main()

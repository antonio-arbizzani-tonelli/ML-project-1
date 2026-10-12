"""Resume raw CV with independent fits, then assemble metrics in the standard runner.

The prepare action performs no training. Run only after the sequential process
has stopped: this driver and that process must not publish the same experiment.
"""

from __future__ import annotations

import os

for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
import concurrent.futures
import gc
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src import run_boosting_cv as runner
from src.feature_preprocessing import tree_feature_matrices
from src.numpy_boosting import HistogramGradientBoostingClassifier
from src.run_preprocessing_cv import _sha256, _stratified_folds
from tools import phase33_raw_preprocessing_benchmark as benchmark
from tools.targeted_preprocessing_analysis import _input_data


CACHE = "results/boosting_artifacts/phase33_raw_preprocessing/parallel_resume"
SOURCE_NAMES = (
    "tools/phase33_parallel_resume.py", "src/feature_preprocessing.py",
    "src/numpy_boosting.py", "src/run_boosting_cv.py", "src/evaluation.py",
    "src/run_preprocessing_cv.py",
)


def _json(path, value=None):
    if value is None:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sources():
    code_root = Path(__file__).resolve().parents[1]
    return {str(code_root / name): _sha256(code_root / name) for name in SOURCE_NAMES}


def _inputs(root, suite_path, suite):
    paths = {"boosting_suite_config": suite_path,
             "preprocessing_config": root / suite["preprocessing_config_path"],
             "feature_metadata": root / suite["feature_metadata_path"]}
    paths.update({key: root / suite["data"][field] for key, field in (
        ("features", "features_path"), ("labels", "labels_path"),
        ("split_indices", "split_indices_path"))})
    return {key: _sha256(path) for key, path in paths.items()}


def _frozen(manifest, check_inputs=False):
    if _sources() != manifest["source_sha256"]:
        raise ValueError("A frozen numerical source or resume driver changed.")
    if check_inputs and _inputs(Path(manifest["root"]), Path(manifest["suite_path"]),
                                manifest["suite"]) != manifest["input_sha256"]:
        raise ValueError("A frozen suite or dataset input changed.")


def _signature(train, labels, valid, parameters, checkpoints):
    return {"train_shape": list(train.shape), "validation_shape": list(valid.shape),
            "train_sha256": runner._array_sha256(train),
            "validation_sha256": runner._array_sha256(valid),
            "labels_sha256": runner._array_sha256(labels),
            "parameters": parameters, "checkpoints": checkpoints}


def _task_arrays(manifest, task):
    root = Path(manifest["root"])
    row_path = Path(manifest["cache_dir"]) / task["rows_file"]
    if _sha256(row_path) != task["rows_sha256"]:
        raise ValueError("Saved partition rows changed.")
    with np.load(row_path) as rows:
        train_rows, valid_rows = rows["training_rows"], rows["validation_rows"]
    x = np.load(root / manifest["suite"]["data"]["features_path"], mmap_mode="r")
    labels = runner._as_binary_labels(np.load(root / manifest["suite"]["data"]["labels_path"]))
    train, valid, names = tree_feature_matrices(
        x[train_rows], x[valid_rows], manifest["feature_names"], [], manifest["plan"])
    if names != manifest["feature_names"]:
        raise ValueError("Raw feature names changed.")
    return train, labels[train_rows], valid, train_rows


def _model_valid(model, task):
    signature = task["signature"]
    if (len(model.trees_) != signature["checkpoints"][-1]
            or model.n_features_in_ != signature["train_shape"][1]):
        raise ValueError("Cached model has the wrong tree or feature count.")
    for name, value in signature["parameters"].items():
        actual = getattr(model, name)
        if isinstance(actual, tuple):
            actual = list(actual)
        if actual != value:
            raise ValueError(f"Cached model parameter changed: {name}")


def _cache_files(manifest, task):
    directory = Path(manifest["cache_dir"]) / task["name"]
    return directory, directory / "model.pkl", directory / "probabilities.npy", directory / "fit.json"


def _cached(manifest, task):
    _, checkpoint, probabilities, info_path = _cache_files(manifest, task)
    if not info_path.exists():
        return None
    info = _json(info_path)
    if (info["signature"] != task["signature"]
            or info["source_sha256"] != manifest["source_sha256"]
            or _sha256(checkpoint) != info["checkpoint_sha256"]
            or _sha256(probabilities) != info["probabilities_sha256"]):
        raise ValueError(f"Invalid completed cache: {task['name']}")
    model, _ = HistogramGradientBoostingClassifier.load_checkpoint(checkpoint)
    _model_valid(model, task)
    scores = np.load(probabilities)
    if (scores.shape != (task["signature"]["validation_shape"][0],)
            or scores.dtype != np.float64 or not np.all(np.isfinite(scores))):
        raise ValueError("Cached validation probabilities are invalid.")
    return model, scores, info


def _store(manifest, task, probabilities, profile, provenance):
    _, checkpoint, score_path, info_path = _cache_files(manifest, task)
    temporary = score_path.with_suffix(".tmp.npy")
    np.save(temporary, probabilities)
    os.replace(temporary, score_path)
    info = {"task": task["name"], "signature": task["signature"],
            "source_sha256": manifest["source_sha256"],
            "checkpoint_sha256": _sha256(checkpoint),
            "probabilities_sha256": _sha256(score_path), "profile": profile,
            "provenance": provenance}
    _json(info_path, info)
    return info


def _reuse_outer(manifest, task, source, train_rows, labels, valid):
    """Accept only a complete standard raw checkpoint with exact training metadata."""
    model, metadata = HistogramGradientBoostingClassifier.load_checkpoint(source)
    expected = {"outer_fold": task["outer_fold"],
                "fitted_tree_count": task["signature"]["checkpoints"][-1],
                "model_name": manifest["suite"]["models"][0]["name"],
                "output_feature_names": manifest["feature_names"],
                "resolved_model_parameters": task["signature"]["parameters"],
                "input_sha256": manifest["input_sha256"],
                "training_labels_sha256": runner._array_sha256(labels),
                "training_row_indices_sha256": runner._array_sha256(train_rows)}
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(f"Sequential checkpoint differs for {name}.")
    np.testing.assert_array_equal(metadata["training_row_indices"], train_rows)
    _model_valid(model, task)
    directory, checkpoint, _, _ = _cache_files(manifest, task)
    directory.mkdir(parents=True, exist_ok=True)
    checksum = _sha256(source)
    shutil.copyfile(source, checkpoint)
    if _sha256(checkpoint) != checksum:
        raise ValueError("Sequential checkpoint changed while being copied.")
    prediction_started = time.perf_counter()
    probabilities = dict(model.staged_predict_proba(valid, task["signature"]["checkpoints"]))
    prediction_seconds = time.perf_counter() - prediction_started
    profile = runner._tree_profile(model.tree_fit_seconds_)
    logged_elapsed = None
    log_path = source.parents[1] / "run_stdout.log"
    if log_path.exists():
        matches = re.findall(rf"DONE fit={task['outer_fold']} stage=outer elapsed=([0-9.]+)s",
                             log_path.read_text(encoding="utf-8"))
        if matches:
            logged_elapsed = float(matches[-1])
    profile.update(fit_and_validation_prediction_seconds=(
        logged_elapsed if logged_elapsed is not None else profile["total_seconds"]),
        bin_count_minimum=int(model.binner_.bin_counts_.min()),
        bin_count_maximum=int(model.binner_.bin_counts_.max()))
    return _store(manifest, task, probabilities[task["signature"]["checkpoints"][-1]], profile,
                  {"kind": "reused_sequential_outer", "source": str(source),
                   "source_sha256": checksum, "historical_elapsed_seconds": logged_elapsed,
                   "elapsed_basis": "Original log rounded to 0.1s; otherwise wall time unavailable.",
                   "cache_prediction_seconds": prediction_seconds})


def prepare(root, suite_path, cache_dir):
    """Precompute the standard runner's ordered partitions and reuse completed outer fits."""
    started = time.perf_counter()
    root, suite_path, cache_dir = Path(root).resolve(), Path(suite_path).resolve(), Path(cache_dir).resolve()
    suite = _json(suite_path)
    runner._validate_suite(suite)
    model = suite["models"][0]
    if len(suite["models"]) != 1 or len(model["checkpoints"]) != 1:
        raise ValueError("Resume supports exactly one model and one checkpoint.")
    if suite["nested_threshold"]["model_checkpoints"] != [
            {"model": model["name"], "checkpoints": model["checkpoints"]}]:
        raise ValueError("Outer and inner checkpoints must match exactly.")
    preprocessing = _json(root / suite["preprocessing_config_path"])
    plan = runner._variant(preprocessing, suite["preprocessing_variant"])["preprocessing"]
    if plan != {"name": "raw_tree_passthrough"}:
        raise ValueError("Resume is restricted to the unchanged raw representation.")
    x, y, development, names, _ = _input_data(root, suite)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"root": str(root), "suite_path": str(suite_path), "suite": suite,
                "cache_dir": str(cache_dir), "plan": plan, "feature_names": names,
                "source_sha256": _sources(), "input_sha256": _inputs(root, suite_path, suite),
                "tasks": []}
    outer = list(_stratified_folds(y[development], suite["fold_count"], suite["seed"]))
    partitions = [(f"outer_{fold}", fold, train, valid) for fold, (train, valid) in enumerate(outer, 1)]
    for fold, (outer_train, _) in enumerate(outer, 1):
        for inner, (train, valid) in enumerate(_stratified_folds(
                y[development[outer_train]], suite["nested_threshold"]["inner_fold_count"],
                suite["nested_threshold"]["inner_seed"] + fold), 1):
            partitions.append((f"outer_{fold}_inner_{inner}", fold, outer_train[train], outer_train[valid]))
    for name, fold, train, valid in partitions:
        rows_path = cache_dir / f"{name}_rows.npz"
        train_rows, valid_rows = development[train], development[valid]
        if rows_path.exists():
            with np.load(rows_path) as existing:
                np.testing.assert_array_equal(existing["training_rows"], train_rows)
                np.testing.assert_array_equal(existing["validation_rows"], valid_rows)
        else:
            np.savez_compressed(rows_path, training_rows=train_rows, validation_rows=valid_rows)
        task = {"name": name, "outer_fold": fold, "rows_file": rows_path.name,
                "rows_sha256": _sha256(rows_path)}
        train_matrix, valid_matrix, _ = tree_feature_matrices(x[train_rows], x[valid_rows], names, [], plan)
        task["signature"] = _signature(train_matrix, y[train_rows], valid_matrix,
            runner._resolved_parameters(model["parameters"], names), model["checkpoints"])
        manifest["tasks"].append(task)
        if _cached(manifest, task) is None and "_inner_" not in name:
            source = root / suite["artifact_dir"] / "model_checkpoints" / (
                f"{suite['experiment_prefix']}_{model['name']}_fold{fold}_{model['checkpoints'][0]}trees.pkl")
            if source.is_file():
                _reuse_outer(manifest, task, source, train_rows, y[train_rows], valid_matrix)
        del train_matrix, valid_matrix
        gc.collect()
    manifest["prepare_elapsed_seconds"] = time.perf_counter() - started
    _frozen(manifest, check_inputs=True)
    _json(cache_dir / "manifest.json", manifest)
    shutil.copyfile(Path(__file__), cache_dir / "executed_parallel_resume.py")
    complete = [task["name"] for task in manifest["tasks"] if _cached(manifest, task) is not None]
    print(f"PREPARED {len(manifest['tasks'])} fits; cached={complete}; no new training.", flush=True)
    return manifest


def _train_task(manifest_path, task_name):
    manifest = _json(manifest_path)
    _frozen(manifest)
    task = next(task for task in manifest["tasks"] if task["name"] == task_name)
    if _cached(manifest, task) is not None:
        return task_name
    train, labels, valid, rows = _task_arrays(manifest, task)
    signature = _signature(train, labels, valid, task["signature"]["parameters"], task["signature"]["checkpoints"])
    if signature != task["signature"]:
        raise ValueError("A worker's exact training or validation data changed.")
    directory, checkpoint, _, _ = _cache_files(manifest, task)
    directory.mkdir(parents=True, exist_ok=True)
    print(f"START {task_name}: train={len(labels)}; pid={os.getpid()}", flush=True)

    def save(step, fitted):
        fitted.save_checkpoint(checkpoint, {"task": task_name, "signature": signature,
                                           "training_row_indices": rows,
                                           "source_sha256": manifest["source_sha256"]})

    scores, profile = runner._fit_checkpoint_probabilities(
        train, labels, valid, signature["parameters"], signature["checkpoints"], save)
    _frozen(manifest)
    _store(manifest, task, scores[signature["checkpoints"][-1]], profile,
           {"kind": "new_parallel_fit", "pid": os.getpid(),
            "elapsed_seconds": profile["fit_and_validation_prediction_seconds"]})
    print(f"DONE {task_name}: {profile['fit_and_validation_prediction_seconds']:.1f}s", flush=True)
    return task_name


def train_missing(manifest, workers):
    """Use isolated Python processes without Windows multiprocessing named pipes."""
    missing = [task["name"] for task in manifest["tasks"] if _cached(manifest, task) is None]
    manifest_path = Path(manifest["cache_dir"]) / "manifest.json"
    def launch(name):
        directory = Path(manifest["cache_dir"]) / name
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / "worker_stdout.log"
        command = [sys.executable, "-B", "-m", "tools.phase33_parallel_resume", "worker",
                   "--manifest", str(manifest_path), "--task", name]
        print(f"START isolated worker {name}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            raise RuntimeError(f"Worker {name} failed; inspect {log_path}.")
        print(f"COMPLETE isolated worker {name}", flush=True)
        return name

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(launch, name) for name in missing]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    return missing


def assemble(manifest, output_dir, *, resume_started=None, workers=1):
    """Run unchanged CV orchestration with exact cached probabilities and outer models."""
    _frozen(manifest, check_inputs=True)
    root = Path(manifest["root"])
    output_dir = Path(output_dir)
    if list(output_dir.glob("*.json")):
        raise ValueError("A completed raw record already exists; do not publish a duplicate.")
    original_fit, original_save = runner._fit_checkpoint_probabilities, runner.save_experiment
    used = []

    def cached_fit(train, labels, valid, parameters, checkpoints, callback=None):
        task = manifest["tasks"][len(used)]
        if _signature(train, labels, valid, parameters, checkpoints) != task["signature"]:
            raise ValueError("Standard runner partition/order/parameters do not match the cache.")
        cached = _cached(manifest, task)
        if cached is None:
            raise ValueError("A required fit cache is incomplete.")
        model, scores, info = cached
        np.testing.assert_array_equal(dict(model.staged_predict_proba(valid, checkpoints))[checkpoints[-1]], scores)
        if callback is not None:
            callback(checkpoints[-1], model)
        used.append(info)
        return {checkpoints[-1]: scores}, info["profile"]

    def save(config, outcome, directory):
        if len(used) != len(manifest["tasks"]):
            raise ValueError("Not every precomputed partition was consumed exactly once.")
        _frozen(manifest, check_inputs=True)
        config["hypothesis"] = manifest["suite"].get("hypothesis", config["hypothesis"])
        config["missing_values"]["method"] = "All raw float32 CSV predictors and NaNs unchanged; no semantic cleaning or exclusions."
        elapsed = None if resume_started is None else time.perf_counter() - resume_started
        outcome["parallel_resume"] = {"manifest": str(Path(manifest["cache_dir"]) / "manifest.json"),
            "source_sha256": manifest["source_sha256"], "fit_caches": used,
            "workers": workers, "resume_elapsed_seconds": elapsed,
            "metric_assembly_seconds": outcome["runtime_seconds"],
            "runtime_scope": "Resume wall time; excludes earlier sequential reused training. Per-fit profiles distinguish historical and new work.",
            "completed_at_utc": datetime.now(timezone.utc).isoformat()}
        if elapsed is not None:
            outcome["runtime_seconds"] = elapsed
            outcome["runtime_basis"] = outcome["parallel_resume"]["runtime_scope"]
        outcome["environment"] = {"python": platform.python_version(), "numpy": np.__version__,
            "threads": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
        outcome["source_sha256"] = manifest["source_sha256"]
        audit_path = root / benchmark.PREFLIGHT
        if audit_path.exists():
            outcome["preprocessing_benchmark"] = _json(audit_path)
        outcome["memory_estimate"]["basis"] = "Largest raw float32 train/validation pair in one process; excludes worker concurrency and temporary arrays."
        return original_save(config, outcome, directory)

    runner._fit_checkpoint_probabilities, runner.save_experiment = cached_fit, save
    try:
        return runner.run_suite(Path(manifest["suite_path"]), root, output_dir)
    finally:
        runner._fit_checkpoint_probabilities, runner.save_experiment = original_fit, original_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "worker"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--task")
    args = parser.parse_args()
    if args.action == "worker":
        if args.manifest is None or args.task is None:
            raise ValueError("An internal worker requires its manifest and task name.")
        _train_task(args.manifest, args.task)
        return
    if not 1 <= args.workers <= 3:
        raise ValueError("Use one to three single-thread workers.")
    root = Path.cwd()
    records = root / benchmark.RECORDS
    if list(records.glob("*.json")):
        raise ValueError("A completed raw record already exists; use the comparison helper.")
    audit = _json(root / benchmark.PREFLIGHT)
    for name, checksum in audit["source_sha256"].items():
        if _sha256(root / name) != checksum:
            raise ValueError(f"Original sequential preflight source/config changed: {name}")
    suite = _json(root / benchmark.SUITE)
    inputs = _inputs(root, root / benchmark.SUITE, suite)
    for name, checksum in audit["input_sha256"].items():
        if inputs[name] != checksum:
            raise ValueError(f"Original sequential preflight dataset changed: {name}")
    started = time.perf_counter()
    manifest = prepare(root, root / benchmark.SUITE, root / CACHE)
    if args.action == "run":
        train_missing(manifest, args.workers)
        record = assemble(manifest, records, resume_started=started, workers=args.workers)[0]
        benchmark.compare(root, record.relative_to(root))


if __name__ == "__main__":
    main()

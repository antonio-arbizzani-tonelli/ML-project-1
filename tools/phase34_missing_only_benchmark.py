"""Controlled nested-CV evaluation of the corrected missing-only split search.

Prepare freezes nine ordered fits without training. Run uses at most two
isolated single-thread workers and resumes validated 25-tree checkpoints.
The unchanged standard CV runner assembles the cached models and scores.
Historical Phase 14 artifacts and the final submission are never replaced.
"""

from __future__ import annotations

import os

THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
for _thread_name in THREADS:
    os.environ[_thread_name] = "1"

import argparse
import concurrent.futures
import gc
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src import run_boosting_cv as runner
from src.evaluation import average_precision, binary_log_loss, classification_metrics
from src.feature_preprocessing import tree_feature_matrices
from src.numpy_boosting import HistogramGradientBoostingClassifier
from src.run_preprocessing_cv import _sha256, _stratified_folds
from src.summarize_phase19_transfer import BASELINE, _arrays, _paired_interval
from tools.phase33_parallel_resume import _cache_files, _cached, _json, _signature, _store
from tools.targeted_preprocessing_analysis import _input_data

SUITE = "configs/experiments/phase34_missing_only_split.json"
ARTIFACTS = "results/boosting_artifacts/phase34_missing_only_split"
CACHE = ARTIFACTS + "/fit_cache"
RECORDS = "results/experiments/phase34_missing_only_split"
COMPARISON = "results/experiments/phase34_missing_only_split_comparison.json"
SUMMARY = "results/eda/phase34_missing_only_split_summary.md"
BEFORE = ARTIFACTS + "/source_snapshot/src/numpy_boosting_before_fix.py"
SOURCE_NAMES = (
    "tools/phase34_missing_only_benchmark.py", "tools/phase33_parallel_resume.py",
    "tools/targeted_preprocessing_analysis.py", "src/numpy_boosting.py",
    "src/feature_preprocessing.py", "src/run_boosting_cv.py",
    "src/run_preprocessing_cv.py", "src/evaluation.py", "src/summarize_phase19_transfer.py",
)


def _sources(root):
    return {name: _sha256(root / name) for name in SOURCE_NAMES}


def _environment():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "threads": {name: os.environ.get(name) for name in THREADS}}


def _inputs(root, suite):
    paths = {"boosting_suite_config": root / SUITE,
             "preprocessing_config": root / suite["preprocessing_config_path"],
             "feature_metadata": root / suite["feature_metadata_path"]}
    paths.update({key: root / suite["data"][field] for key, field in (
        ("features", "features_path"), ("labels", "labels_path"),
        ("split_indices", "split_indices_path"))})
    return {key: _sha256(path) for key, path in paths.items()}


def _frozen(manifest, check_inputs=False):
    root = Path(manifest["root"])
    if _environment() != manifest["environment"]:
        raise ValueError("The frozen Python/NumPy versions or numerical thread settings changed.")
    if _sources(root) != manifest["source_sha256"]:
        raise ValueError("The frozen benchmark or numerical sources changed.")
    if check_inputs and _inputs(root, manifest["suite"]) != manifest["input_sha256"]:
        raise ValueError("The frozen configuration or dataset changed.")
    for name, checksum in manifest["control_sha256"].items():
        if _sha256(root / name) != checksum:
            raise ValueError(f"Historical control changed: {name}")


def _partitions(labels, development, suite):
    outer = list(_stratified_folds(labels[development], 3, suite["seed"]))
    for fold, (train, valid) in enumerate(outer, 1):
        yield f"outer_{fold}", fold, development[train], development[valid]
    for fold, (train, _) in enumerate(outer, 1):
        for inner, (itrain, ivalid) in enumerate(_stratified_folds(
                labels[development[train]], 2,
                suite["nested_threshold"]["inner_seed"] + fold), 1):
            yield f"outer_{fold}_inner_{inner}", fold, development[train[itrain]], development[train[ivalid]]


def _task_arrays(manifest, task):
    root = Path(manifest["root"])
    row_path = Path(manifest["cache_dir"]) / task["rows_file"]
    if _sha256(row_path) != task["rows_sha256"]:
        raise ValueError("A frozen partition changed.")
    with np.load(row_path) as rows:
        train_rows, valid_rows = rows["training_rows"], rows["validation_rows"]
    x, labels, _, names, metadata = _input_data(root, manifest["suite"])
    train, valid, output = tree_feature_matrices(
        x[train_rows], x[valid_rows], names, metadata, manifest["plan"])
    if output != task["output_names"]:
        raise ValueError("The cleaned representation changed.")
    return train, labels[train_rows], valid, train_rows, labels[valid_rows]


def prepare(root):
    """Freeze input matrices, rows and sources after the split fix is ready."""
    root = Path(root).resolve()
    cache = root / CACHE
    existing = cache / "manifest.json"
    if existing.exists():
        manifest = _json(existing)
        _frozen(manifest, check_inputs=True)
        print("Prepared manifest already exists and remains valid; no training.", flush=True)
        return manifest
    suite, reference = _json(root / SUITE), _json(root / "configs/experiments/phase14_boosting_codebook_corrections_only.json")
    expected = dict(reference)
    expected.update(experiment_prefix=suite["experiment_prefix"], artifact_dir=ARTIFACTS)
    if {k: v for k, v in suite.items() if k != "hypothesis"} != expected:
        raise ValueError("Phase 34 must change only provenance and the corrected implementation.")
    runner._validate_suite(suite)
    if _sha256(root / BEFORE) == _sha256(root / "src/numpy_boosting.py"):
        raise ValueError("The booster fix is not active; prepare after its tests pass.")
    baseline = _json(root / BASELINE)
    base_arrays = _arrays(root, baseline)
    inputs = _inputs(root, suite)
    for key, value in inputs.items():
        if key != "boosting_suite_config" and value != baseline["outcome"]["input_sha256"][key]:
            raise ValueError(f"Phase 14 input changed: {key}")
    plan = runner._variant(_json(root / suite["preprocessing_config_path"]), suite["preprocessing_variant"])["preprocessing"]
    x, labels, development, names, metadata = _input_data(root, suite)
    np.testing.assert_array_equal(development, base_arrays["development_indices"])
    np.testing.assert_array_equal(labels[development], base_arrays["labels"])
    expected_folds = np.empty(development.size, dtype=np.int8)
    for fold, (_, valid) in enumerate(_stratified_folds(labels[development], 3, suite["seed"]), 1):
        expected_folds[valid] = fold
    np.testing.assert_array_equal(expected_folds, base_arrays["fold_ids"])
    controls = [BASELINE, BEFORE, baseline["outcome"]["oof_predictions_artifact"]["path"]]
    controls += [entry["path"] for entry in baseline["outcome"]["model_checkpoints"]]
    cache.mkdir(parents=True, exist_ok=True)
    manifest = {"root": str(root), "suite_path": str(root / SUITE), "suite": suite,
                "cache_dir": str(cache), "plan": plan, "source_sha256": _sources(root),
                "environment": _environment(),
                "input_sha256": inputs, "control_sha256": {p: _sha256(root / p) for p in controls},
                "prepared_at_utc": datetime.now(timezone.utc).isoformat(), "tasks": []}
    model = suite["models"][0]
    for name, fold, train_rows, valid_rows in _partitions(labels, development, suite):
        rows = cache / f"{name}_rows.npz"
        np.savez_compressed(rows, training_rows=train_rows, validation_rows=valid_rows)
        train, valid, output = tree_feature_matrices(x[train_rows], x[valid_rows], names, metadata, plan)
        if len(output) != 295:
            raise ValueError("Phase 34 must retain the same 295 cleaned columns.")
        task = {"name": name, "outer_fold": fold, "rows_file": rows.name,
                "rows_sha256": _sha256(rows), "output_names": output,
                "signature": _signature(train, labels[train_rows], valid,
                    runner._resolved_parameters(model["parameters"], output), model["checkpoints"])}
        manifest["tasks"].append(task)
        print(f"PREPARE {name}: train={len(train_rows)} valid={len(valid_rows)} features={len(output)}", flush=True)
        del train, valid
        gc.collect()
    _frozen(manifest, check_inputs=True)
    _json(existing, manifest)
    snapshot = root / ARTIFACTS / "source_snapshot"
    for name in SOURCE_NAMES:
        destination = snapshot / name
        if name == "src/numpy_boosting.py":
            destination = snapshot / "src/numpy_boosting_after_fix.py"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, destination)
    print("PREPARED nine fresh fits; no model trained.", flush=True)
    return manifest


def _worker(manifest_path, task_name):
    manifest = _json(manifest_path)
    _frozen(manifest, check_inputs=True)
    task = next(t for t in manifest["tasks"] if t["name"] == task_name)
    if _cached(manifest, task) is not None:
        print(f"CACHED {task_name}", flush=True)
        return
    train, labels, valid, rows, _ = _task_arrays(manifest, task)
    signature = task["signature"]
    if _signature(train, labels, valid, signature["parameters"], signature["checkpoints"]) != signature:
        raise ValueError("Worker matrices, labels or parameters changed.")
    directory, checkpoint, _, _ = _cache_files(manifest, task)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    steps = list(range(25, 201, 25))
    metadata = {"task": task_name, "signature": signature,
                "training_row_indices": rows, "source_sha256": manifest["source_sha256"]}

    def progress(step, model):
        model.save_checkpoint(checkpoint, metadata)
        print(f"PROGRESS {task_name}: {step}/200 trees; elapsed={time.perf_counter()-started:.1f}s", flush=True)

    fitted_before = 0
    if checkpoint.exists():
        model, saved = HistogramGradientBoostingClassifier.load_checkpoint(checkpoint)
        for key in ("task", "signature", "source_sha256"):
            if saved.get(key) != metadata[key]:
                raise ValueError(f"Partial checkpoint metadata changed: {key}")
        np.testing.assert_array_equal(saved["training_row_indices"], rows)
        for key, value in signature["parameters"].items():
            if getattr(model, key) != value:
                raise ValueError(f"Partial model parameter changed: {key}")
        fitted_before = len(model.trees_)
        if not 0 < fitted_before <= 200 or model.n_features_in_ != train.shape[1]:
            raise ValueError("Partial checkpoint has an invalid size.")
        if fitted_before < 200:
            model.continue_fit(train, labels, 200-fitted_before,
                [step for step in steps if step > fitted_before], progress)
    else:
        model = HistogramGradientBoostingClassifier(**signature["parameters"], n_estimators=200)
        model.fit(train, labels, steps, progress)
    scores = model.predict_proba(valid)[:, 1]
    profile = runner._tree_profile(model.tree_fit_seconds_)
    profile.update(fit_and_validation_prediction_seconds=time.perf_counter()-started,
                   bin_count_minimum=int(model.binner_.bin_counts_.min()),
                   bin_count_maximum=int(model.binner_.bin_counts_.max()))
    _frozen(manifest)
    _store(manifest, task, scores, profile, {"kind": "fresh_corrected_fit",
        "resumed_from_tree_count": fitted_before, "pid": os.getpid(),
        "elapsed_scope": "Current worker invocation; saved tree timings include any completed partial fitting."})
    print(f"DONE {task_name}: {profile['fit_and_validation_prediction_seconds']:.1f}s", flush=True)


def train_missing(manifest, workers):
    """Launch isolated processes, each with one BLAS thread."""
    missing = [t["name"] for t in manifest["tasks"] if _cached(manifest, t) is None]
    def launch(name):
        directory = Path(manifest["cache_dir"]) / name
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / "worker_stdout.log"
        print(f"START {name}", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, "-B", "-m", "tools.phase34_missing_only_benchmark",
                "worker", "--manifest", str(Path(manifest["cache_dir"])/"manifest.json"), "--task", name],
                cwd=manifest["root"], stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            raise RuntimeError(f"Worker {name} failed; inspect {log_path}.")
        print(f"COMPLETE {name}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(launch, name) for name in missing]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def assemble(manifest, started, workers):
    """Use unchanged run_suite after validating every cached fit signature."""
    _frozen(manifest, check_inputs=True)
    root = Path(manifest["root"])
    if list((root / RECORDS).glob("*.json")):
        raise ValueError("A completed Phase 34 record already exists.")
    original_fit, original_save = runner._fit_checkpoint_probabilities, runner.save_experiment
    used = []
    def cached_fit(train, labels, valid, parameters, checkpoints, callback=None):
        task = manifest["tasks"][len(used)]
        if _signature(train, labels, valid, parameters, checkpoints) != task["signature"]:
            raise ValueError("Standard runner partition, matrix or parameter order differs.")
        cached = _cached(manifest, task)
        if cached is None:
            raise ValueError("A required fit remains incomplete.")
        model, scores, info = cached
        np.testing.assert_array_equal(model.predict_proba(valid)[:, 1], scores)
        if callback is not None:
            callback(checkpoints[-1], model)
        used.append(info)
        return {checkpoints[-1]: scores}, info["profile"]
    def save(config, outcome, directory):
        if len(used) != 9:
            raise ValueError("All nine fits must be consumed exactly once.")
        _frozen(manifest, check_inputs=True)
        config["hypothesis"] = manifest["suite"]["hypothesis"]
        outcome["source_sha256"] = manifest["source_sha256"]
        outcome["environment"] = _environment()
        outcome["controlled_missing_only_fix"] = {"control_record": BASELINE,
            "manifest": str(Path(manifest["cache_dir"])/"manifest.json"), "fit_caches": used,
            "workers": workers, "metric_assembly_seconds": outcome["runtime_seconds"],
            "control_sha256": manifest["control_sha256"]}
        outcome["runtime_seconds"] = time.perf_counter()-started
        outcome["runtime_basis"] = "Current run wall time, including cached metric assembly; excludes earlier interrupted fitting and preparation."
        return original_save(config, outcome, directory)
    runner._fit_checkpoint_probabilities, runner.save_experiment = cached_fit, save
    try:
        return runner.run_suite(root / SUITE, root, root / RECORDS)[0]
    finally:
        runner._fit_checkpoint_probabilities, runner.save_experiment = original_fit, original_save


def _pure_missing_nodes(model, features):
    """Count selected nodes whose actual training partition separates NaNs."""
    binned = model.binner_.transform(features)
    total, endpoint, by_feature = 0, 0, {}
    def visit(node, rows):
        nonlocal total, endpoint
        if node.feature is None:
            return
        if node.split_bin == int(model.binner_.bin_counts_[node.feature])-1:
            endpoint += 1
        values = binned[rows, node.feature]
        missing = values == 255
        known_left = values[~missing] <= node.split_bin
        if np.any(missing) and known_left.size and (
                (np.all(known_left) and not node.missing_go_left)
                or (not np.any(known_left) and node.missing_go_left)):
            total += 1
            key = str(node.feature)
            by_feature[key] = by_feature.get(key, 0)+1
        left = values <= node.split_bin
        left[missing] = node.missing_go_left
        visit(node.left, rows[left])
        visit(node.right, rows[~left])
    for tree in model.trees_:
        visit(tree.root_, np.arange(features.shape[0]))
    return {"selected_nodes": total, "global_endpoint_nodes": endpoint,
            "by_feature_index": by_feature, "training_rows": features.shape[0],
            "definition": "On the complete ordered training data reaching a selected node, all finite values go to one child and all NaNs to the other. Includes lower thresholds equivalent to a global-endpoint split within that node."}


def _sample_staged(model, features, labels, sample_indices):
    return [{"trees": step, "log_loss": binary_log_loss(labels[sample_indices], scores),
             "average_precision": average_precision(labels[sample_indices], scores)}
            for step, scores in model.staged_predict_proba(features[sample_indices], [50, 100, 150, 200])]


def compare(root, record_path):
    """Reconstruct paired metrics and inexpensive saved-model diagnostics."""
    root = Path(root)
    manifest = _json(root / CACHE / "manifest.json")
    _frozen(manifest, check_inputs=True)
    baseline, candidate = _json(root / BASELINE), _json(root / record_path)
    ba, ca = _arrays(root, baseline), _arrays(root, candidate)
    for key in ("development_indices", "fold_ids", "labels"):
        np.testing.assert_array_equal(ba[key], ca[key])
    if baseline["config"]["model"] != candidate["config"]["model"]:
        raise ValueError("The controlled comparison changed model parameters.")
    for record, arrays in ((baseline, ba), (candidate, ca)):
        np.testing.assert_array_equal(arrays["nested_predictions"], arrays["probabilities"] >= arrays["nested_thresholds"])
        _, metrics = classification_metrics(arrays["labels"], arrays["nested_predictions"])
        if abs(metrics["f1"]-record["outcome"]["nested_threshold_evaluation"]["pooled_metrics"]["f1"]) > 1e-14:
            raise ValueError("Recorded F1 does not match saved decisions.")
    bn, cn = baseline["outcome"]["nested_threshold_evaluation"], candidate["outcome"]["nested_threshold_evaluation"]
    bm, cm = bn["pooled_metrics"], cn["pooled_metrics"]
    old, new, positive = ba["nested_predictions"], ca["nested_predictions"], ba["labels"] == 1
    transitions = {"recovered_positives": int(np.sum(positive & (old == 0) & (new == 1))),
        "lost_positives": int(np.sum(positive & (old == 1) & (new == 0))),
        "introduced_false_positives": int(np.sum(~positive & (old == 0) & (new == 1))),
        "removed_false_positives": int(np.sum(~positive & (old == 1) & (new == 0)))}
    result = {"baseline_record": BASELINE, "candidate_record": str(record_path),
        "baseline_metrics": bm, "candidate_metrics": cm,
        "metric_deltas": {key: cm[key]-bm[key] for key in bm},
        "baseline_confusion_counts": bn["pooled_confusion_counts"],
        "candidate_confusion_counts": cn["pooled_confusion_counts"], "prediction_transitions": transitions,
        "paired_conditional_f1_interval_95": _paired_interval(ba["labels"], old, new),
        "bootstrap": {"repetitions": 20000, "seed": 20260929,
            "scope": "Class-stratified paired predictions, fixed fitted models and thresholds; excludes training and repeated model-selection uncertainty."},
        "folds": [{"fold": b["outer_fold"], "baseline_metrics": b["outer_evaluation_metrics"],
                   "candidate_metrics": c["outer_evaluation_metrics"],
                   "baseline_confusion_counts": b["outer_evaluation_confusion_counts"],
                   "candidate_confusion_counts": c["outer_evaluation_confusion_counts"],
                   "baseline_threshold": b["inner_threshold"], "candidate_threshold": c["inner_threshold"]}
                  for b, c in zip(bn["folds"], cn["folds"])],
        "diagnostics": [], "interpretation": "One implementation correction at fixed Phase 14 data, cleaning, parameters and nested threshold protocol. Nine fresh corrected fits; historical control reused without retraining. Final configuration and submission are unchanged."}
    rng = np.random.default_rng(20261008)
    for task, old_checkpoint in zip(manifest["tasks"][:3], baseline["outcome"]["model_checkpoints"]):
        print(f"DIAGNOSTICS outer fold {task['outer_fold']}", flush=True)
        train, labels, valid, _, valid_labels = _task_arrays(manifest, task)
        fitted, _, _ = _cached(manifest, task)
        historic, _ = HistogramGradientBoostingClassifier.load_checkpoint(root / old_checkpoint["path"])
        train_sample = np.sort(rng.choice(len(labels), min(3000, len(labels)), replace=False))
        valid_sample = np.sort(rng.choice(len(valid_labels), min(3000, len(valid_labels)), replace=False))
        old_scores = ba["probabilities"][ba["fold_ids"] == task["outer_fold"]]
        np.testing.assert_array_equal(historic.predict_proba(valid[valid_sample])[:, 1], old_scores[valid_sample])
        diagnostics = {"fold": task["outer_fold"], "sampling_seed": 20261008,
            "training_sample_size": len(train_sample), "validation_sample_size": len(valid_sample),
            "training_sample_indices_sha256": runner._array_sha256(train_sample),
            "validation_sample_indices_sha256": runner._array_sha256(valid_sample),
            "baseline_pure_missing_nodes": _pure_missing_nodes(historic, train),
            "candidate_pure_missing_nodes": _pure_missing_nodes(fitted, train)}
        for name, model in (("baseline", historic), ("candidate", fitted)):
            diagnostics[name+"_training_staged"] = _sample_staged(model, train, labels, train_sample)
            diagnostics[name+"_validation_staged"] = _sample_staged(model, valid, valid_labels, valid_sample)
        result["diagnostics"].append(diagnostics)
        del train, valid, fitted, historic
        gc.collect()
    _json(root / COMPARISON, result)
    interval = result["paired_conditional_f1_interval_95"]
    lines = ["# Phase 34 — corrected missing-only split candidates", "", result["interpretation"], "",
        "| Pipeline | Nested F1 | Precision | Recall | AP | Log loss |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, metrics in (("Historical Phase 14", bm), ("Corrected split search", cm)):
        lines.append(f"| {label} | "+" | ".join(f"{metrics[k]:.6f}" for k in ("f1", "precision", "recall", "average_precision", "log_loss"))+" |")
    lines += ["", f"Corrected minus historical F1: {cm['f1']-bm['f1']:+.6f}. Paired conditional 95% interval: [{interval[0]:+.6f}, {interval[1]:+.6f}].",
        "The interval conditions on fixed models and thresholds. Both approaches reuse development folds previously consulted for model selection and retain HAREHAB1.", "",
        "| Fold | Historical F1 | Corrected F1 | Delta | Corrected threshold |", "| --- | ---: | ---: | ---: | ---: |"]
    for fold in result["folds"]:
        b, c = fold["baseline_metrics"]["f1"], fold["candidate_metrics"]["f1"]
        lines.append(f"| {fold['fold']} | {b:.6f} | {c:.6f} | {c-b:+.6f} | {fold['candidate_threshold']:.6f} |")
    lines += ["", "Prediction transitions: "+", ".join(f"{k.replace('_', ' ')}: {v:,}" for k, v in transitions.items())+".", "",
        "Selected observed-versus-missing nodes and staged log loss/AP on identical 3,000-row training/validation samples per fold are recorded in the comparison JSON. These samples diagnose fit behaviour; they do not replace the full nested evaluation.", "",
        f"Reproduce: `python -B -m tools.phase34_missing_only_benchmark prepare`, then `python -B -m tools.phase34_missing_only_benchmark run --workers 2`.", "",
        f"Candidate record: `{record_path}`.", "Comparison: [JSON](../experiments/phase34_missing_only_split_comparison.json)."]
    (root / SUMMARY).write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"baseline_f1": bm["f1"], "corrected_f1": cm["f1"], "delta_f1": cm["f1"]-bm["f1"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "worker", "compare"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--task")
    args = parser.parse_args()
    root = Path.cwd()
    if args.action == "worker":
        if args.manifest is None or args.task is None:
            raise ValueError("An internal worker needs a manifest and task name.")
        _worker(args.manifest, args.task)
    elif args.action == "compare":
        records = list((root / RECORDS).glob("*.json"))
        if len(records) != 1:
            raise ValueError("Expected exactly one completed Phase 34 record.")
        compare(root, records[0].relative_to(root))
    else:
        if not 1 <= args.workers <= 2:
            raise ValueError("Use one or two single-thread workers.")
        if list((root / RECORDS).glob("*.json")):
            raise ValueError("A completed benchmark exists; use compare.")
        manifest = prepare(root)
        if args.action == "run":
            started = time.perf_counter()
            train_missing(manifest, args.workers)
            record = assemble(manifest, started, args.workers)
            compare(root, record.relative_to(root))


if __name__ == "__main__":
    main()

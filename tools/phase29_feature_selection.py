"""Isolated NumPy feature selection and nested 100/60-feature Phase 14 trials."""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import ctypes
import gc
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Set only this process's numerical-library limits, before importing NumPy.
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np

from src.evaluation import average_precision, best_f1_threshold, binary_log_loss, classification_metrics
from src.experiment_logger import save_experiment
from src.feature_preprocessing import tree_feature_matrices
from src.numpy_boosting import HistogramGradientBoostingClassifier, _sigmoid
from src.run_boosting_cv import _array_sha256, _metrics, _save_inner_oof_predictions, _save_oof_predictions, _variant
from src.run_preprocessing_cv import _sha256, _stratified_folds
from src.summarize_phase19_transfer import BASELINE, _arrays, _paired_interval
from tools.run_targeted_preprocessing_trials import _Tee
from tools.targeted_preprocessing_analysis import _input_data, _partitions

CONFIG = "configs/experiments/phase29_phase14_feature_selection.json"
SOURCES = ["src/numpy_boosting.py", "src/feature_preprocessing.py", "src/conditional_imputation.py",
           "src/evaluation.py", "src/run_boosting_cv.py", "src/run_preprocessing_cv.py",
           "src/experiment_logger.py", "src/summarize_phase19_transfer.py",
           "tools/targeted_preprocessing_analysis.py", "tools/run_targeted_preprocessing_trials.py",
           "tools/phase29_feature_selection.py"]


def read(root, relative):
    return json.loads((root / relative).read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def relative(root, path):
    return path.relative_to(root).as_posix()


def array_artifact(root, path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    return {"path": relative(root, path), "sha256": _sha256(path),
            "array_sha256": {key: _array_sha256(value) for key, value in arrays.items()}}


def load_arrays(root, info):
    if _sha256(root / info["path"]) != info["sha256"]:
        raise ValueError("Saved numeric artifact fingerprint differs.")
    with np.load(root / info["path"], allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    for key, fingerprint in info["array_sha256"].items():
        if _array_sha256(arrays[key]) != fingerprint:
            raise ValueError("Saved numeric array differs.")
    return arrays


def configuration(root):
    config = read(root, CONFIG)
    suite = read(root, config["reference_suite"])
    plan = _variant(read(root, suite["preprocessing_config_path"]), suite["preprocessing_variant"])["preprocessing"]
    if config["feature_counts"] != [100, 60] or config["candidate_max_features_fraction"] != 128 / 295:
        raise ValueError("Unexpected feature counts or candidate sampling fraction.")
    if config["ranking_order"] != ["mean_f1_drop", "mean_ap_drop", "mean_log_loss_increase", "split_count", "original_column_order"]:
        raise ValueError("Unexpected ranking rule.")
    if not config["groups_are_diagnostic_only"] or config["permutation_repetitions"] < 2:
        raise ValueError("Groups must remain diagnostic; permutation repetitions are required.")
    return config, suite, plan


def frozen_hashes(root, config, suite):
    files = [*SOURCES, CONFIG, config["reference_suite"], BASELINE, "configs/final_model.json",
             suite["feature_metadata_path"], suite["preprocessing_config_path"],
             *suite["data"].values(), "results/eda/phase26_two_seeds_preflight.json"]
    return {name: _sha256(root / name) for name in files}


def verify_frozen(root, config, suite, check):
    if frozen_hashes(root, config, suite) != check["frozen_sha256"]:
        raise ValueError("Frozen Phase 29 code, inputs or configuration changed.")
    for files in check["protected_parallel_hashes"].values():
        for name, fingerprint in files.items():
            if _sha256(root / name) != fingerprint:
                raise ValueError(f"Protected Phase 27/28 input changed: {name}")


def node_splits(node, depth=0):
    if node.feature is None:
        return
    yield node.feature, depth
    yield from node_splits(node.left, depth + 1)
    yield from node_splits(node.right, depth + 1)


def split_usage(model):
    counts = np.zeros(model.n_features_in_, dtype=np.int64)
    shallow = counts.copy()
    tree_features = []
    for tree in model.trees_:
        used = set()
        for column, depth in node_splits(tree.root_):
            counts[column] += 1
            shallow[column] += depth <= 1
            used.add(column)
        tree_features.append(used)
    return counts, shallow, tree_features


class PermutationPredictor:
    """Reuse unchanged trees while retaining the original addition order exactly."""

    def __init__(self, model, features):
        self.model = model
        self.binned = model.binner_.transform(features)
        self.counts, self.shallow_counts, self.tree_features = split_usage(model)
        self.cache = [tree.predict(self.binned) for tree in model.trees_]
        self.original_probabilities = self.probabilities((), None)

    def probabilities(self, columns, permutation):
        columns = np.asarray(columns, dtype=np.int64)
        if columns.size:
            if permutation is None or not np.array_equal(np.sort(permutation), np.arange(self.binned.shape[0])):
                raise ValueError("Permutation must contain each row exactly once.")
            original = self.binned[:, columns].copy()
            self.binned[:, columns] = original[permutation]
        changed = set(columns.tolist())
        try:
            scores = np.full(self.binned.shape[0], self.model.base_score_, dtype=np.float64)
            for i, tree in enumerate(self.model.trees_):
                output = tree.predict(self.binned) if changed & self.tree_features[i] else self.cache[i]
                scores += self.model.learning_rate * output
            return _sigmoid(scores)
        finally:
            if columns.size:
                self.binned[:, columns] = original


def selector_partition(labels, row_ids, seed, holdout_folds, max_rows):
    labels, row_ids = np.asarray(labels), np.asarray(row_ids)
    if labels.shape != row_ids.shape or np.unique(row_ids).size != row_ids.size:
        raise ValueError("Selector training rows must be unique and aligned.")
    fit, holdout = next(_stratified_folds(labels, holdout_folds, seed))
    rng = np.random.default_rng(seed + 100000)
    sample_count = min(int(max_rows), int(holdout.size))
    selected = []
    positives = int(np.rint(sample_count * np.mean(labels[holdout])))
    for label, count in ((1, positives), (0, sample_count - positives)):
        candidates = holdout[labels[holdout] == label]
        selected.extend(rng.choice(candidates, size=count, replace=False).tolist())
    sample = np.sort(np.asarray(selected, dtype=np.int64))
    if np.intersect1d(row_ids[fit], row_ids[holdout]).size or not np.all(np.isin(sample, holdout)):
        raise ValueError("Selector holdout must exclude fitting rows.")
    return fit, holdout, sample


def ordered_columns(rows, count):
    """Use F1 first, AP then log loss and usage only to break exact ties."""
    if count < 1 or count > len(rows) or len({r["column_index"] for r in rows}) != len(rows):
        raise ValueError("Feature ranking or requested count is invalid.")
    ordered = sorted(rows, key=lambda r: (-r["mean_f1_drop"], -r["mean_ap_drop"],
                       -r["mean_log_loss_increase"], -r["split_count"], r["column_index"]))
    return sorted(r["column_index"] for r in ordered[:count]), ordered


def score(labels, probabilities, threshold):
    counts, metrics = classification_metrics(labels, probabilities >= threshold)
    metrics.update(average_precision=average_precision(labels, probabilities), log_loss=binary_log_loss(labels, probabilities))
    return metrics, vars(counts)


def importance(predictor, columns, labels, threshold, permutations, original):
    values = []
    for permutation in permutations:
        metrics, _ = score(labels, predictor.probabilities(columns, permutation), threshold)
        values.append([original["f1"] - metrics["f1"],
                       original["average_precision"] - metrics["average_precision"],
                       metrics["log_loss"] - original["log_loss"]])
    values = np.asarray(values)
    return {"mean_f1_drop": float(values[:, 0].mean()), "std_f1_drop": float(values[:, 0].std(ddof=1)),
            "mean_ap_drop": float(values[:, 1].mean()), "mean_log_loss_increase": float(values[:, 2].mean()),
            "permutation_metric_deltas": values.tolist()}


def fit_model(features, labels, parameters, description):
    started = time.perf_counter()
    print(f"START {description}: rows={labels.size} features={features.shape[1]}", flush=True)
    model = HistogramGradientBoostingClassifier(**parameters)
    checkpoints = list(range(25, model.n_estimators + 1, 25))

    def progress(step, fitted):
        print(f"PROGRESS {description} trees={step}/{model.n_estimators} elapsed={time.perf_counter()-started:.1f}s", flush=True)

    model.fit(features, labels, checkpoints=checkpoints, checkpoint_callback=progress)
    elapsed = time.perf_counter() - started
    print(f"FIT COMPLETE {description} seconds={elapsed:.1f}", flush=True)
    return model, {"description": description, "training_rows": int(labels.size), "feature_count": features.shape[1],
                   "fit_seconds": elapsed, "seconds_per_tree": model.tree_fit_seconds_, "parameters": parameters,
                   "bin_counts": model.binner_.bin_counts_.tolist()}


def preflight(root):
    config, suite, plan = configuration(root)
    directory = root / config["artifact_dir"]
    if (directory / "executed_runner.py").exists():
        raise ValueError("Training snapshot exists; preserve its original preflight.")
    previous = read(root, "results/eda/phase26_two_seeds_preflight.json")
    fingerprints = frozen_hashes(root, config, suite)
    for name, expected in previous["source_sha256"].items():
        if _sha256(root / name) != expected:
            raise ValueError(f"Phase 14 baseline source changed: {name}")
    baseline = read(root, BASELINE)
    for key, path in (("features", suite["data"]["features_path"]), ("labels", suite["data"]["labels_path"]),
                      ("split_indices", suite["data"]["split_indices_path"]), ("feature_metadata", suite["feature_metadata_path"])):
        if fingerprints[path] != baseline["outcome"]["input_sha256"][key]:
            raise ValueError(f"Phase 14 input differs: {key}")
    for path, key in ((BASELINE, "baseline_record_sha256"), (config["reference_suite"], "reference_suite_sha256"),
                      ("configs/final_model.json", "final_config_sha256")):
        if fingerprints[path] != previous[key]:
            raise ValueError(f"Phase 14 reference differs: {path}")
    parallel = {}
    for phase, prefix in ((27, "phase27_phase14_lr0025_400trees"), (28, "phase28_phase14_ablate_alcday5")):
        path = f"results/eda/{prefix}_preflight.json"
        if (root / path).exists():
            protected = read(root, path)["frozen_sha256"]
            for name, expected in protected.items():
                if _sha256(root / name) != expected:
                    raise ValueError(f"Phase {phase} protected input already differs: {name}")
            parallel[str(phase)] = protected
    x, y, development, names, metadata = _input_data(root, suite)
    arrays = _arrays(root, baseline)
    np.testing.assert_array_equal(development, arrays["development_indices"])
    np.testing.assert_array_equal(y[development], arrays["labels"])
    result = {"frozen_sha256": fingerprints, "protected_parallel_hashes": parallel, "partitions": [],
              "baseline_checkpoint_checks": [], "diagnostic_split_counts": []}
    for (label, train, valid), old in zip(_partitions(y, development, suite), previous["partitions"]):
        item = {"partition": label, "training_rows": int(train.size), "validation_rows": int(valid.size),
                "training_row_indices_sha256": _array_sha256(development[train]),
                "training_labels_sha256": _array_sha256(y[development[train]]), "output_names": old["output_names"]}
        for key in ("partition", "training_rows", "validation_rows", "training_row_indices_sha256", "training_labels_sha256"):
            if item[key] != old[key]:
                raise ValueError(f"Partition differs from baseline: {label}/{key}")
        if len(item["output_names"]) != 295:
            raise ValueError("Expected 295 baseline columns.")
        result["partitions"].append(item)
        if "inner" not in label:
            fold = int(label[-1])
            ct, cv, cn = tree_feature_matrices(x[development[train]], x[development[valid]], names, metadata, plan)
            if cn != item["output_names"]:
                raise ValueError("Baseline output order differs.")
            checkpoint = baseline["outcome"]["model_checkpoints"][fold - 1]
            if _sha256(root / checkpoint["path"]) != checkpoint["sha256"]:
                raise ValueError("Baseline checkpoint differs.")
            model, saved = HistogramGradientBoostingClassifier.load_checkpoint(root / checkpoint["path"])
            np.testing.assert_array_equal(saved["training_row_indices"], development[train])
            np.testing.assert_array_equal(model.predict_proba(cv)[:, 1], arrays["probabilities"][valid])
            np.testing.assert_array_equal(arrays["fold_ids"][valid], np.full(valid.size, fold))
            counts, shallow, _ = split_usage(model)
            result["baseline_checkpoint_checks"].append({"fold": fold, "maximum_probability_difference": 0.0})
            result["diagnostic_split_counts"].append({"fold": fold, "feature_names": cn, "counts": counts.tolist(),
                                                    "shallow_counts": shallow.tolist()})
            del ct, cv, model
        print(f"PREFLIGHT {label}: boundaries and baseline checked", flush=True)
        gc.collect()
    write(directory / "preflight.json", result)
    return result


def checked_ranking(root, manifest, training_ids, excluded_ids, frozen):
    if manifest["frozen_sha256"] != frozen or manifest["training_row_indices_sha256"] != _array_sha256(training_ids):
        raise ValueError("Cached selector code or training boundary differs.")
    arrays = load_arrays(root, manifest["row_artifact"])
    np.testing.assert_array_equal(arrays["training_row_indices"], training_ids)
    np.testing.assert_array_equal(arrays["excluded_validation_row_indices"], excluded_ids)
    fitting, holdout = arrays["selector_fit_row_indices"], arrays["selector_holdout_row_indices"]
    if np.intersect1d(fitting, holdout).size or np.intersect1d(training_ids, excluded_ids).size:
        raise ValueError("Cached selector leaks validation rows.")
    np.testing.assert_array_equal(np.sort(np.concatenate([fitting, holdout])), np.sort(training_ids))
    if not np.all(np.isin(arrays["permutation_row_indices"], holdout)):
        raise ValueError("Permutation rows are outside the selector holdout.")
    if _sha256(root / manifest["checkpoint"]["path"]) != manifest["checkpoint"]["sha256"]:
        raise ValueError("Cached selector checkpoint differs.")
    for count in (100, 60):
        selected, ordered = ordered_columns(manifest["ranking"], count)
        if selected != manifest["selected_indices"][str(count)]:
            raise ValueError("Cached selected columns differ from the frozen rule.")
    return manifest


def select_features(root, config, suite, plan, check, x, y, names, metadata, training_ids, excluded_ids, label, seed):
    directory = root / config["artifact_dir"] / label
    path = directory / "selection.json"
    if path.exists():
        print(f"REUSE selector {label}", flush=True)
        return checked_ranking(root, read(root, relative(root, path)), training_ids, excluded_ids, check["frozen_sha256"])
    started = time.perf_counter()
    if np.intersect1d(training_ids, excluded_ids).size:
        raise ValueError("Selection training and future validation overlap.")
    fit, holdout, sample = selector_partition(y[training_ids], training_ids, seed,
                       config["selector_holdout_folds"], config["permutation_max_rows"])
    ct, cv, cn = tree_feature_matrices(x[training_ids[fit]], x[training_ids[holdout]], names, metadata, plan)
    if cn != check["partitions"][0]["output_names"]:
        raise ValueError("Selector preprocessing differs from the baseline representation.")
    parameters = {**suite["models"][0]["parameters"], "n_estimators": 200}
    model, profile = fit_model(ct, y[training_ids[fit]], parameters, f"selector/{label}")
    holdout_probabilities = model.predict_proba(cv)[:, 1]
    threshold = best_f1_threshold(y[training_ids[holdout]], holdout_probabilities).threshold
    inverse = np.full(training_ids.size, -1, dtype=np.int64)
    inverse[holdout] = np.arange(holdout.size)
    rank_features, rank_labels = cv[inverse[sample]], y[training_ids[sample]]
    predictor = PermutationPredictor(model, rank_features)
    np.testing.assert_array_equal(predictor.original_probabilities, holdout_probabilities[inverse[sample]])
    original, counts = score(rank_labels, predictor.original_probabilities, threshold)
    rng = np.random.default_rng(seed + 200000)
    permutations = [rng.permutation(sample.size) for _ in range(config["permutation_repetitions"])]
    registry = {entry["name"]: entry for entry in metadata}
    rows = []
    for column, name in enumerate(cn):
        values = importance(predictor, [column], rank_labels, threshold, permutations, original)
        rows.append({"column_index": column, "feature": name, **values,
                     "split_count": int(predictor.counts[column]), "shallow_split_count": int(predictor.shallow_counts[column]),
                     "fit_missing_fraction": float(np.mean(np.isnan(ct[:, column]))),
                     "semantic_type": registry[name].get("semantic_type", ""),
                     "description": registry[name].get("title", ""), "leakage_risk": registry[name].get("leakage_risk", "")})
        if (column + 1) % 30 == 0:
            print(f"RANK {label} columns={column+1}/{len(cn)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    _, ordered = ordered_columns(rows, len(rows))
    groups = []
    for group, configured in config["diagnostic_groups"].items():
        columns = [cn.index(n) for n in configured if n in cn]
        if len(columns) > 1:
            groups.append({"group": group, "members": [cn[i] for i in columns],
                           **importance(predictor, columns, rank_labels, threshold, permutations, original)})
    checkpoint_path = directory / "selector_model.pkl"
    model.save_checkpoint(checkpoint_path, {"training_row_indices": training_ids[fit], "output_feature_names": cn,
                    "training_labels_sha256": _array_sha256(y[training_ids[fit]]), "input_sha256": check["frozen_sha256"]})
    row_info = array_artifact(root, directory / "selector_rows.npz", {
        "training_row_indices": training_ids, "excluded_validation_row_indices": excluded_ids,
        "selector_fit_row_indices": training_ids[fit], "selector_holdout_row_indices": training_ids[holdout],
        "permutation_row_indices": training_ids[sample], "permutation_labels": rank_labels,
        "holdout_labels": y[training_ids[holdout]], "holdout_probabilities": holdout_probabilities,
        "permutation_original_probabilities": predictor.original_probabilities})
    manifest = {"partition": label, "seed": seed, "frozen_sha256": check["frozen_sha256"],
        "training_row_indices_sha256": _array_sha256(training_ids), "training_labels_sha256": _array_sha256(y[training_ids]),
        "threshold": threshold, "permutation_sample_rows": int(sample.size),
        "permutation_sample_metrics": original, "permutation_sample_confusion_counts": counts,
        "feature_names": cn, "ranking": ordered, "groups": groups,
        "selected_indices": {str(n): ordered_columns(rows, n)[0] for n in config["feature_counts"]},
        "row_artifact": row_info, "checkpoint": {"path": relative(root, checkpoint_path), "sha256": _sha256(checkpoint_path)},
        "profile": profile, "runtime_seconds": time.perf_counter() - started}
    checked_ranking(root, manifest, training_ids, excluded_ids, check["frozen_sha256"])
    analysis = root / config["analysis_dir"] / label
    analysis.mkdir(parents=True, exist_ok=True)
    csv_fields = [key for key in ordered[0] if key != "permutation_metric_deltas"]
    with (analysis / "ranking.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["rank", *csv_fields])
        writer.writeheader()
        writer.writerows({"rank": i, **{key: row[key] for key in csv_fields}} for i, row in enumerate(ordered, 1))
    for count in config["feature_counts"]:
        write(analysis / f"top{count}.json", {"partition": label, "training_row_indices_sha256": manifest["training_row_indices_sha256"],
              "features_in_original_order": [cn[i] for i in manifest["selected_indices"][str(count)]],
              "features_in_rank_order": [row["feature"] for row in ordered[:count]],
              "scope": "Feature selector fitted on this training partition only; never a global list reused inside CV."})
    write(analysis / "group_importance.json", groups)
    write(path, manifest)
    print(f"SELECTION COMPLETE {label}: top20={[r['feature'] for r in ordered[:20]]} seconds={manifest['runtime_seconds']:.1f}", flush=True)
    del ct, cv, model, predictor
    gc.collect()
    return manifest


def candidate_fit(root, config, suite, plan, check, x, y, names, metadata, training_ids, validation_ids, selection, count, label):
    directory = root / config["artifact_dir"] / label
    path = directory / f"candidate{count}.json"
    if path.exists():
        result = read(root, relative(root, path))
        if result["frozen_sha256"] != check["frozen_sha256"] or result["selected_indices"] != selection["selected_indices"][str(count)]:
            raise ValueError("Cached candidate inputs or selection differ.")
        arrays = load_arrays(root, result["predictions_artifact"])
        np.testing.assert_array_equal(arrays["training_row_indices"], training_ids)
        np.testing.assert_array_equal(arrays["validation_row_indices"], validation_ids)
        np.testing.assert_array_equal(arrays["labels"], y[validation_ids])
        if result["training_labels_sha256"] != _array_sha256(y[training_ids]) or _sha256(root / result["checkpoint"]["path"]) != result["checkpoint"]["sha256"]:
            raise ValueError("Cached candidate labels or checkpoint differ.")
        print(f"REUSE candidate top{count}/{label}", flush=True)
        return result
    started = time.perf_counter()
    ct, cv, cn = tree_feature_matrices(x[training_ids], x[validation_ids], names, metadata, plan)
    if cn != selection["feature_names"] or np.intersect1d(training_ids, validation_ids).size:
        raise ValueError("Candidate feature names or evaluation boundary differ.")
    selected = selection["selected_indices"][str(count)]
    selected_names = [cn[i] for i in selected]
    ct, cv = ct[:, selected], cv[:, selected]
    parameters = {**suite["models"][0]["parameters"], "n_estimators": 200,
                  "max_features": config["candidate_max_features_fraction"]}
    model, profile = fit_model(ct, y[training_ids], parameters, f"top{count}/{label}")
    probabilities = model.predict_proba(cv)[:, 1]
    checkpoint = directory / f"top{count}_model.pkl"
    model.save_checkpoint(checkpoint, {"training_row_indices": training_ids, "output_feature_names": selected_names,
             "training_labels_sha256": _array_sha256(y[training_ids]), "input_sha256": check["frozen_sha256"],
             "selection_artifact": relative(root, directory / "selection.json")})
    info = array_artifact(root, directory / f"top{count}_predictions.npz", {
             "training_row_indices": training_ids, "validation_row_indices": validation_ids,
             "labels": y[validation_ids], "probabilities": probabilities, "selected_indices": np.asarray(selected, dtype=np.int64)})
    result = {"partition": label, "feature_count": count, "feature_names": selected_names,
              "selected_indices": selected, "frozen_sha256": check["frozen_sha256"],
              "training_labels_sha256": _array_sha256(y[training_ids]), "parameters": parameters,
              "candidates_per_node": math.ceil(parameters["max_features"] * count), "profile": profile,
              "checkpoint": {"path": relative(root, checkpoint), "sha256": _sha256(checkpoint)},
              "predictions_artifact": info, "runtime_seconds": time.perf_counter() - started}
    write(path, result)
    del model, ct, cv
    gc.collect()
    return result


def verify_candidate(root, result, config, suite, plan, check, x, y, names, metadata, training_ids, valid_ids, selection):
    arrays = load_arrays(root, result["predictions_artifact"])
    np.testing.assert_array_equal(arrays["training_row_indices"], training_ids)
    np.testing.assert_array_equal(arrays["validation_row_indices"], valid_ids)
    np.testing.assert_array_equal(arrays["labels"], y[valid_ids])
    count = result["feature_count"]
    selected = selection["selected_indices"][str(count)]
    expected_names = [selection["feature_names"][i] for i in selected]
    expected_parameters = {**suite["models"][0]["parameters"], "n_estimators": 200,
                           "max_features": config["candidate_max_features_fraction"]}
    if (result["selected_indices"] != selected or result["feature_names"] != expected_names
            or len(selected) != count or result["parameters"] != expected_parameters
            or result["frozen_sha256"] != check["frozen_sha256"]):
        raise ValueError("Candidate differs from its fold-local selected pipeline.")
    np.testing.assert_array_equal(arrays["selected_indices"], np.asarray(selected))
    cp = result["checkpoint"]
    if _sha256(root / cp["path"]) != cp["sha256"]:
        raise ValueError("Candidate checkpoint hash differs.")
    model, saved = HistogramGradientBoostingClassifier.load_checkpoint(root / cp["path"])
    np.testing.assert_array_equal(saved["training_row_indices"], training_ids)
    if (saved["training_labels_sha256"] != _array_sha256(y[training_ids]) or saved["output_feature_names"] != expected_names
            or saved["input_sha256"] != check["frozen_sha256"]):
        raise ValueError("Candidate checkpoint training boundary or inputs differ.")
    for key, value in expected_parameters.items():
        if getattr(model, key) != value:
            raise ValueError("Saved model parameters differ.")
    ct, cv, cn = tree_feature_matrices(x[training_ids], x[valid_ids], names, metadata, plan)
    np.testing.assert_array_equal(model.predict_proba(cv[:, selected])[:, 1], arrays["probabilities"])
    print(f"VERIFIED top{count}/{result['partition']}: checkpoint probabilities exact", flush=True)
    del ct, cv, model
    gc.collect()
    return arrays


def evaluate(root, config, suite, plan, check, x, y, development, names, metadata, selections, fitted):
    baseline = read(root, BASELINE)
    base_arrays = _arrays(root, baseline)
    base_nested = baseline["outcome"]["nested_threshold_evaluation"]
    outer = list(_stratified_folds(y[development], 3, suite["seed"]))
    results = []
    for count in config["feature_counts"]:
        probabilities = np.full(development.size, np.nan)
        thresholds = probabilities.copy()
        fold_ids = np.zeros(development.size, dtype=np.int8)
        details, checkpoints, profiles = [], [], []
        for fold, (train, valid) in enumerate(outer, 1):
            label = f"outer_{fold}"
            item = fitted[(label, count)]
            ca = verify_candidate(root, item, config, suite, plan, check, x, y, names, metadata,
                                  development[train], development[valid], selections[label])
            probabilities[valid], fold_ids[valid] = ca["probabilities"], fold
            checkpoints.append({"fold": fold, **item["checkpoint"], "feature_names": item["feature_names"]})
            profiles.append(item["profile"])
            inner_prob = np.full(train.size, np.nan)
            inner_ids = np.zeros(train.size, dtype=np.int8)
            for number, (it, iv) in enumerate(_stratified_folds(y[development[train]], 2, suite["nested_threshold"]["inner_seed"] + fold), 1):
                ilabel = f"outer_{fold}_inner_{number}"
                ii = fitted[(ilabel, count)]
                ia = verify_candidate(root, ii, config, suite, plan, check, x, y, names, metadata,
                                      development[train[it]], development[train[iv]], selections[ilabel])
                inner_prob[iv], inner_ids[iv] = ia["probabilities"], number
                profiles.append(ii["profile"])
            if not np.all(np.isfinite(inner_prob)):
                raise ValueError("Inner OOF is incomplete.")
            threshold = best_f1_threshold(y[development[train]], inner_prob).threshold
            thresholds[valid] = threshold
            inner_info = _save_inner_oof_predictions(root / config["artifact_dir"] / "inner_oof_predictions",
                f"phase29_top{count}", fold, suite["nested_threshold"]["inner_seed"] + fold,
                development[train], development[valid], inner_ids, y[development[train]], inner_prob)
            inner_info["path"] = relative(root, Path(inner_info["path"]))
            fm, fc = _metrics(y[development[valid]], probabilities[valid], threshold)
            details.append({"outer_fold": fold, "inner_threshold": threshold,
                "inner_selection_metrics": _metrics(y[development[train]], inner_prob, threshold)[0],
                "outer_evaluation_metrics": fm, "outer_evaluation_confusion_counts": fc,
                "inner_oof_predictions_artifact": inner_info})
        if not np.all(np.isfinite(probabilities)) or not np.all(np.isfinite(thresholds)):
            raise ValueError("Outer OOF is incomplete.")
        np.testing.assert_array_equal(fold_ids, base_arrays["fold_ids"])
        np.testing.assert_array_equal(development, base_arrays["development_indices"])
        np.testing.assert_array_equal(y[development], base_arrays["labels"])
        counts, metrics = classification_metrics(y[development], probabilities >= thresholds)
        metrics.update(average_precision=average_precision(y[development], probabilities), log_loss=binary_log_loss(y[development], probabilities))
        nested = {"protocol": "Every inner and outer training fits its own selector on an internal held-out subset; thresholds use inner OOF only.",
            "folds": details, "pooled_metrics": metrics, "pooled_confusion_counts": vars(counts),
            "threshold_summary": {"minimum": float(thresholds.min()), "maximum": float(thresholds.max()),
                                  "mean": float(np.mean([d["inner_threshold"] for d in details]))}}
        config_record = copy.deepcopy(baseline["config"])
        config_record["experiment_name"] = f"phase29-phase14-fold-local-top{count}-permutation-features"
        config_record["hypothesis"] = config["hypothesis"]
        config_record["features"].update(output_feature_count=count, selection_config=CONFIG,
                         selection_rule=config["ranking_order"], fold_local_feature_sets=True)
        config_record["model"]["parameters"]["max_features"] = config["candidate_max_features_fraction"]
        config_record["model"]["name"] = f"depth5_top{count}_fraction128of295"
        config_record["threshold"] = {"method": "nested_inner_oof_best_f1", "value": nested["threshold_summary"]["mean"]}
        exploratory = best_f1_threshold(y[development], probabilities).threshold
        oof_path, oof_sha = _save_oof_predictions(root / config["artifact_dir"] / "oof_predictions", f"phase29_top{count}",
                                              development, fold_ids, y[development], probabilities, exploratory, thresholds)
        runtime = sum(fitted[(label, count)]["runtime_seconds"] for label in selections if label != "development")
        shared_selector_runtime = sum(s["runtime_seconds"] for label, s in selections.items() if label != "development")
        outcome = {"status": "completed", "metrics": metrics, "confusion_counts": vars(counts),
            "runtime_seconds": runtime + shared_selector_runtime,
            "runtime_note": "Includes all nine selectors, shared between the two candidate dimensions. Do not sum this runtime across candidates.",
            "candidate_runtime_seconds": runtime, "shared_selector_runtime_seconds": shared_selector_runtime,
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
            "frozen_sha256": check["frozen_sha256"], "individual_fit_profiles": profiles,
            "nested_threshold_evaluation": nested, "model_checkpoints": checkpoints,
            "selection_artifacts": {label: relative(root, root / config["artifact_dir"] / label / "selection.json") for label in selections if label != "development"},
            "oof_predictions_artifact": {"path": relative(root, oof_path), "sha256": oof_sha},
            "final_model_changed": False}
        record = save_experiment(config_record, outcome, root / config["record_dir"])
        deltas = {key: value - base_nested["pooled_metrics"][key] for key, value in metrics.items()}
        fold_deltas = [d["outer_evaluation_metrics"]["f1"] - b["outer_evaluation_metrics"]["f1"] for d, b in zip(details, base_nested["folds"])]
        selected_sets = [set(fitted[(f"outer_{fold}", count)]["feature_names"]) for fold in (1, 2, 3)]
        common = sorted(set.intersection(*selected_sets))
        jaccard = [len(selected_sets[a] & selected_sets[b]) / len(selected_sets[a] | selected_sets[b]) for a, b in ((0, 1), (0, 2), (1, 2))]
        results.append({"feature_count": count, "record": relative(root, record), "nested": nested, "metric_deltas": deltas,
            "fold_f1_deltas": fold_deltas, "positive_fold_count": sum(d > 0 for d in fold_deltas),
            "meets_plan_promotion_rule": deltas["f1"] > 0 and sum(d > 0 for d in fold_deltas) >= 2,
            "paired_conditional_f1_delta_interval_95": _paired_interval(y[development], base_arrays["nested_predictions"], probabilities >= thresholds),
            "confusion_deltas": {key: value - base_nested["pooled_confusion_counts"][key] for key, value in vars(counts).items()},
            "common_outer_features": common, "outer_pairwise_jaccard": jaccard,
            "candidate_runtime_seconds": runtime, "shared_selector_runtime_seconds": shared_selector_runtime,
            "candidates_per_node": math.ceil(config["candidate_max_features_fraction"] * count)})
        print(f"RESULT top{count}: F1={metrics['f1']:.9f} delta={deltas['f1']:+.9f} folds={fold_deltas}", flush=True)
    return results


def report(root, config, check, selections, results, runtime):
    development = selections["development"]
    summary = {"baseline_record": BASELINE, "baseline_f1": read(root, BASELINE)["outcome"]["nested_threshold_evaluation"]["pooled_metrics"]["f1"],
               "candidates": results, "runtime_seconds": runtime, "selectors_completed": len(selections),
               "candidate_fits_completed": 18, "protected_parallel_inputs_unchanged": True,
               "final_model_unchanged": True, "frozen_sha256": check["frozen_sha256"],
               "bootstrap_scope": "Paired stratified 20000-repetition row bootstrap, fixed fits and thresholds; excludes retraining and repeated model selection.",
               "deployment_selection_scope": "Development only, fitted after candidate pipelines were fixed; these exact deployment lists do not have independent OOF scores.",
               "development_ranking": relative(root, root / config["analysis_dir"] / "development" / "ranking.csv")}
    write(root / config["analysis_dir"] / "comparison.json", summary)
    lines = ["# Phase 29 — Selezione di 100 e 60 feature", "",
        "## Risultati", "", f"Controllo Phase 14: F1 annidato **{summary['baseline_f1']:.9f}**.", "",
        "| Input | F1 | Delta F1 | Precision | Recall | AP | Log loss | Fold favorevoli |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for result in results:
        m = result["nested"]["pooled_metrics"]
        lines.append(f"| {result['feature_count']} | {m['f1']:.9f} | {result['metric_deltas']['f1']:+.9f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['average_precision']:.6f} | {m['log_loss']:.6f} | {result['positive_fold_count']}/3 |")
    lines.extend(["", "## Protocollo e interpretazione", "",
        "Ogni training esterno e interno costruisce la propria selezione: booster Phase 14 a 200 alberi allenato sull'80% del relativo training, con holdout stratificato del 20%. Il preprocessing del selettore viene appreso solo sulla sua parte di fit. La soglia viene scelta una volta sul suo holdout originale; non cambia durante le permutazioni. L'importanza usa fino a 10.000 righe del medesimo holdout, campionate mantenendo la proporzione delle classi, e tre permutazioni, compresi i NaN.", "",
        "Ordine fissato prima dei risultati: calo medio F1, poi calo AP, aumento log loss, numero di split e ordine originale, usati come spareggi esatti. La deviazione standard descrive l'instabilità: tre ripetizioni non sono una prova statistica. Le permutazioni dei gruppi semantici sono diagnostiche e non cambiano il ranking o il budget esatto delle colonne. Non si eliminano colonne in base alla sola percentuale di mancanti.", "",
        "Top 60 è un sottoinsieme di top 100 in ogni training. Il booster ridotto viene riaddestrato su tutto quel training. Max_features resta la frazione 128/295: 44 colonne candidate con 100 input, 27 con 60. Soglie finali scelte da OOF dei due fold interni; tre fold esterni e tutti i seed di controllo invariati.", "",
        "I valori di importanza descrivono il modello del selettore. Possono sottovalutare variabili ridondanti o utili dopo un riaddestramento; il confronto annidato verifica la pipeline di selezione, non dimostra che sia stata trovata la migliore combinazione possibile.", ""])
    for result in results:
        count = result["feature_count"]
        lines.extend([f"## Dettaglio top {count}", "", f"Record: [{Path(result['record']).name}](../../{result['record']}).", "",
            f"Regola di promozione del piano soddisfatta: **{'sì' if result['meets_plan_promotion_rule'] else 'no'}**. Il modello finale non viene modificato automaticamente.", "",
            f"Delta F1 per fold: {', '.join(f'{v:+.9f}' for v in result['fold_f1_deltas'])}.",
            f"Delta dei conteggi: {result['confusion_deltas']}.",
            f"Intervallo bootstrap descrittivo 95% del delta F1: {result['paired_conditional_f1_delta_interval_95']}. Fit e soglie fissi; non comprende l'incertezza del training e delle molte prove sul development.",
            f"Feature comuni alle tre selezioni esterne: {len(result['common_outer_features'])}/{count}. Jaccard fra coppie: {result['outer_pairwise_jaccard']}.",
            f"Tempo dei nove fit ridotti e preprocessing: {result['candidate_runtime_seconds']/60:.2f} minuti; costo dei nove selettori condiviso: {result['shared_selector_runtime_seconds']/60:.2f} minuti.", ""])
    lines.extend(["## Classifica e liste sul development", "",
        "Questa decima selezione usa soltanto le 262.508 righe development. Produce liste utilizzabili come punto di partenza per un training successivo; non vengono riutilizzate per calcolare gli F1 riportati sopra. Un futuro training sull'intero dataset deve rifare la selezione su quel training.", "",
        "- [Classifica completa](phase29_feature_selection/development/ranking.csv).",
        "- [Lista delle 100 feature](phase29_feature_selection/development/top100.json).",
        "- [Lista delle 60 feature](phase29_feature_selection/development/top60.json).",
        "- [Importanza dei gruppi](phase29_feature_selection/development/group_importance.json).", "",
        "| Posizione | Feature | Calo medio F1 | Deviazione standard | Calo AP | Mancanti nel fit |", "| ---: | --- | ---: | ---: | ---: | ---: |"])
    for position, row in enumerate(development["ranking"][:30], 1):
        lines.append(f"| {position} | {row['feature']} | {row['mean_f1_drop']:.6f} | {row['std_f1_drop']:.6f} | {row['mean_ap_drop']:.6f} | {row['fit_missing_fraction']:.2%} |")
    lines.extend(["", "La selezione non risolve il dubbio di ammissibilità di HAREHAB1. Il 20% storico esterno al development non è stato consultato in questa prova. I fold development sono già stati usati per molti esperimenti: i risultati non equivalgono a una valutazione finale indipendente.", "",
        "## Artefatti e isolamento", "", f"Tempo totale di questa esecuzione: {runtime/60:.2f} minuti. Dieci selettori e diciotto fit dei candidati completati, in sequenza, con un thread NumPy e priorità inferiore su Windows.", "",
        f"Artefatti numerici e tutti i checkpoint: `{config['artifact_dir']}`. Ogni selettore conserva le righe di fit, holdout, permutazione ed esclusione; ogni candidato conserva lista, righe, etichette, probabilità e checkpoint. Tutti i diciotto checkpoint riproducono esattamente le probabilità salvate. OOF esterni e interni, sorgente eseguito, preflight e hash conservati.", "",
        "I sorgenti, le configurazioni e gli input protetti di Phase 27/28 sono stati verificati prima e dopo; nessuna scrittura nelle loro cartelle o nei registri condivisi. I due nuovi record restano nel registro dedicato `results/experiments/phase29_feature_selection/`. Nessuna nuova submission, modifica del modello finale, commit o push.", ""])
    (root / "results/eda/phase29_feature_selection_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def lower_own_priority():
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel.GetPriorityClass.argtypes = [ctypes.c_void_p]
        handle = kernel.GetCurrentProcess()
        if not kernel.SetPriorityClass(handle, 0x4000) or kernel.GetPriorityClass(handle) != 0x4000:
            raise OSError(ctypes.get_last_error(), "Cannot set this process to below-normal priority.")


def run(root):
    config, suite, plan = configuration(root)
    directory = root / config["artifact_dir"]
    check = read(root, relative(root, directory / "preflight.json"))
    verify_frozen(root, config, suite, check)
    if (directory / "comparison_complete.json").exists():
        raise ValueError("Completed experiment exists; inspect it instead of retraining.")
    snapshot = directory / "executed_runner.py"
    if snapshot.exists():
        if snapshot.read_bytes() != (root / "tools/phase29_feature_selection.py").read_bytes():
            raise ValueError("Runner differs from its original training snapshot.")
    else:
        snapshot.write_bytes((root / "tools/phase29_feature_selection.py").read_bytes())
    write(directory / "training_config.json", config)
    x, y, development, names, metadata = _input_data(root, suite)
    outer = list(_stratified_folds(y[development], 3, suite["seed"]))
    partitions = []
    for fold, (train, valid) in enumerate(outer, 1):
        partitions.append((f"outer_{fold}", train, valid))
        for number, (it, iv) in enumerate(_stratified_folds(y[development[train]], 2, suite["nested_threshold"]["inner_seed"] + fold), 1):
            partitions.append((f"outer_{fold}_inner_{number}", train[it], train[iv]))
    expected = {p["partition"]: p for p in check["partitions"]}
    selections, fitted = {}, {}
    started = time.perf_counter()
    status = {"status": "running", "pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "partitions_completed": [], "resource_policy": config["resource_policy"]}
    write(directory / "status.json", status)
    try:
        for index, (label, train, valid) in enumerate(partitions):
            if (_array_sha256(development[train]) != expected[label]["training_row_indices_sha256"]
                    or _array_sha256(y[development[train]]) != expected[label]["training_labels_sha256"]):
                raise ValueError("Runtime partition differs from preflight.")
            status["active_partition"] = label
            write(directory / "status.json", status)
            selections[label] = select_features(root, config, suite, plan, check, x, y, names, metadata,
                                      development[train], development[valid], label, config["selector_seed"] + index)
            for count in config["feature_counts"]:
                fitted[(label, count)] = candidate_fit(root, config, suite, plan, check, x, y, names, metadata,
                                  development[train], development[valid], selections[label], count, label)
            status["partitions_completed"].append(label)
            write(directory / "status.json", status)
        # These final lists never determine the masks used for the nine CV fits.
        status["active_partition"] = "development"
        write(directory / "status.json", status)
        unused = np.setdiff1d(np.arange(y.size, dtype=np.int64), development)
        selections["development"] = select_features(root, config, suite, plan, check, x, y, names, metadata,
                                      development, unused, "development", config["selector_seed"] + 9)
        results = evaluate(root, config, suite, plan, check, x, y, development, names, metadata, selections, fitted)
        verify_frozen(root, config, suite, check)
        summary = report(root, config, check, selections, results, time.perf_counter() - started)
        write(directory / "comparison_complete.json", summary)
        status.update(status="completed", active_partition=None, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      runtime_seconds=summary["runtime_seconds"])
        write(directory / "status.json", status)
        print("COMPLETE Phase 29: both full nested evaluations and deployment rankings saved", flush=True)
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        write(directory / "status.json", status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run"))
    args = parser.parse_args()
    lower_own_priority()
    root = Path.cwd()
    if args.action == "preflight":
        preflight(root)
    else:
        config, _, _ = configuration(root)
        directory = root / config["artifact_dir"]
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "run_stdout.log").open("a", encoding="utf-8") as stream:
            with contextlib.redirect_stdout(_Tee(sys.stdout, stream)), contextlib.redirect_stderr(_Tee(sys.stderr, stream)):
                run(root)


if __name__ == "__main__":
    main()

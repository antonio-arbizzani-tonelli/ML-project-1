"""Compare random and hybrid hard-negative undersampling against corrected Phase 34.

Sampling and pilot fitting use only each eligible training partition. The
standard runner preserves all outer/inner validation rows and selects nested
thresholds. Final preprocessing and histogram bins are fitted on retained rows.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import copy
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
from src.evaluation import best_f1_threshold, classification_metrics
from src.feature_preprocessing import tree_feature_matrices
from src.numpy_boosting import HistogramGradientBoostingClassifier
from src.run_preprocessing_cv import _sha256, _stratified_folds
from src.summarize_phase19_transfer import _arrays, _paired_interval, _publish
from tools.targeted_preprocessing_analysis import _input_data

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "results/boosting_artifacts/phase36_undersampling"
REFERENCE = "configs/experiments/phase34_missing_only_split.json"
BASELINE = "results/experiments/phase34_missing_only_split/20261007T092637381255Z_phase34-missing-only-split-depth5-features128-200trees.json"
METHODS = ("random", "hybrid_hard")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def suite_path(method):
    return ROOT / f"configs/experiments/phase36_undersampling_{method}.json"


def configuration(method):
    suite = read(suite_path(method))
    expected = copy.deepcopy(read(ROOT / REFERENCE))
    expected.update(experiment_prefix=f"phase36_undersampling_{method}",
                    artifact_dir=f"results/boosting_artifacts/phase36_undersampling/{method}",
                    sampling={"method": method, "negative_to_positive_ratio": 3,
                              "seed": 20261007, "hard_fraction": 0.5,
                              "pilot_trees": 24, "pilot_max_depth": 3})
    if {k: v for k, v in suite.items() if k != "hypothesis"} != {k: v for k, v in expected.items() if k != "hypothesis"}:
        raise ValueError("Only the specified sampling procedure and provenance may change.")
    runner._validate_suite(suite)
    return suite


def sources():
    paths = list((ROOT / "src").glob("*.py"))
    paths += [Path(__file__), ROOT / "tools/targeted_preprocessing_analysis.py"]
    return {p.relative_to(ROOT).as_posix(): _sha256(p) for p in paths}


def inputs(suite):
    paths = {"preprocessing_config": suite["preprocessing_config_path"],
             "feature_metadata": suite["feature_metadata_path"]}
    paths.update({key: suite["data"][field] for key, field in (
        ("features", "features_path"), ("labels", "labels_path"),
        ("split_indices", "split_indices_path"))})
    return {key: _sha256(ROOT / path) for key, path in paths.items()}


def frozen(manifest, check_inputs=False):
    if sources() != manifest["source_sha256"]:
        raise ValueError("Frozen execution sources changed.")
    for path, expected in manifest["control_sha256"].items():
        if _sha256(ROOT / path) != expected:
            raise ValueError(f"Control or suite changed: {path}")
    if check_inputs and inputs(configuration("random")) != manifest["input_sha256"]:
        raise ValueError("Dataset or preprocessing changed.")
    if {"python": platform.python_version(), "numpy": np.__version__} != manifest["environment"]:
        raise ValueError("Numerical environment changed.")


def signature(train, labels, valid):
    return {"train_shape": list(train.shape), "validation_shape": list(valid.shape),
            "train_sha256": runner._array_sha256(train),
            "labels_sha256": runner._array_sha256(labels),
            "validation_sha256": runner._array_sha256(valid)}


def select_positions(labels, method, seed, scores=None, ratio=3):
    """Retain every positive; select unique negatives without using held-out rows."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Sampling requires a one-dimensional binary target.")
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    if not positive.size or not negative.size or ratio < 1:
        raise ValueError("Sampling requires both classes and a positive ratio.")
    target = min(negative.size, int(ratio * positive.size))
    rng = np.random.default_rng(seed)
    if method == "random":
        chosen = rng.choice(negative, target, replace=False)
    elif method == "hybrid_hard":
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != labels.shape or not np.all(np.isfinite(scores)):
            raise ValueError("Hybrid sampling requires one finite training-only score per row.")
        uniform = rng.choice(negative, target // 2, replace=False)
        remaining = np.setdiff1d(negative, uniform, assume_unique=True)
        # Stable row-index tie break makes exact replay independent of sort stability.
        order = np.lexsort((remaining, -scores[remaining]))
        chosen = np.concatenate((uniform, remaining[order[:target - uniform.size]]))
    else:
        raise ValueError("Unknown undersampling method.")
    selected = np.sort(np.concatenate((positive, chosen)))
    if np.unique(selected).size != selected.size:
        raise ValueError("Sampling introduced duplicate rows.")
    return selected


def partitions(labels, development, suite):
    outer = list(_stratified_folds(labels[development], 3, suite["seed"]))
    for fold, (train, valid) in enumerate(outer, 1):
        yield f"outer_{fold}", development[train], development[valid]
    for fold, (train, _) in enumerate(outer, 1):
        for inner, (it, iv) in enumerate(_stratified_folds(labels[development[train]], 2,
                suite["nested_threshold"]["inner_seed"] + fold), 1):
            yield f"outer_{fold}_inner_{inner}", development[train[it]], development[train[iv]]


def prepare():
    path = SHARED / "preflight.json"
    if path.exists():
        manifest = read(path)
        frozen(manifest, True)
        return manifest
    suites = {m: configuration(m) for m in METHODS}
    suite, baseline = suites["random"], read(ROOT / BASELINE)
    provenance = sources()
    for name in ("src/numpy_boosting.py", "src/feature_preprocessing.py", "src/run_boosting_cv.py",
                 "src/run_preprocessing_cv.py", "src/evaluation.py"):
        if provenance[name] != baseline["outcome"]["source_sha256"][name]:
            raise ValueError(f"Execution no longer matches corrected Phase 34: {name}")
    fingerprints = inputs(suite)
    for key, value in fingerprints.items():
        if value != baseline["outcome"]["input_sha256"][key]:
            raise ValueError(f"Baseline input changed: {key}")
    ba = _arrays(ROOT, baseline)
    x, labels, development, names, metadata = _input_data(ROOT, suite)
    np.testing.assert_array_equal(development, ba["development_indices"])
    np.testing.assert_array_equal(labels[development], ba["labels"])
    plan = runner._variant(read(ROOT / suite["preprocessing_config_path"]), suite["preprocessing_variant"])["preprocessing"]
    controls = [BASELINE, REFERENCE, baseline["outcome"]["oof_predictions_artifact"]["path"]]
    controls += [entry["path"] for entry in baseline["outcome"]["model_checkpoints"]]
    controls += [suite_path(m).relative_to(ROOT).as_posix() for m in METHODS]
    manifest = {"source_sha256": provenance, "input_sha256": fingerprints,
                "control_sha256": {p: _sha256(ROOT / p) for p in controls}, "plan": plan,
                "environment": {"python": platform.python_version(), "numpy": np.__version__},
                "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                "observed_final_config_sha256": _sha256(ROOT / "configs/final_model.json"),
                "partitions": [], "baseline_checkpoint_checks": []}
    for index, (name, train_rows, valid_rows) in enumerate(partitions(labels, development, suite)):
        rows_path = SHARED / "partitions" / f"{name}.npz"
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(rows_path, training_rows=train_rows, validation_rows=valid_rows)
        train, valid, output = tree_feature_matrices(x[train_rows], x[valid_rows], names, metadata, plan)
        if len(output) != 295 or np.intersect1d(train_rows, valid_rows).size:
            raise ValueError("Unexpected features or overlapping partitions.")
        entry = {"name": name, "rows_path": rows_path.relative_to(ROOT).as_posix(),
                 "rows_sha256": _sha256(rows_path), "output_names": output,
                 "signature": signature(train, labels[train_rows], valid),
                 "eligible_rows": len(train_rows), "positive_rows": int(labels[train_rows].sum()),
                 "validation_rows": len(valid_rows)}
        if index < 3:
            cp = baseline["outcome"]["model_checkpoints"][index]
            model, saved = HistogramGradientBoostingClassifier.load_checkpoint(ROOT / cp["path"])
            np.testing.assert_array_equal(saved["training_row_indices"], train_rows)
            positions = np.flatnonzero(ba["fold_ids"] == index + 1)
            np.testing.assert_array_equal(development[positions], valid_rows)
            np.testing.assert_array_equal(model.predict_proba(valid)[:, 1], ba["probabilities"][positions])
            manifest["baseline_checkpoint_checks"].append({"fold": index + 1, "maximum_probability_difference": 0.0})
            del model
        manifest["partitions"].append(entry)
        print(f"PREFLIGHT {name}: {len(train_rows)} eligible; {entry['positive_rows'] * 4} retained; validation unchanged", flush=True)
        del train, valid
        gc.collect()
    frozen(manifest, True)
    for name in provenance:
        target = SHARED / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    write(path, manifest)
    return manifest


def run(method):
    manifest, suite = read(SHARED / "preflight.json"), configuration(method)
    frozen(manifest, True)
    directory = ROOT / suite["artifact_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    if list((directory / "records").glob("*.json")):
        compare(method)
        return
    x, all_labels, _, names, metadata = _input_data(ROOT, suite)
    parameters = runner._resolved_parameters(suite["models"][0]["parameters"], manifest["partitions"][0]["output_names"])
    profiles, checkpoints_saved = [], []

    def fit(eligible, labels, valid, actual_parameters, checkpoints, callback=None):
        number = len(profiles) + 1
        task = manifest["partitions"][number - 1]
        if signature(eligible, labels, valid) != task["signature"] or actual_parameters != parameters or checkpoints != [200]:
            raise ValueError("Runner partition, parameters or order differs from preflight.")
        rows_path = ROOT / task["rows_path"]
        if _sha256(rows_path) != task["rows_sha256"]:
            raise ValueError("Eligible row artifact changed.")
        with np.load(rows_path) as rows:
            train_rows, valid_rows = rows["training_rows"], rows["validation_rows"]
        seed = suite["sampling"]["seed"] + number
        cache = directory / "fit_cache" / task["name"]
        cache.mkdir(parents=True, exist_ok=True)
        selection_path, selection_info = cache / "selection.npz", cache / "selection.json"
        started, cpu_started = time.perf_counter(), time.process_time()
        if selection_path.exists() and selection_info.exists():
            selection = read(selection_info)
            if _sha256(selection_path) != selection["sha256"]:
                raise ValueError("Saved selected rows changed.")
            with np.load(selection_path) as saved:
                selected = saved["positions"]
                np.testing.assert_array_equal(saved["eligible_training_rows"], train_rows)
                np.testing.assert_array_equal(saved["selected_training_rows"], train_rows[selected])
            if selection["sampling"] != suite["sampling"] or selection["seed"] != seed:
                raise ValueError("Saved selection belongs to another procedure.")
        else:
            pilot_info, scores = None, None
            if method == "hybrid_hard":
                pilot_positions = select_positions(labels, "random", seed)
                pt, pe, output = tree_feature_matrices(x[train_rows[pilot_positions]], x[train_rows], names, metadata, manifest["plan"])
                pilot_parameters = {**parameters, "n_estimators": 24, "max_depth": 3}
                pilot = HistogramGradientBoostingClassifier(**pilot_parameters)
                pilot_started = time.perf_counter()
                pilot.fit(pt, labels[pilot_positions])
                scores = pilot.predict_proba(pe)[:, 1]
                pilot_info = {"parameters": pilot_parameters, "fit_and_scoring_seconds": time.perf_counter() - pilot_started,
                              "training_rows": len(pilot_positions),
                              "training_row_indices_sha256": runner._array_sha256(train_rows[pilot_positions]),
                              "training_only_scores_sha256": runner._array_sha256(scores)}
                np.savez_compressed(cache / "pilot_selection_scores.npz", positions=pilot_positions, scores=scores)
                pilot_info["artifact_sha256"] = _sha256(cache / "pilot_selection_scores.npz")
                pilot.save_checkpoint(cache / "pilot.pkl", {"training_row_indices": train_rows[pilot_positions]})
                del pilot, pt, pe
            selected = select_positions(labels, method, seed, scores)
            np.savez_compressed(selection_path, positions=selected,
                                eligible_training_rows=train_rows, selected_training_rows=train_rows[selected])
            selection = {"sampling": suite["sampling"], "seed": seed, "sha256": _sha256(selection_path),
                         "pilot": pilot_info, "selection_seconds": time.perf_counter() - started,
                         "selection_cpu_seconds": time.process_time() - cpu_started,
                         "eligible_rows": len(labels), "retained_rows": len(selected),
                         "retained_positives": int(labels[selected].sum()),
                         "retained_negatives": int((labels[selected] == 0).sum())}
            write(selection_info, selection)
        if not np.array_equal(selected, np.unique(selected)) or selected.min() < 0 or selected.max() >= len(labels):
            raise ValueError("Invalid selected positions.")
        np.testing.assert_array_equal(np.flatnonzero(labels[selected] == 1).size, np.sum(labels))
        if selection["retained_negatives"] != 3 * int(np.sum(labels)):
            raise ValueError("Unexpected class ratio.")
        train, validation, output = tree_feature_matrices(x[train_rows[selected]], x[valid_rows], names, metadata, manifest["plan"])
        if output != task["output_names"]:
            raise ValueError("Sampling changed the feature representation.")
        model_path = cache / "model.pkl"
        checkpoint_metadata = {"method": method, "partition": task["name"],
                               "signature": signature(train, labels[selected], validation),
                               "training_row_indices": train_rows[selected], "eligible_training_row_indices": train_rows,
                               "validation_row_indices": valid_rows, "output_feature_names": output,
                               "parameters": parameters, "selection_sha256": selection["sha256"],
                               "source_sha256": manifest["source_sha256"]}
        model_started, model_cpu = time.perf_counter(), time.process_time()

        def progress(step, model):
            model.save_checkpoint(model_path, checkpoint_metadata)
            print(f"PROGRESS {method} fit={number}/9 trees={step}/200 elapsed={time.perf_counter()-model_started:.1f}s", flush=True)

        prior = 0
        if model_path.exists():
            model, saved = HistogramGradientBoostingClassifier.load_checkpoint(model_path)
            for key, value in checkpoint_metadata.items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(saved[key], value)
                elif saved[key] != value:
                    raise ValueError(f"Checkpoint mismatch: {key}")
            prior = len(model.trees_)
            if not 0 < prior <= 200:
                raise ValueError("Unexpected checkpoint tree count.")
            if prior < 200:
                model.continue_fit(train, labels[selected], 200-prior, list(range(prior+25, 201, 25)), progress)
        else:
            print(f"START {method} fit={number}/9 retained={len(selected)}/{len(labels)}", flush=True)
            model = HistogramGradientBoostingClassifier(**parameters, n_estimators=200)
            model.fit(train, labels[selected], list(range(25, 201, 25)), progress)
        probabilities = model.predict_proba(validation)[:, 1]
        profile = runner._tree_profile(model.tree_fit_seconds_)
        profile.update(fit_and_validation_prediction_seconds=time.perf_counter() - model_started,
                       bin_count_minimum=int(model.binner_.bin_counts_.min()), bin_count_maximum=int(model.binner_.bin_counts_.max()))
        entry = {"partition": task["name"], **selection, "selection_path": selection_path.relative_to(ROOT).as_posix(),
                 "model_path": model_path.relative_to(ROOT).as_posix(), "model_sha256": _sha256(model_path),
                 "model_fit_and_prediction_seconds": profile["fit_and_validation_prediction_seconds"],
                 "model_cpu_seconds": time.process_time() - model_cpu, "resumed_from_tree_count": prior,
                 "tree_seconds": profile["total_seconds"], "validation_rows": len(valid_rows)}
        profiles.append(entry)
        write(cache / "completed.json", {"fit": entry, "profile": profile})
        if number <= 3:
            checkpoints_saved.append({"fold": number, "path": entry["model_path"], "sha256": entry["model_sha256"],
                                      "training_rows": len(selected), "eligible_training_rows": len(train_rows)})
        print(f"DONE {method} fit={number}/9 seconds={profile['fit_and_validation_prediction_seconds']:.1f}", flush=True)
        return {200: probabilities}, profile

    original_fit, original_save = runner._fit_checkpoint_probabilities, runner.save_experiment

    def save(config, outcome, output_dir):
        frozen(manifest, True)
        config.update(hypothesis=suite["hypothesis"], sampling=suite["sampling"])
        outcome.update(source_sha256=manifest["source_sha256"], environment=manifest["environment"],
                       individual_fit_profiles=profiles, model_checkpoints=checkpoints_saved,
                       sampling_fit_partition="Each eligible outer/inner training partition only; validation untouched.",
                       retained_training_rows_total=sum(p["retained_rows"] for p in profiles),
                       full_fitting_and_sampling_seconds=sum(p["model_fit_and_prediction_seconds"] + p["selection_seconds"] for p in profiles),
                       runtime_scope="Runner wall time; full fitting and sampling time is the sum of nine per-fit timings; resume counts recorded.")
        for fold, detail in enumerate(outcome["folds"] if "folds" in outcome else [], 1):
            detail["retained_training_rows"] = profiles[fold-1]["retained_rows"]
        return original_save(config, outcome, output_dir)

    runner._fit_checkpoint_probabilities, runner.save_experiment = fit, save
    try:
        records = runner.run_suite(suite_path(method), ROOT, directory / "records")
        if len(records) != 1 or len(profiles) != 9:
            raise ValueError("Expected one record and nine completed fits.")
    finally:
        runner._fit_checkpoint_probabilities, runner.save_experiment = original_fit, original_save
    compare(method)


def compare(method):
    manifest, suite = read(SHARED / "preflight.json"), configuration(method)
    frozen(manifest, True)
    directory = ROOT / suite["artifact_dir"]
    records = list((directory / "records").glob("*.json"))
    if len(records) != 1:
        raise ValueError("Expected exactly one completed candidate record.")
    candidate, baseline = read(records[0]), read(ROOT / BASELINE)
    ca, ba = _arrays(ROOT, candidate), _arrays(ROOT, baseline)
    outcome = candidate["outcome"]
    for key in ("development_indices", "fold_ids", "labels"):
        np.testing.assert_array_equal(ca[key], ba[key])
    for key, expected in manifest["input_sha256"].items():
        if outcome["input_sha256"][key] != expected:
            raise ValueError(f"Candidate input mismatch: {key}")
    if outcome["source_sha256"] != manifest["source_sha256"] or candidate["config"]["sampling"] != suite["sampling"]:
        raise ValueError("Candidate provenance mismatch.")
    for key in ("model", "features", "split"):
        if candidate["config"][key] != baseline["config"][key]:
            raise ValueError(f"Unexpected change: {key}")
    nested, bn = outcome["nested_threshold_evaluation"], baseline["outcome"]["nested_threshold_evaluation"]
    np.testing.assert_array_equal(ca["nested_predictions"], ca["probabilities"] >= ca["nested_thresholds"])
    counts, metrics = classification_metrics(ca["labels"], ca["nested_predictions"])
    if vars(counts) != nested["pooled_confusion_counts"] or any(metrics[k] != nested["pooled_metrics"][k] for k in metrics):
        raise ValueError("Pooled metrics do not reproduce.")
    x, labels, _, names, metadata = _input_data(ROOT, suite)
    checks = []
    for number, (task, profile) in enumerate(zip(manifest["partitions"], outcome["individual_fit_profiles"]), 1):
        with np.load(ROOT / task["rows_path"]) as rows:
            train_rows, valid_rows = rows["training_rows"], rows["validation_rows"]
        if _sha256(ROOT / profile["selection_path"]) != profile["sha256"]:
            raise ValueError("Selection artifact hash changed.")
        with np.load(ROOT / profile["selection_path"]) as selection:
            selected = selection["positions"]
            np.testing.assert_array_equal(selection["eligible_training_rows"], train_rows)
            np.testing.assert_array_equal(selection["selected_training_rows"], train_rows[selected])
        cache = (ROOT / profile["selection_path"]).parent
        scores = None
        if method == "hybrid_hard":
            pilot_artifact = cache / "pilot_selection_scores.npz"
            if _sha256(pilot_artifact) != profile["pilot"]["artifact_sha256"]:
                raise ValueError("Pilot score artifact changed.")
            with np.load(pilot_artifact) as pilot_arrays:
                scores, pilot_positions = pilot_arrays["scores"], pilot_arrays["positions"]
            np.testing.assert_array_equal(pilot_positions, select_positions(labels[train_rows], "random", profile["seed"]))
            if number <= 3:
                pilot, saved = HistogramGradientBoostingClassifier.load_checkpoint(cache / "pilot.pkl")
                np.testing.assert_array_equal(saved["training_row_indices"], train_rows[pilot_positions])
                pt, pe, _ = tree_feature_matrices(x[train_rows[pilot_positions]], x[train_rows], names, metadata, manifest["plan"])
                np.testing.assert_array_equal(pilot.predict_proba(pe)[:, 1], scores)
                del pilot, pt, pe
        np.testing.assert_array_equal(selected, select_positions(labels[train_rows], method, profile["seed"], scores))
        if np.intersect1d(train_rows[selected], valid_rows).size or int(labels[train_rows[selected]].sum()) != int(labels[train_rows].sum()):
            raise ValueError("Sampling dropped positives or crossed the validation boundary.")
        if number <= 3:
            if _sha256(ROOT / profile["model_path"]) != profile["model_sha256"]:
                raise ValueError("Final checkpoint changed.")
            model, saved = HistogramGradientBoostingClassifier.load_checkpoint(ROOT / profile["model_path"])
            np.testing.assert_array_equal(saved["training_row_indices"], train_rows[selected])
            train, valid, output = tree_feature_matrices(x[train_rows[selected]], x[valid_rows], names, metadata, manifest["plan"])
            for key, value in baseline["config"]["model"]["parameters"].items():
                if getattr(model, key) != value:
                    raise ValueError(f"Checkpoint parameter mismatch: {key}")
            np.testing.assert_array_equal(model.predict_proba(valid)[:, 1], ca["probabilities"][ca["fold_ids"] == number])
            detail = nested["folds"][number - 1]
            info = detail["inner_oof_predictions_artifact"]
            if _sha256(ROOT / info["path"]) != info["sha256"]:
                raise ValueError("Inner OOF artifact changed.")
            with np.load(ROOT / info["path"]) as inner:
                np.testing.assert_array_equal(inner["training_row_indices"], train_rows)
                np.testing.assert_array_equal(inner["outer_validation_row_indices"], valid_rows)
                np.testing.assert_array_equal(inner["labels"], labels[train_rows])
                ids = np.zeros(len(train_rows), dtype=np.int8)
                for fold, (_, iv) in enumerate(_stratified_folds(labels[train_rows], 2, suite["nested_threshold"]["inner_seed"] + number), 1):
                    ids[iv] = fold
                np.testing.assert_array_equal(inner["fold_ids"], ids)
                threshold = best_f1_threshold(inner["labels"], inner["probabilities"]).threshold
            if threshold != detail["inner_threshold"]:
                raise ValueError("Nested threshold changed.")
            np.testing.assert_array_equal(ca["nested_thresholds"][ca["fold_ids"] == number], np.full(len(valid_rows), threshold))
            checks.append({"fold": number, "checkpoint_probability_difference": 0.0,
                           "threshold": threshold, "retained_training_rows": len(selected)})
            del model, train, valid
        gc.collect()
    if len(outcome["individual_fit_profiles"]) != 9 or len(checks) != 3:
        raise ValueError("Incomplete candidate verification.")
    deltas = {k: v - bn["pooled_metrics"][k] for k, v in nested["pooled_metrics"].items()}
    fold_deltas = [f["outer_evaluation_metrics"]["f1"] - b["outer_evaluation_metrics"]["f1"] for f, b in zip(nested["folds"], bn["folds"])]
    summary = {"method": method, "candidate_record": records[0].relative_to(ROOT).as_posix(),
               "baseline_record": BASELINE, "nested": nested, "metric_deltas": deltas,
               "fold_f1_deltas": fold_deltas, "positive_fold_count": sum(d > 0 for d in fold_deltas),
               "paired_conditional_f1_delta_interval_95": _paired_interval(ca["labels"], ba["nested_predictions"], ca["nested_predictions"]),
               "bootstrap_scope": "20,000 paired stratified draws; fixed fits and thresholds; excludes training and repeated selection uncertainty.",
               "individual_fit_profiles": outcome["individual_fit_profiles"], "verification": checks,
               "fitting_and_sampling_seconds": outcome["full_fitting_and_sampling_seconds"],
               "final_config_changed_since_preflight": _sha256(ROOT / "configs/final_model.json") != manifest["observed_final_config_sha256"]}
    write(directory / "comparison.json", summary)
    print(f"VERIFIED {method}: F1={nested['pooled_metrics']['f1']:.9f} delta={deltas['f1']:+.9f}", flush=True)
    return summary


def report(publish=False):
    baseline = read(ROOT / BASELINE)
    summaries = [read(SHARED / method / "comparison.json") for method in METHODS]
    if publish:
        for summary in summaries:
            summary["published_record"] = _publish(ROOT / summary["candidate_record"], ROOT).relative_to(ROOT).as_posix()
            write(SHARED / summary["method"] / "comparison.json", summary)
    aggregate = {"baseline_record": BASELINE, "baseline_nested": baseline["outcome"]["nested_threshold_evaluation"],
                 "candidates": summaries, "final_model_modified_by_experiment": False,
                 "scope": "Corrected Phase 34; same 295 columns, seeds and 200-tree parameters; 3 outer x 2 inner folds; validation prevalence unchanged."}
    write(ROOT / "results/experiments/phase36_undersampling_comparison.json", aggregate)
    lines = ["# Phase 36 — Due metodi di undersampling", "",
             "Confronto con la versione corretta Phase 34. Tutti i positivi restano nel training; si conservano tre negativi per positivo. Random sceglie i negativi uniformemente. Hybrid hard conserva metà negativi casuali e metà tra quelli con probabilità positiva maggiore secondo un pilot di 24 alberi, profondità 3, addestrato esclusivamente sul training del fold. Il pilot usa a sua volta un campione casuale 3:1.", "",
             "Stesse 295 colonne, parametri del modello finale e seed del controllo. Campionamento indipendente nei tre fold esterni e nei sei interni. Preprocessing finale e bin apprendono solo dalle righe conservate. Validazione esterna e interna mantengono tutte le righe e la prevalenza originale. Soglie F1 scelte tramite OOF interno; holdout e test non consultati.", "",
             "| Procedura | F1 | Precision | Recall | AP | Falsi positivi | Delta F1 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    rows = [("Tutti i dati", aggregate["baseline_nested"], 0.0)]
    rows += [(s["method"], s["nested"], s["metric_deltas"]["f1"]) for s in summaries]
    for method, nested, delta in rows:
        m, c = nested["pooled_metrics"], nested["pooled_confusion_counts"]
        lines.append(f"| {method} | {m['f1']:.9f} | {m['precision']:.9f} | {m['recall']:.9f} | {m['average_precision']:.9f} | {c['false_positives']} | {delta:+.9f} |")
    for summary in summaries:
        lines += ["", f"## {summary['method']}", "", "| Fold | F1 | Delta F1 | Soglia interna |", "| --- | ---: | ---: | ---: |"]
        for i, fold in enumerate(summary["nested"]["folds"]):
            lines.append(f"| {i+1} | {fold['outer_evaluation_metrics']['f1']:.9f} | {summary['fold_f1_deltas'][i]:+.9f} | {fold['inner_threshold']:.9f} |")
        ci = summary["paired_conditional_f1_delta_interval_95"]
        profiles = summary["individual_fit_profiles"]
        lines += ["", f"Fold favorevoli: {summary['positive_fold_count']}/3. Intervallo bootstrap condizionale del delta F1: [{ci[0]:+.9f}, {ci[1]:+.9f}].",
                  f"Tempo totale dei nove fit, selezione/pilot inclusi: {summary['fitting_and_sampling_seconds']:.1f} secondi (somma dei tempi per fit, due candidati eseguiti in parallelo).",
                  f"Righe conservate nei fold esterni: {', '.join(str(p['retained_rows']) for p in profiles[:3])}; positivi tutti conservati. Checkpoint ripresi: {sum(p['resumed_from_tree_count'] > 0 for p in profiles)}/9.",
                  f"Record: `{summary.get('published_record', summary['candidate_record'])}`."]
    lines += ["", "Verifiche: replay dei tre checkpoint del controllo; riproduzione esatta di tutte le selezioni; controllo dei confini di tutti i nove training; replay dei tre checkpoint finali e dei tre pilot esterni; OOF interni, soglie e metriche riprodotti. Snapshot sorgenti e checkpoint ogni 25 alberi conservati localmente.", "",
              "I fold development sono già stati usati per varie scelte. Gli intervalli condizionano sui fit e sulle soglie; non includono variabilità del campionamento, riaddestramento o selezione ripetuta. Rapporto 3:1 e pilot fissati prima dei risultati: nessuna ricerca di iperparametri. L'undersampling cambia la prevalenza e, soprattutto con selezione hard, la distribuzione dei negativi: le probabilità grezze non sono calibrate sulla popolazione originale.", "",
              "Riproduzione: `python -m tools.phase36_undersampling prepare`, poi `python -m tools.phase36_undersampling run`, infine `python -m tools.phase36_undersampling report --publish`. Configurazione finale e submission non modificate.", ""]
    (ROOT / "results/eda/phase36_undersampling_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("REPORT phase36_undersampling_summary.md", flush=True)


def orchestrate():
    prepare()
    jobs, streams = {}, []
    try:
        for method in METHODS:
            directory = SHARED / method
            directory.mkdir(parents=True, exist_ok=True)
            stream = (directory / "run_stdout.log").open("a", encoding="utf-8")
            streams.append(stream)
            jobs[method] = subprocess.Popen([sys.executable, "-u", "-m", "tools.phase36_undersampling", "candidate", method],
                cwd=ROOT, stdout=stream, stderr=stream, creationflags=subprocess.CREATE_NO_WINDOW)
            print(f"STARTED {method} pid={jobs[method].pid}", flush=True)
        remaining = set(jobs)
        while remaining:
            for method in list(remaining):
                if jobs[method].poll() is not None:
                    print(f"FINISHED {method} exit={jobs[method].returncode}", flush=True)
                    remaining.remove(method)
            write(SHARED / "status.json", {"updated_at_utc": datetime.now(timezone.utc).isoformat(),
                  "trials": {m: {"pid": p.pid, "exit_code": p.poll(), "status": "running" if p.poll() is None else "completed" if p.returncode == 0 else "failed"} for m, p in jobs.items()}})
            if remaining:
                time.sleep(5)
        if any(p.returncode != 0 for p in jobs.values()):
            raise RuntimeError("Candidate failed; inspect logs and resume saved checkpoints.")
        report()
    finally:
        for stream in streams:
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "candidate", "compare", "report"))
    parser.add_argument("method", nargs="?", choices=METHODS)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if args.action in ("candidate", "compare") and not args.method:
        parser.error("A method is required.")
    if args.action == "prepare":
        prepare()
    elif args.action == "run":
        orchestrate()
    elif args.action == "candidate":
        run(args.method)
    elif args.action == "compare":
        compare(args.method)
    else:
        report(args.publish)


if __name__ == "__main__":
    main()

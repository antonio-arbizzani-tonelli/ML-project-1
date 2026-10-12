"""Historical Phase 14 random undersampling at 5:1 and 7:1.

An isolated Phase 35 numerical snapshot preserves the historical split search.
Samples are nested prefixes of a shared random negative permutation per fit.
The standard runner owns the unchanged validation folds and nested thresholds.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import copy
import gc
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "results/boosting_artifacts/phase37_undersampling_ratios"
RUNTIME = SHARED / "source_snapshot"
HISTORICAL = ROOT / "results/boosting_artifacts/phase35_positive_weights/source_snapshot"
REFERENCE = "configs/experiments/phase14_boosting_codebook_corrections_only.json"
BASELINE = "results/experiments/20260921T105139261213Z_phase14-boosting-codebook-corrections-only-depth5-features128-200trees.json"
RATIOS = (5, 7)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap():
    """Import the verified historical numerical source before any project module."""
    expected = read(HISTORICAL / "ready.json")
    marker = RUNTIME / "ready.json"
    if not marker.exists():
        for name, value in expected.items():
            if checksum(HISTORICAL / name) != value:
                raise ValueError(f"Historical snapshot changed: {name}")
            destination = RUNTIME / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(HISTORICAL / name, destination)
        write(marker, expected)
    if read(marker) != expected:
        raise ValueError("The runtime no longer matches the historical snapshot.")
    for name, value in expected.items():
        if checksum(RUNTIME / name) != value:
            raise ValueError(f"Frozen runtime changed: {name}")
    if any(name == "src" or name.startswith("src.") for name in sys.modules):
        raise RuntimeError("Load this historical driver in a fresh process before other src imports.")
    sys.path.insert(0, str(RUNTIME))


if __name__ == "__main__" or os.environ.get("PHASE37_HISTORICAL_RUNTIME") == "1":
    bootstrap()
import numpy as np
from src import run_boosting_cv as runner
from src.evaluation import average_precision, best_f1_threshold, binary_log_loss, classification_metrics
from src.feature_preprocessing import tree_feature_matrices
from src.numpy_boosting import HistogramBinner, HistogramGradientBoostingClassifier
from src.run_preprocessing_cv import _stratified_folds
from src.summarize_phase19_transfer import _arrays, _paired_interval, _publish
from tools.targeted_preprocessing_analysis import _input_data


def require_historical_runtime():
    if Path(runner.__file__).resolve() != (RUNTIME / "src/run_boosting_cv.py").resolve():
        raise RuntimeError("Execute through this driver in a fresh process to use the historical snapshot.")


def suite_path(ratio):
    return ROOT / f"configs/experiments/phase37_phase14_random_ratio{ratio}.json"


def configuration(ratio):
    suite = read(suite_path(ratio))
    expected = copy.deepcopy(read(ROOT / REFERENCE))
    expected.update(experiment_prefix=f"phase37_phase14_random_ratio{ratio}",
                    artifact_dir=f"results/boosting_artifacts/phase37_undersampling_ratios/ratio{ratio}",
                    sampling={"method": "random_permutation_prefix", "negative_to_positive_ratio": ratio,
                              "seed": 20261007, "retain_all_positives": True})
    if {k: v for k, v in suite.items() if k != "hypothesis"} != expected:
        raise ValueError("Only sampling and provenance may differ from historical Phase 14.")
    runner._validate_suite(suite)
    return suite


def sources():
    paths = {p.relative_to(RUNTIME).as_posix(): p for p in (RUNTIME / "src").glob("*.py")}
    paths.update({"tools/phase37_undersampling_ratios.py": Path(__file__),
                  "tools/targeted_preprocessing_analysis.py": ROOT / "tools/targeted_preprocessing_analysis.py"})
    return {name: checksum(path) for name, path in paths.items()}


def inputs(suite):
    paths = {"preprocessing_config": suite["preprocessing_config_path"], "feature_metadata": suite["feature_metadata_path"]}
    paths.update({key: suite["data"][field] for key, field in (
        ("features", "features_path"), ("labels", "labels_path"), ("split_indices", "split_indices_path"))})
    return {key: checksum(ROOT / path) for key, path in paths.items()}


def frozen(manifest, check_inputs=False):
    if sources() != manifest["source_sha256"]:
        raise ValueError("Execution sources changed after preflight.")
    for path, value in manifest["control_sha256"].items():
        if checksum(ROOT / path) != value:
            raise ValueError(f"Control or suite changed: {path}")
    if check_inputs and inputs(configuration(5)) != manifest["input_sha256"]:
        raise ValueError("Dataset or preprocessing changed.")
    if {"python": platform.python_version(), "numpy": np.__version__} != manifest["environment"]:
        raise ValueError("Numerical environment changed.")


def signature(train, labels, valid):
    return {"train_shape": list(train.shape), "validation_shape": list(valid.shape),
            "train_sha256": runner._array_sha256(train), "labels_sha256": runner._array_sha256(labels),
            "validation_sha256": runner._array_sha256(valid)}


def select_positions(labels, ratio, seed):
    """All positives plus a uniform, nested prefix of randomly ordered negatives."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.all(np.isin(labels, [0, 1])) or type(ratio) is not int or ratio < 1:
        raise ValueError("Sampling requires a binary vector and a positive integer ratio.")
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    if not positive.size or not negative.size:
        raise ValueError("Both classes must be present.")
    ordered = np.random.default_rng(seed).permutation(negative)
    return np.sort(np.r_[positive, ordered[:min(negative.size, ratio * positive.size)]])


def partitions(labels, development, suite):
    outer = list(_stratified_folds(labels[development], 3, suite["seed"]))
    for fold, (train, valid) in enumerate(outer, 1):
        yield f"outer_{fold}", development[train], development[valid]
    for fold, (train, _) in enumerate(outer, 1):
        for inner, (it, iv) in enumerate(_stratified_folds(labels[development[train]], 2,
                suite["nested_threshold"]["inner_seed"] + fold), 1):
            yield f"outer_{fold}_inner_{inner}", development[train[it]], development[train[iv]]


def prepare():
    require_historical_runtime()
    path = SHARED / "preflight.json"
    if path.exists():
        manifest = read(path)
        frozen(manifest, True)
        return manifest
    suites = {ratio: configuration(ratio) for ratio in RATIOS}
    suite, baseline = suites[5], read(ROOT / BASELINE)
    fingerprint = inputs(suite)
    if any(value != baseline["outcome"]["input_sha256"][key] for key, value in fingerprint.items()):
        raise ValueError("Historical baseline inputs differ.")
    ba = _arrays(ROOT, baseline)
    x, labels, development, names, metadata = _input_data(ROOT, suite)
    np.testing.assert_array_equal(development, ba["development_indices"])
    np.testing.assert_array_equal(labels[development], ba["labels"])
    plan = runner._variant(read(ROOT / suite["preprocessing_config_path"]), suite["preprocessing_variant"])["preprocessing"]
    controls = [BASELINE, REFERENCE, baseline["outcome"]["oof_predictions_artifact"]["path"]]
    controls += [entry["path"] for entry in baseline["outcome"]["model_checkpoints"]]
    controls += [suite_path(r).relative_to(ROOT).as_posix() for r in RATIOS]
    manifest = {"source_sha256": sources(), "input_sha256": fingerprint,
                "control_sha256": {p: checksum(ROOT / p) for p in controls}, "plan": plan,
                "environment": {"python": platform.python_version(), "numpy": np.__version__},
                "historical_source": str(HISTORICAL.relative_to(ROOT)).replace("\\", "/"),
                "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                "observed_final_config_sha256": checksum(ROOT / "configs/final_model.json"),
                "partitions": [], "baseline_checkpoint_checks": []}
    for index, (name, train_rows, valid_rows) in enumerate(partitions(labels, development, suite)):
        row_path = SHARED / "partitions" / f"{name}.npz"
        row_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(row_path, training_rows=train_rows, validation_rows=valid_rows)
        train, valid, output = tree_feature_matrices(x[train_rows], x[valid_rows], names, metadata, plan)
        if len(output) != 295 or np.intersect1d(train_rows, valid_rows).size:
            raise ValueError("Unexpected representation or overlapping rows.")
        entry = {"name": name, "rows_path": row_path.relative_to(ROOT).as_posix(), "rows_sha256": checksum(row_path),
                 "signature": signature(train, labels[train_rows], valid), "output_names": output,
                 "eligible_rows": len(train_rows), "positive_rows": int(labels[train_rows].sum()),
                 "validation_rows": len(valid_rows), "retained_rows": {str(r): (r+1)*int(labels[train_rows].sum()) for r in RATIOS}}
        selected5 = select_positions(labels[train_rows], 5, 20261008 + index)
        selected7 = select_positions(labels[train_rows], 7, 20261008 + index)
        if not np.all(np.isin(selected5, selected7)):
            raise ValueError("Random samples are not nested.")
        if index < 3:
            cp = baseline["outcome"]["model_checkpoints"][index]
            model, saved = HistogramGradientBoostingClassifier.load_checkpoint(ROOT / cp["path"])
            np.testing.assert_array_equal(saved["training_row_indices"], train_rows)
            mask = ba["fold_ids"] == index + 1
            np.testing.assert_array_equal(development[mask], valid_rows)
            np.testing.assert_array_equal(model.predict_proba(valid)[:, 1], ba["probabilities"][mask])
            manifest["baseline_checkpoint_checks"].append({"fold": index+1, "maximum_probability_difference": 0.0})
            del model
        manifest["partitions"].append(entry)
        print(f"PREFLIGHT {name}: eligible={len(train_rows)}, ratio5={len(selected5)}, ratio7={len(selected7)}; original validation", flush=True)
        del train, valid
        gc.collect()
    frozen(manifest, True)
    for name in ("tools/phase37_undersampling_ratios.py", "tools/targeted_preprocessing_analysis.py"):
        target = RUNTIME / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    write(path, manifest)
    return manifest


def run(ratio):
    require_historical_runtime()
    manifest, suite = read(SHARED / "preflight.json"), configuration(ratio)
    frozen(manifest, True)
    directory = ROOT / suite["artifact_dir"]
    if list((directory / "records").glob("*.json")):
        compare(ratio)
        return
    x, _, _, names, metadata = _input_data(ROOT, suite)
    parameters = runner._resolved_parameters(suite["models"][0]["parameters"], manifest["partitions"][0]["output_names"])
    profiles, outer_checkpoints = [], []

    def fit(eligible, labels, valid, actual_parameters, checkpoints, callback=None):
        number = len(profiles) + 1
        task = manifest["partitions"][number - 1]
        if signature(eligible, labels, valid) != task["signature"] or actual_parameters != parameters or checkpoints != [200]:
            raise ValueError("Runner inputs, fit order or parameters differ from preflight.")
        row_path = ROOT / task["rows_path"]
        if checksum(row_path) != task["rows_sha256"]:
            raise ValueError("Eligible partition changed.")
        with np.load(row_path) as rows:
            train_rows, valid_rows = rows["training_rows"], rows["validation_rows"]
        selected = select_positions(labels, ratio, suite["sampling"]["seed"] + number)
        if len(selected) != (ratio+1)*int(labels.sum()) or int(labels[selected].sum()) != int(labels.sum()):
            raise ValueError("Wrong row budget or dropped positives.")
        cache = directory / "fit_cache" / task["name"]
        cache.mkdir(parents=True, exist_ok=True)
        selection_path = cache / "selection.npz"
        if selection_path.exists():
            with np.load(selection_path) as saved:
                np.testing.assert_array_equal(saved["positions"], selected)
                np.testing.assert_array_equal(saved["eligible_training_rows"], train_rows)
                np.testing.assert_array_equal(saved["selected_training_rows"], train_rows[selected])
        else:
            np.savez_compressed(selection_path, positions=selected, eligible_training_rows=train_rows,
                                selected_training_rows=train_rows[selected])
        train, validation, output = tree_feature_matrices(x[train_rows[selected]], x[valid_rows], names, metadata, manifest["plan"])
        if output != task["output_names"]:
            raise ValueError("Sampling changed output columns.")
        checkpoint = cache / "model.pkl"
        saved_metadata = {"ratio": ratio, "partition": task["name"], "parameters": parameters,
                          "signature": signature(train, labels[selected], validation),
                          "source_sha256": manifest["source_sha256"], "selection_sha256": checksum(selection_path),
                          "training_row_indices": train_rows[selected], "eligible_training_row_indices": train_rows,
                          "validation_row_indices": valid_rows, "output_feature_names": output}
        started = time.perf_counter()

        def progress(step, model):
            model.save_checkpoint(checkpoint, saved_metadata)
            print(f"PROGRESS ratio={ratio} fit={number}/9 trees={step}/200 elapsed={time.perf_counter()-started:.1f}s", flush=True)

        prior = 0
        if checkpoint.exists():
            model, saved = HistogramGradientBoostingClassifier.load_checkpoint(checkpoint)
            for key, value in saved_metadata.items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(saved[key], value)
                elif saved[key] != value:
                    raise ValueError(f"Checkpoint mismatch: {key}")
            prior = len(model.trees_)
            if not 0 < prior <= 200 or prior % 25:
                raise ValueError("Invalid partial tree count.")
            if prior < 200:
                model.continue_fit(train, labels[selected], 200-prior, list(range(prior+25, 201, 25)), progress)
        else:
            print(f"START ratio={ratio} fit={number}/9 retained={len(selected)}/{len(labels)}", flush=True)
            model = HistogramGradientBoostingClassifier(**parameters, n_estimators=200)
            model.fit(train, labels[selected], list(range(25, 201, 25)), progress)
        probabilities = model.predict_proba(validation)[:, 1]
        profile = runner._tree_profile(model.tree_fit_seconds_)
        profile.update(fit_and_validation_prediction_seconds=time.perf_counter()-started,
                       bin_count_minimum=int(model.binner_.bin_counts_.min()), bin_count_maximum=int(model.binner_.bin_counts_.max()))
        entry = {"partition": task["name"], "ratio": ratio, "seed": suite["sampling"]["seed"] + number,
                 "eligible_rows": len(train_rows), "retained_rows": len(selected), "retained_positives": int(labels.sum()),
                 "retained_negatives": ratio*int(labels.sum()), "validation_rows": len(valid_rows),
                 "selection_path": selection_path.relative_to(ROOT).as_posix(), "selection_sha256": checksum(selection_path),
                 "model_path": checkpoint.relative_to(ROOT).as_posix(), "model_sha256": checksum(checkpoint),
                 "fit_and_prediction_seconds": profile["fit_and_validation_prediction_seconds"],
                 "tree_seconds": profile["total_seconds"], "resumed_from_tree_count": prior}
        profiles.append(entry)
        write(cache / "completed.json", {"fit": entry, "profile": profile})
        if number <= 3:
            outer_checkpoints.append({"fold": number, "path": entry["model_path"], "sha256": entry["model_sha256"],
                                      "training_rows": len(selected), "eligible_training_rows": len(train_rows)})
        print(f"DONE ratio={ratio} fit={number}/9 seconds={entry['fit_and_prediction_seconds']:.1f}", flush=True)
        return {200: probabilities}, profile

    original_fit, original_save = runner._fit_checkpoint_probabilities, runner.save_experiment

    def save(config, outcome, output):
        frozen(manifest, True)
        config.update(hypothesis=suite["hypothesis"], sampling=suite["sampling"])
        outcome.update(source_sha256=manifest["source_sha256"], environment=manifest["environment"],
                       individual_fit_profiles=profiles, model_checkpoints=outer_checkpoints,
                       isolated_historical_implementation=True, source_snapshot=RUNTIME.relative_to(ROOT).as_posix(),
                       sampling_fit_partition="Corresponding outer/inner training rows only; validation retains original prevalence.",
                       fit_and_prediction_seconds_sum=sum(p["fit_and_prediction_seconds"] for p in profiles),
                       runtime_scope="Full runner wall time includes preprocessing and sampling; per-fit sum excludes preprocessing. Resume counts recorded.")
        for index, detail in enumerate(outcome["folds"]):
            detail["retained_training_rows"] = profiles[index]["retained_rows"]
        return original_save(config, outcome, output)

    runner._fit_checkpoint_probabilities, runner.save_experiment = fit, save
    try:
        records = runner.run_suite(suite_path(ratio), ROOT, directory / "records")
        if len(records) != 1 or len(profiles) != 9:
            raise ValueError("Expected one record and nine completed fits.")
    finally:
        runner._fit_checkpoint_probabilities, runner.save_experiment = original_fit, original_save
    compare(ratio)


def compare(ratio):
    require_historical_runtime()
    manifest, suite = read(SHARED / "preflight.json"), configuration(ratio)
    frozen(manifest, True)
    directory = ROOT / suite["artifact_dir"]
    records = list((directory / "records").glob("*.json"))
    if len(records) != 1:
        raise ValueError("Expected one completed record.")
    candidate, baseline = read(records[0]), read(ROOT / BASELINE)
    outcome = candidate["outcome"]
    ca, ba = _arrays(ROOT, candidate), _arrays(ROOT, baseline)
    for key in ("development_indices", "fold_ids", "labels"):
        np.testing.assert_array_equal(ca[key], ba[key])
    for key, value in manifest["input_sha256"].items():
        if outcome["input_sha256"][key] != value:
            raise ValueError(f"Candidate input changed: {key}")
    for key in ("model", "features", "split"):
        if candidate["config"][key] != baseline["config"][key]:
            raise ValueError(f"Unexpected candidate change: {key}")
    if outcome["source_sha256"] != manifest["source_sha256"] or candidate["config"]["sampling"] != suite["sampling"]:
        raise ValueError("Candidate source or sampling provenance differs.")
    nested, bn = outcome["nested_threshold_evaluation"], baseline["outcome"]["nested_threshold_evaluation"]
    np.testing.assert_array_equal(ca["nested_predictions"], ca["probabilities"] >= ca["nested_thresholds"])
    counts, metrics = classification_metrics(ca["labels"], ca["nested_predictions"])
    metrics.update(average_precision=average_precision(ca["labels"], ca["probabilities"]), log_loss=binary_log_loss(ca["labels"], ca["probabilities"]))
    if metrics != nested["pooled_metrics"] or vars(counts) != nested["pooled_confusion_counts"]:
        raise ValueError("Pooled metrics do not reproduce.")
    x, labels, _, names, metadata = _input_data(ROOT, suite)
    profiles, checks = outcome["individual_fit_profiles"], []
    if len(profiles) != 9:
        raise ValueError("Incomplete fit count.")
    for number, (task, profile) in enumerate(zip(manifest["partitions"], profiles), 1):
        with np.load(ROOT / task["rows_path"]) as rows:
            train_rows, valid_rows = rows["training_rows"], rows["validation_rows"]
        if checksum(ROOT / profile["selection_path"]) != profile["selection_sha256"]:
            raise ValueError("Selection artifact changed.")
        with np.load(ROOT / profile["selection_path"]) as selection:
            selected = selection["positions"]
            np.testing.assert_array_equal(selection["eligible_training_rows"], train_rows)
            np.testing.assert_array_equal(selection["selected_training_rows"], train_rows[selected])
        np.testing.assert_array_equal(selected, select_positions(labels[train_rows], ratio, profile["seed"]))
        if np.intersect1d(train_rows[selected], valid_rows).size or int(labels[train_rows[selected]].sum()) != int(labels[train_rows].sum()):
            raise ValueError("Sampling crossed a validation boundary or dropped positives.")
        if checksum(ROOT / profile["model_path"]) != profile["model_sha256"]:
            raise ValueError("Model checkpoint changed.")
        model, saved = HistogramGradientBoostingClassifier.load_checkpoint(ROOT / profile["model_path"])
        np.testing.assert_array_equal(saved["training_row_indices"], train_rows[selected])
        np.testing.assert_array_equal(saved["validation_row_indices"], valid_rows)
        train, valid, output = tree_feature_matrices(x[train_rows[selected]], x[valid_rows], names, metadata, manifest["plan"])
        if saved["signature"] != signature(train, labels[train_rows[selected]], valid) or output != saved["output_feature_names"]:
            raise ValueError("Checkpoint input signature changed.")
        for key, value in baseline["config"]["model"]["parameters"].items():
            if getattr(model, key) != value:
                raise ValueError(f"Checkpoint parameter differs: {key}")
        rebuilt = HistogramBinner(model.n_bins).fit(train)
        for actual, expected in zip(model.binner_.edges_, rebuilt.edges_):
            np.testing.assert_array_equal(actual, expected)
        positive_rate = float(labels[train_rows[selected]].mean())
        if model.base_score_ != float(np.log(positive_rate/(1-positive_rate))):
            raise ValueError("Unexpected fitted class prevalence.")
        scores = model.predict_proba(valid)[:, 1]
        if number <= 3:
            np.testing.assert_array_equal(scores, ca["probabilities"][ca["fold_ids"] == number])
            detail = nested["folds"][number - 1]
            info = detail["inner_oof_predictions_artifact"]
            if checksum(ROOT / info["path"]) != info["sha256"]:
                raise ValueError("Inner OOF artifact changed.")
            with np.load(ROOT / info["path"]) as inner:
                np.testing.assert_array_equal(inner["training_row_indices"], train_rows)
                np.testing.assert_array_equal(inner["outer_validation_row_indices"], valid_rows)
                np.testing.assert_array_equal(inner["labels"], labels[train_rows])
                ids = np.zeros(len(train_rows), dtype=np.int8)
                for inner_fold, (_, iv) in enumerate(_stratified_folds(labels[train_rows], 2, suite["nested_threshold"]["inner_seed"] + number), 1):
                    ids[iv] = inner_fold
                np.testing.assert_array_equal(inner["fold_ids"], ids)
                threshold = best_f1_threshold(inner["labels"], inner["probabilities"]).threshold
            if threshold != detail["inner_threshold"]:
                raise ValueError("Inner threshold changed.")
            np.testing.assert_array_equal(ca["nested_thresholds"][ca["fold_ids"] == number], np.full(len(valid_rows), threshold))
            fm, fc = runner._metrics(labels[valid_rows], scores, threshold)
            if fm != detail["outer_evaluation_metrics"] or fc != detail["outer_evaluation_confusion_counts"]:
                raise ValueError("Fold metrics differ.")
        else:
            outer_fold = (number-4)//2 + 1
            info = nested["folds"][outer_fold-1]["inner_oof_predictions_artifact"]
            with np.load(ROOT / info["path"]) as inner:
                mask = np.isin(inner["training_row_indices"], valid_rows)
                np.testing.assert_array_equal(inner["training_row_indices"][mask], valid_rows)
                np.testing.assert_array_equal(scores, inner["probabilities"][mask])
        checks.append({"partition": task["name"], "checkpoint_probability_difference": 0.0,
                       "retained_row_binner_edges_rebuilt": True, "retained_rows": len(selected)})
        print(f"REPLAY ratio={ratio} {task['name']}: exact scores, rows and bins", flush=True)
        del model, train, valid, rebuilt
        gc.collect()
    deltas = {k: v-bn["pooled_metrics"][k] for k, v in metrics.items()}
    fold_deltas = [f["outer_evaluation_metrics"]["f1"]-b["outer_evaluation_metrics"]["f1"] for f, b in zip(nested["folds"], bn["folds"])]
    summary = {"ratio": ratio, "candidate_record": records[0].relative_to(ROOT).as_posix(), "baseline_record": BASELINE,
               "nested": nested, "metric_deltas": deltas, "fold_f1_deltas": fold_deltas,
               "positive_fold_count": sum(d > 0 for d in fold_deltas), "individual_fit_profiles": profiles,
               "paired_conditional_f1_delta_interval_95": _paired_interval(ca["labels"], ba["nested_predictions"], ca["nested_predictions"]),
               "bootstrap_scope": "20,000 paired stratified draws; fixed fits, samples and thresholds; excludes retraining, sampling and repeated selection uncertainty.",
               "verification": checks, "complete_runner_seconds": outcome["runtime_seconds"],
               "final_config_changed_since_preflight": checksum(ROOT / "configs/final_model.json") != manifest["observed_final_config_sha256"]}
    write(directory / "comparison.json", summary)
    print(f"VERIFIED ratio={ratio} F1={metrics['f1']:.9f} delta={deltas['f1']:+.9f}", flush=True)
    return summary


def report(publish=False):
    baseline = read(ROOT / BASELINE)
    candidates = [read(SHARED / f"ratio{ratio}/comparison.json") for ratio in RATIOS]
    for candidate in candidates:
        if publish:
            candidate["published_record"] = _publish(ROOT / candidate["candidate_record"], ROOT).relative_to(ROOT).as_posix()
            write(SHARED / f"ratio{candidate['ratio']}/comparison.json", candidate)
    # Confirm nesting on all nine actual saved selections, not only synthetic samples.
    for p5, p7 in zip(candidates[0]["individual_fit_profiles"], candidates[1]["individual_fit_profiles"]):
        with np.load(ROOT / p5["selection_path"]) as f5, np.load(ROOT / p7["selection_path"]) as f7:
            np.testing.assert_array_equal(f5["eligible_training_rows"], f7["eligible_training_rows"])
            if not np.all(np.isin(f5["positions"], f7["positions"])):
                raise ValueError("Completed samples are not nested.")
    bn = baseline["outcome"]["nested_threshold_evaluation"]
    aggregate = {"baseline_record": BASELINE, "baseline_nested": bn, "candidates": candidates,
                 "nested_sample_checks": 9, "historical_implementation": True, "final_model_modified_by_experiment": False,
                 "scope": "Phase 14 historical control, 295 columns, 200 trees, three outer and two inner folds; one shared negative permutation seed per fit."}
    write(ROOT / "results/experiments/phase37_undersampling_ratios_comparison.json", aggregate)
    lines = ["# Phase 37 — Undersampling casuale 5:1 e 7:1 su Phase 14", "",
             "Confronto sul miglior modello storico Phase 14 (F1 nested 0,442155227), usando lo snapshot numerico storico già verificato per Phase 35. Tutti i positivi restano nel training. I negativi seguono una permutazione casuale condivisa per ciascun fit: il campione 5:1 è contenuto nel 7:1. Tutte le righe di validazione sono conservate; preprocessing e bin apprendono solo dalle righe selezionate. Soglie scelte tramite OOF interno. Holdout e test non valutati.", "",
             "Stessi 295 predittori, parametri, seed e partizioni del controllo. Nove fit per candidato; tre esterni e sei interni. Il risultato 3:1 di Phase 36 usa invece la versione corretta Phase 34 e non costituisce un controllo storico comparabile.", "",
             "| Pipeline storica | F1 | Precision | Recall | AP | Falsi positivi | Delta F1 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    rows = [("Tutte le righe — Phase 14", bn, 0.0)]
    rows += [(f"Casuale {c['ratio']}:1", c["nested"], c["metric_deltas"]["f1"]) for c in candidates]
    for name, nested, delta in rows:
        m, counts = nested["pooled_metrics"], nested["pooled_confusion_counts"]
        lines.append(f"| {name} | {m['f1']:.9f} | {m['precision']:.9f} | {m['recall']:.9f} | {m['average_precision']:.9f} | {counts['false_positives']} | {delta:+.9f} |")
    for c in candidates:
        lines += ["", f"## Rapporto {c['ratio']}:1", "", "| Fold | F1 | Delta F1 | Soglia interna |", "| --- | ---: | ---: | ---: |"]
        for i, fold in enumerate(c["nested"]["folds"]):
            lines.append(f"| {i+1} | {fold['outer_evaluation_metrics']['f1']:.9f} | {c['fold_f1_deltas'][i]:+.9f} | {fold['inner_threshold']:.9f} |")
        ci = c["paired_conditional_f1_delta_interval_95"]
        profiles = c["individual_fit_profiles"]
        lines += ["", f"Fold favorevoli: {c['positive_fold_count']}/3. Intervallo condizionale del delta F1: [{ci[0]:+.9f}, {ci[1]:+.9f}].",
                  f"Training esterni conservati: {', '.join(str(p['retained_rows']) for p in profiles[:3])}; tutti i positivi. Runner completo: {c['complete_runner_seconds']/60:.1f} minuti; i due candidati eseguiti in parallelo. Checkpoint ripresi: {sum(p['resumed_from_tree_count']>0 for p in profiles)}/9.",
                  f"Record: `{c.get('published_record', c['candidate_record'])}`."]
    lines += ["", "Verifiche: replay dei tre modelli del controllo; replay di tutti i 18 modelli nuovi e di tutte le probabilità OOF interne; ricostruzione delle soglie dei bin dalle righe conservate; riproduzione delle soglie interne e delle metriche; tutti i positivi conservati e tutte le validazioni escluse dai training; campioni 5:1 contenuti nei 7:1 in tutti i nove fit. Snapshot e checkpoint ogni 25 alberi conservati localmente.", "",
              "Un solo seed di campionamento e rapporti fissati prima dei risultati. I fold development sono già stati consultati per varie scelte. Il bootstrap condiziona sui fit, sui campioni e sulle soglie, senza includere l'incertezza da riaddestramento o selezione ripetuta. Le probabilità grezze riflettono la prevalenza modificata dal campionamento. L'attuale configurazione corretta Phase 34 e la submission non sono modificate da questi test storici.", "",
              "Riproduzione: `python -m tools.phase37_undersampling_ratios prepare`, poi `python -m tools.phase37_undersampling_ratios run`, infine `python -m tools.phase37_undersampling_ratios report --publish`. Richiede i dati e lo snapshot storico locale di Phase 35.", ""]
    (ROOT / "results/eda/phase37_undersampling_ratios_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("REPORT phase37_undersampling_ratios_summary.md", flush=True)


def orchestrate():
    prepare()
    jobs, streams = {}, []
    try:
        for ratio in RATIOS:
            directory = SHARED / f"ratio{ratio}"
            directory.mkdir(parents=True, exist_ok=True)
            stream = (directory / "run_stdout.log").open("a", encoding="utf-8")
            streams.append(stream)
            jobs[ratio] = subprocess.Popen([sys.executable, "-u", "-m", "tools.phase37_undersampling_ratios", "candidate", str(ratio)],
                cwd=ROOT, stdout=stream, stderr=stream, creationflags=subprocess.CREATE_NO_WINDOW)
            print(f"STARTED ratio={ratio} pid={jobs[ratio].pid}", flush=True)
        remaining = set(jobs)
        while remaining:
            for ratio in list(remaining):
                if jobs[ratio].poll() is not None:
                    print(f"FINISHED ratio={ratio} exit={jobs[ratio].returncode}", flush=True)
                    remaining.remove(ratio)
            write(SHARED / "status.json", {"updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "trials": {str(r): {"pid": p.pid, "exit_code": p.poll(), "status": "running" if p.poll() is None else "completed" if p.returncode==0 else "failed"} for r, p in jobs.items()}})
            if remaining:
                time.sleep(5)
        if any(p.returncode != 0 for p in jobs.values()):
            raise RuntimeError("A candidate failed; inspect its log. Saved checkpoints remain resumable.")
        report()
    finally:
        for stream in streams:
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "candidate", "compare", "report"))
    parser.add_argument("ratio", nargs="?", type=int, choices=RATIOS)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if args.action in ("candidate", "compare") and args.ratio is None:
        parser.error("A ratio is required.")
    if args.action == "prepare":
        prepare()
    elif args.action == "run":
        orchestrate()
    elif args.action == "candidate":
        run(args.ratio)
    elif args.action == "compare":
        compare(args.ratio)
    else:
        report(args.publish)


if __name__ == "__main__":
    main()

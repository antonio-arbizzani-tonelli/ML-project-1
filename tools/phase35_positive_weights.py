"""Controlled, resumable weight trials using the exact Phase 30 implementation.

The active booster has a subsequent split-search correction. These experiments
use an isolated source copy so that only the positive-class weight changes.
The standard CV runner still owns partitions, thresholds and final artifacts.
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
SHARED = ROOT / "results/boosting_artifacts/phase35_positive_weights"
RUNTIME = SHARED / "source_snapshot"
REFERENCE = "configs/experiments/phase14_boosting_codebook_corrections_only.json"
WEIGHT2 = "results/experiments/20261001T000013003849Z_phase30-phase14-positive-weight2-depth5-features128-200trees.json"
TRIALS = {"1p25": 1.25, "1p5": 1.5}


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
    """Freeze imports before importing NumPy or any numerical project module."""
    historical = read(ROOT / WEIGHT2)["outcome"]["source_sha256"]
    marker = RUNTIME / "ready.json"
    if not marker.exists():
        (RUNTIME / "src").mkdir(parents=True, exist_ok=True)
        for source in (ROOT / "src").glob("*.py"):
            shutil.copyfile(source, RUNTIME / "src" / source.name)
        original = ROOT / "results/boosting_artifacts/phase30_31_parallel/source_snapshot"
        for name, expected in historical.items():
            if checksum(original / name) != expected:
                raise ValueError(f"Historical source snapshot changed: {name}")
            shutil.copyfile(original / name, RUNTIME / name)
        write(marker, {str(p.relative_to(RUNTIME)).replace("\\", "/"): checksum(p)
                       for p in (RUNTIME / "src").glob("*.py")})
    for name, expected in read(marker).items():
        if checksum(RUNTIME / name) != expected:
            raise ValueError(f"Frozen execution source changed: {name}")
    for name, expected in historical.items():
        if checksum(RUNTIME / name) != expected:
            raise ValueError(f"Execution differs from Phase 30: {name}")
    sys.path.insert(0, str(RUNTIME))


bootstrap()
import numpy as np
from src import run_boosting_cv as runner
from src.evaluation import average_precision, best_f1_threshold, binary_log_loss, classification_metrics
from src.feature_preprocessing import tree_feature_matrices
from src.numpy_boosting import HistogramBinner, HistogramGradientBoostingClassifier
from src.run_preprocessing_cv import _stratified_folds
from src.summarize_phase19_transfer import BASELINE, _arrays, _paired_interval
from tools.targeted_preprocessing_analysis import _input_data, _partitions


def suite_path(token):
    return ROOT / f"configs/experiments/phase35_phase14_positive_weight{token}.json"


def configuration(token):
    candidate, reference = read(suite_path(token)), read(ROOT / REFERENCE)
    expected = copy.deepcopy(reference)
    prefix = f"phase35_phase14_positive_weight{token}"
    expected.update(experiment_prefix=prefix, artifact_dir=f"results/boosting_artifacts/{prefix}")
    expected["models"][0]["parameters"]["positive_class_weight"] = TRIALS[token]
    if {k: v for k, v in candidate.items() if k != "hypothesis"} != expected:
        raise ValueError("The candidate must change only positive-class weight and provenance.")
    runner._validate_suite(candidate)
    return candidate


def sources():
    paths = {str(p.relative_to(RUNTIME)).replace("\\", "/"): p
             for p in (RUNTIME / "src").glob("*.py")}
    paths.update({"tools/phase35_positive_weights.py": Path(__file__),
                  "tools/targeted_preprocessing_analysis.py": ROOT / "tools/targeted_preprocessing_analysis.py"})
    return {name: checksum(path) for name, path in paths.items()}


def fingerprint_inputs(suite):
    paths = {"preprocessing_config": suite["preprocessing_config_path"],
             "feature_metadata": suite["feature_metadata_path"]}
    paths.update({key: suite["data"][field] for key, field in (
        ("features", "features_path"), ("labels", "labels_path"),
        ("split_indices", "split_indices_path"))})
    return {key: checksum(ROOT / path) for key, path in paths.items()}


def frozen(manifest, inputs=False):
    if sources() != manifest["source_sha256"]:
        raise ValueError("Execution sources changed after preflight.")
    for name, expected in manifest["control_sha256"].items():
        if checksum(ROOT / name) != expected:
            raise ValueError(f"Control or candidate configuration changed: {name}")
    if inputs and fingerprint_inputs(configuration("1p25")) != manifest["input_sha256"]:
        raise ValueError("Data or preprocessing changed after preflight.")
    if {"python": platform.python_version(), "numpy": np.__version__} != manifest["environment"]:
        raise ValueError("Numerical environment changed.")


def signature(train, labels, valid):
    return {"train_shape": list(train.shape), "validation_shape": list(valid.shape),
            "train_sha256": runner._array_sha256(train),
            "labels_sha256": runner._array_sha256(labels),
            "validation_sha256": runner._array_sha256(valid)}


def prepare():
    path = SHARED / "preflight.json"
    if path.exists():
        manifest = read(path)
        frozen(manifest, inputs=True)
        return manifest
    suites = {token: configuration(token) for token in TRIALS}
    suite = suites["1p25"]
    baseline, arrays = read(ROOT / BASELINE), None
    inputs = fingerprint_inputs(suite)
    for key, value in inputs.items():
        if value != baseline["outcome"]["input_sha256"][key]:
            raise ValueError(f"Historical baseline input changed: {key}")
    arrays = _arrays(ROOT, baseline)
    x, labels, development, names, metadata = _input_data(ROOT, suite)
    np.testing.assert_array_equal(development, arrays["development_indices"])
    np.testing.assert_array_equal(labels[development], arrays["labels"])
    plan = runner._variant(read(ROOT / suite["preprocessing_config_path"]),
                           suite["preprocessing_variant"])["preprocessing"]
    controls = [BASELINE, WEIGHT2, REFERENCE, "configs/final_model.json"]
    controls += [str(suite_path(t).relative_to(ROOT)).replace("\\", "/") for t in TRIALS]
    manifest = {"source_sha256": sources(), "input_sha256": inputs,
                "control_sha256": {p: checksum(ROOT / p) for p in controls},
                "environment": {"python": platform.python_version(), "numpy": np.__version__},
                "prepared_at_utc": datetime.now(timezone.utc).isoformat(), "partitions": [],
                "baseline_checkpoint_checks": [], "trained_models": 0}
    previous = read(ROOT / "results/eda/phase30_phase14_positive_weight2_preflight.json")
    for index, (name, train, valid) in enumerate(_partitions(labels, development, suite)):
        ct, cv, output = tree_feature_matrices(x[development[train]], x[development[valid]], names, metadata, plan)
        if len(output) != 295 or output != previous["partitions"][index]["output_names"]:
            raise ValueError("Feature representation differs from Phase 30.")
        entry = {"partition": name, "signature": signature(ct, labels[development[train]], cv),
                 "feature_count": len(output), "training_rows": len(train), "validation_rows": len(valid),
                 "training_row_indices_sha256": runner._array_sha256(development[train]),
                 "training_labels_sha256": runner._array_sha256(labels[development[train]]),
                 "output_names": output, "bin_counts": HistogramBinner(64).fit(ct).bin_counts_.tolist()}
        for key in ("partition", "training_rows", "validation_rows", "training_row_indices_sha256",
                    "training_labels_sha256", "bin_counts"):
            if entry[key] != previous["partitions"][index][key]:
                raise ValueError(f"Historical partition or binning changed: {name}/{key}")
        if index < 3:
            cp = baseline["outcome"]["model_checkpoints"][index]
            if checksum(ROOT / cp["path"]) != cp["sha256"]:
                raise ValueError("Baseline checkpoint fingerprint differs.")
            model, saved = HistogramGradientBoostingClassifier.load_checkpoint(ROOT / cp["path"])
            np.testing.assert_array_equal(saved["training_row_indices"], development[train])
            np.testing.assert_array_equal(arrays["fold_ids"][valid], np.full(len(valid), index + 1))
            np.testing.assert_array_equal(model.predict_proba(cv)[:, 1], arrays["probabilities"][valid])
            manifest["baseline_checkpoint_checks"].append({"fold": index + 1, "maximum_probability_difference": 0.0})
            del model
        manifest["partitions"].append(entry)
        print(f"PREFLIGHT {name}: {len(output)} features, historical rows/bins verified", flush=True)
        del ct, cv
        gc.collect()
    frozen(manifest, inputs=True)
    (RUNTIME / "tools").mkdir(exist_ok=True)
    shutil.copyfile(Path(__file__), RUNTIME / "tools/phase35_positive_weights.py")
    write(path, manifest)
    return manifest


def run(token):
    manifest = read(SHARED / "preflight.json")
    frozen(manifest, inputs=True)
    suite = configuration(token)
    directory = ROOT / suite["artifact_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    if list((directory / "records").glob("*.json")):
        compare(token)
        return
    parameters = runner._resolved_parameters(suite["models"][0]["parameters"], manifest["partitions"][0]["output_names"])
    profiles = []
    started = time.perf_counter()

    def fit(train, labels, valid, actual_parameters, checkpoints, callback=None):
        number = len(profiles) + 1
        expected = manifest["partitions"][number - 1]
        sig = signature(train, labels, valid)
        if sig != expected["signature"] or actual_parameters != parameters or checkpoints != [200]:
            raise ValueError("Standard runner matrices, order or parameters differ from preflight.")
        cache = directory / "fit_cache" / expected["partition"]
        cache.mkdir(parents=True, exist_ok=True)
        checkpoint = cache / "model.pkl"
        metadata = {"signature": sig, "parameters": parameters, "source_sha256": manifest["source_sha256"]}
        fit_started = time.perf_counter()
        steps = list(range(25, 201, 25))

        def progress(step, model):
            model.save_checkpoint(checkpoint, metadata)
            print(f"PROGRESS weight={TRIALS[token]} fit={number}/9 trees={step}/200 elapsed={time.perf_counter()-fit_started:.1f}s", flush=True)

        prior = 0
        if checkpoint.exists():
            model, saved = HistogramGradientBoostingClassifier.load_checkpoint(checkpoint)
            if saved != metadata:
                raise ValueError("Partial checkpoint does not match the frozen experiment.")
            prior = len(model.trees_)
            if not 0 < prior <= 200 or model.n_features_in_ != train.shape[1]:
                raise ValueError("Invalid partial model dimensions.")
            for key, value in parameters.items():
                if getattr(model, key) != value:
                    raise ValueError(f"Partial model parameter differs: {key}")
            if prior < 200:
                model.continue_fit(train, labels, 200-prior, [s for s in steps if s > prior], progress)
        else:
            print(f"START weight={TRIALS[token]} fit={number}/9 train={labels.size} validation={valid.shape[0]}", flush=True)
            model = HistogramGradientBoostingClassifier(**parameters, n_estimators=200).fit(train, labels, steps, progress)
        if model.binner_.bin_counts_.tolist() != expected["bin_counts"]:
            raise ValueError("Model bins differ from the historical representation.")
        scores = model.predict_proba(valid)[:, 1]
        if callback is not None:
            callback(200, model)
        profile = runner._tree_profile(model.tree_fit_seconds_)
        profile.update(fit_and_validation_prediction_seconds=time.perf_counter()-fit_started,
                       bin_count_minimum=int(model.binner_.bin_counts_.min()), bin_count_maximum=int(model.binner_.bin_counts_.max()))
        entry = {**expected, "fit_number": number, "stage": "outer" if number <= 3 else "inner",
                 "elapsed_seconds": time.perf_counter()-fit_started, "resumed_from_tree_count": prior}
        profiles.append(entry)
        write(cache / "completed.json", {"profile": profile, "checkpoint_sha256": checksum(checkpoint), "fit": entry})
        print(f"DONE weight={TRIALS[token]} fit={number}/9 elapsed={entry['elapsed_seconds']:.1f}s", flush=True)
        return {200: scores}, profile

    original_fit, original_save = runner._fit_checkpoint_probabilities, runner.save_experiment

    def save(config, outcome, output):
        frozen(manifest, inputs=True)
        config["hypothesis"] = suite["hypothesis"]
        outcome.update(environment=manifest["environment"], source_sha256=manifest["source_sha256"],
                       individual_fit_profiles=profiles, isolated_historical_implementation=True,
                       source_snapshot=str(RUNTIME.relative_to(ROOT)).replace("\\", "/"),
                       invocation_seconds=time.perf_counter()-started)
        return original_save(config, outcome, output)

    runner._fit_checkpoint_probabilities, runner.save_experiment = fit, save
    try:
        records = runner.run_suite(suite_path(token), ROOT, directory / "records")
        if len(records) != 1 or len(profiles) != 9:
            raise ValueError("Expected one record and nine completed fits.")
    finally:
        runner._fit_checkpoint_probabilities, runner.save_experiment = original_fit, original_save
    compare(token)


def compare(token):
    manifest, suite = read(SHARED / "preflight.json"), configuration(token)
    frozen(manifest, inputs=True)
    directory = ROOT / suite["artifact_dir"]
    records = list((directory / "records").glob("*.json"))
    if len(records) != 1:
        raise ValueError("Expected one completed record.")
    candidate, baseline = read(records[0]), read(ROOT / BASELINE)
    ca, ba = _arrays(ROOT, candidate), _arrays(ROOT, baseline)
    outcome = candidate["outcome"]
    if outcome["source_sha256"] != manifest["source_sha256"] or len(outcome["individual_fit_profiles"]) != 9:
        raise ValueError("Candidate source provenance or fit count differs.")
    for key, value in manifest["input_sha256"].items():
        if outcome["input_sha256"][key] != value:
            raise ValueError(f"Candidate input differs: {key}")
    if outcome["input_sha256"]["boosting_suite_config"] != checksum(suite_path(token)):
        raise ValueError("Executed candidate suite differs.")
    expected = {**suite["models"][0]["parameters"], "n_estimators": 200}
    if candidate["config"]["model"]["parameters"] != expected or candidate["config"]["features"] != baseline["config"]["features"] or candidate["config"]["split"] != baseline["config"]["split"]:
        raise ValueError("Candidate changes more than class weight.")
    for key in ("development_indices", "fold_ids", "labels"):
        np.testing.assert_array_equal(ca[key], ba[key])
    nested, bn = outcome["nested_threshold_evaluation"], baseline["outcome"]["nested_threshold_evaluation"]
    np.testing.assert_array_equal(ca["nested_predictions"], ca["probabilities"] >= ca["nested_thresholds"])
    counts, metrics = classification_metrics(ca["labels"], ca["nested_predictions"])
    metrics.update(average_precision=average_precision(ca["labels"], ca["probabilities"]), log_loss=binary_log_loss(ca["labels"], ca["probabilities"]))
    if metrics != nested["pooled_metrics"] or vars(counts) != nested["pooled_confusion_counts"]:
        raise ValueError("Pooled metrics do not reproduce.")
    x, labels, development, names, metadata = _input_data(ROOT, suite)
    plan = runner._variant(read(ROOT / suite["preprocessing_config_path"]), suite["preprocessing_variant"])["preprocessing"]
    checks = []
    for fold, (train, valid) in enumerate(_stratified_folds(labels[development], 3, suite["seed"]), 1):
        cp = outcome["model_checkpoints"][fold - 1]
        if checksum(ROOT / cp["path"]) != cp["sha256"]:
            raise ValueError("Candidate checkpoint fingerprint differs.")
        model, saved = HistogramGradientBoostingClassifier.load_checkpoint(ROOT / cp["path"])
        np.testing.assert_array_equal(saved["training_row_indices"], development[train])
        for key, value in expected.items():
            if getattr(model, key) != value:
                raise ValueError(f"Candidate checkpoint parameter differs: {key}")
        pos = TRIALS[token] * float(np.sum(labels[development[train]]))
        neg = float(len(train) - np.sum(labels[development[train]]))
        prevalence = float(np.clip(pos / (pos + neg), 1e-12, 1 - 1e-12))
        if model.base_score_ != float(np.log(prevalence / (1 - prevalence))):
            raise ValueError("Weighted initial prevalence differs.")
        ct, cv, output = tree_feature_matrices(x[development[train]], x[development[valid]], names, metadata, plan)
        if output != saved["output_feature_names"]:
            raise ValueError("Candidate feature order differs.")
        np.testing.assert_array_equal(model.predict_proba(cv)[:, 1], ca["probabilities"][valid])
        detail = nested["folds"][fold - 1]
        info = detail["inner_oof_predictions_artifact"]
        if checksum(ROOT / info["path"]) != info["sha256"]:
            raise ValueError("Inner OOF artifact fingerprint differs.")
        with np.load(ROOT / info["path"]) as inner:
            for key, value in info["array_sha256"].items():
                if runner._array_sha256(inner[key]) != value:
                    raise ValueError("Inner OOF arrays changed.")
            np.testing.assert_array_equal(inner["training_row_indices"], development[train])
            np.testing.assert_array_equal(inner["outer_validation_row_indices"], development[valid])
            np.testing.assert_array_equal(inner["labels"], labels[development[train]])
            ids = np.zeros(len(train), dtype=np.int8)
            for number, (_, iv) in enumerate(_stratified_folds(labels[development[train]], 2, suite["nested_threshold"]["inner_seed"] + fold), 1):
                ids[iv] = number
            np.testing.assert_array_equal(inner["fold_ids"], ids)
            threshold = best_f1_threshold(inner["labels"], inner["probabilities"]).threshold
        if threshold != detail["inner_threshold"]:
            raise ValueError("Inner selected threshold differs.")
        np.testing.assert_array_equal(ca["nested_thresholds"][valid], np.full(len(valid), threshold))
        fm, fc = runner._metrics(ca["labels"][valid], ca["probabilities"][valid], threshold)
        if fm != detail["outer_evaluation_metrics"] or fc != detail["outer_evaluation_confusion_counts"]:
            raise ValueError("Outer fold metrics differ.")
        checks.append({"fold": fold, "maximum_probability_difference": 0.0, "verified_inner_threshold": threshold})
        print(f"VERIFIED weight={TRIALS[token]} fold={fold}: checkpoint, scores and inner threshold", flush=True)
        del model, ct, cv
        gc.collect()
    deltas = {key: value - bn["pooled_metrics"][key] for key, value in metrics.items()}
    fold_deltas = [f["outer_evaluation_metrics"]["f1"] - b["outer_evaluation_metrics"]["f1"] for f, b in zip(nested["folds"], bn["folds"])]
    summary = {"positive_class_weight": TRIALS[token], "candidate_record": str(records[0].relative_to(ROOT)).replace("\\", "/"),
               "baseline_record": BASELINE, "candidate_nested": nested, "metric_deltas": deltas,
               "fold_f1_deltas": fold_deltas, "positive_fold_count": sum(d > 0 for d in fold_deltas),
               "meets_plan_promotion_rule": deltas["f1"] > 0 and sum(d > 0 for d in fold_deltas) >= 2,
               "paired_conditional_f1_delta_interval_95": _paired_interval(ca["labels"], ba["nested_predictions"], ca["nested_predictions"]),
               "bootstrap_scope": "20,000 paired stratified draws; fixed models/thresholds; excludes retraining and repeated selection.",
               "checkpoint_and_inner_oof_checks": checks, "final_model_changed": False}
    write(directory / "comparison.json", summary)
    print(f"RESULT weight={TRIALS[token]} F1={metrics['f1']:.9f} delta={deltas['f1']:+.9f}", flush=True)
    return summary


def report():
    baseline, weight2 = read(ROOT / BASELINE), read(ROOT / WEIGHT2)
    rows = [(1.0, baseline["outcome"]["nested_threshold_evaluation"], None)]
    for token in TRIALS:
        summary = read(ROOT / configuration(token)["artifact_dir"] / "comparison.json")
        rows.append((TRIALS[token], summary["candidate_nested"], summary))
    rows.append((2.0, weight2["outcome"]["nested_threshold_evaluation"], None))
    lines = ["# Phase 35 — Pesi dei positivi 1,25 e 1,5", "",
             "Confronto controllato con Phase 14 e peso 2 già valutato. I due nuovi candidati usano il codice numerico storico di Phase 30, isolato dalla successiva correzione della ricerca degli split sui mancanti. Cambia soltanto il peso della loss positiva; peso negativo 1. Stesse 262.508 righe development, 295 feature, seed e parametri; tre fold esterni e due interni per ogni soglia. Nove fit per candidato.", "",
             "| Peso | F1 | Precision | Recall | AP | Falsi positivi | Delta F1 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    base_f1 = rows[0][1]["pooled_metrics"]["f1"]
    for weight, nested, summary in rows:
        m, c = nested["pooled_metrics"], nested["pooled_confusion_counts"]
        lines.append(f"| {weight:g} | {m['f1']:.9f} | {m['precision']:.9f} | {m['recall']:.9f} | {m['average_precision']:.9f} | {c['false_positives']} | {m['f1']-base_f1:+.9f} |")
    for weight, nested, summary in rows:
        if summary is None:
            continue
        lines += ["", f"## Peso {weight:g}", "", "| Fold | F1 | Delta F1 | Soglia interna |", "| --- | ---: | ---: | ---: |"]
        for i, f in enumerate(nested["folds"]):
            lines.append(f"| {i+1} | {f['outer_evaluation_metrics']['f1']:.9f} | {summary['fold_f1_deltas'][i]:+.9f} | {f['inner_threshold']:.9f} |")
        ci = summary["paired_conditional_f1_delta_interval_95"]
        lines += ["", f"Fold favorevoli: {summary['positive_fold_count']}/3. Criterio operativo (F1 superiore e almeno due fold favorevoli): {'soddisfatto' if summary['meets_plan_promotion_rule'] else 'non soddisfatto'}. Intervallo bootstrap condizionale del delta: [{ci[0]:+.9f}; {ci[1]:+.9f}].", "", f"Record: `{summary['candidate_record']}`."]
    lines += ["", "Verifica: replay completo dei tre checkpoint storici prima delle prove; nove partizioni e bin confrontati con Phase 30; tutti i checkpoint esterni, OOF interni, soglie e metriche dei candidati riprodotti. Sorgenti eseguiti conservati in `results/boosting_artifacts/phase35_positive_weights/source_snapshot/`; checkpoint intermedi ogni 25 alberi permettono la ripresa.", "",
              "I fold development sono già stati usati per più scelte; il bootstrap considera fit e soglie fissi. Queste prove riguardano la versione storica, e non stabiliscono il miglior peso per la successiva correzione degli split. Configurazione finale e submission non modificate.", "",
              "Riproduzione: `python -m tools.phase35_positive_weights prepare`, poi `python -m tools.phase35_positive_weights run`. Richiede i dati e lo snapshot storico Phase 30 conservati localmente.", ""]
    path = ROOT / "results/eda/phase35_positive_weights_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"REPORT {path}", flush=True)


def orchestrate():
    manifest = prepare()
    frozen(manifest, inputs=True)
    jobs, streams = {}, []
    try:
        for token in TRIALS:
            path = ROOT / configuration(token)["artifact_dir"]
            path.mkdir(parents=True, exist_ok=True)
            out = (path / "run_stdout.log").open("a", encoding="utf-8")
            streams.append(out)
            jobs[token] = subprocess.Popen([sys.executable, "-u", "-m", "tools.phase35_positive_weights", "candidate", token], cwd=ROOT, stdout=out, stderr=out, creationflags=subprocess.CREATE_NO_WINDOW)
            print(f"STARTED weight={TRIALS[token]} pid={jobs[token].pid}", flush=True)
        remaining = set(jobs)
        while remaining:
            for token in list(remaining):
                code = jobs[token].poll()
                if code is not None:
                    print(f"FINISHED weight={TRIALS[token]} exit={code}", flush=True)
                    remaining.remove(token)
            write(SHARED / "status.json", {"updated_at_utc": datetime.now(timezone.utc).isoformat(), "trials": {t: {"weight": TRIALS[t], "pid": p.pid, "exit_code": p.poll(), "status": "running" if p.poll() is None else "completed" if p.poll() == 0 else "failed"} for t, p in jobs.items()}})
            if remaining:
                time.sleep(5)
        if any(p.returncode != 0 for p in jobs.values()):
            raise RuntimeError("A candidate failed; inspect its log. Completed checkpoints remain reusable.")
        report()
    finally:
        for stream in streams:
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "candidate", "compare", "report"))
    parser.add_argument("token", nargs="?", choices=tuple(TRIALS))
    args = parser.parse_args()
    if args.action in ("candidate", "compare") and args.token is None:
        parser.error("A candidate token is required.")
    if args.action == "prepare":
        prepare()
    elif args.action == "run":
        orchestrate()
    elif args.action == "candidate":
        run(args.token)
    elif args.action == "compare":
        compare(args.token)
    else:
        report()


if __name__ == "__main__":
    main()

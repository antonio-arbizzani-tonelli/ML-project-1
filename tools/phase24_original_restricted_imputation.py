"""Confirm original restricted imputation using verified outer fits and six inner fits."""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from src import run_boosting_cv as runner
from src.evaluation import best_f1_threshold
from src.experiment_logger import save_experiment
from src.feature_preprocessing import tree_feature_matrices
from src.numpy_boosting import HistogramBinner, HistogramGradientBoostingClassifier
from src.run_logged_boosting_cv import _logged_fit
from src.run_preprocessing_cv import _sha256, _stratified_folds
from src.summarize_phase19_transfer import BASELINE, _arrays, _paired_interval, _publish
from tools.phase23_missing_reason_flags import SOURCES
from tools.run_targeted_preprocessing_trials import _Tee
from tools.targeted_preprocessing_analysis import _input_data


PREFIX = "phase24_original_restricted_imputation"
SUITE = f"configs/experiments/{PREFIX}.json"
PREFLIGHT = "results/eda/phase24_original_restricted_imputation_preflight.json"
COMPARISON = "results/experiments/phase24_original_restricted_imputation_comparison.json"
REPORT = "results/eda/phase24_original_restricted_imputation_summary.md"
OWN_SOURCES = (*SOURCES, "tools/phase24_original_restricted_imputation.py")


def read(root, path):
    return json.loads((root / path).read_text(encoding="utf-8"))


def configuration(root):
    suite = read(root, SUITE)
    runner._validate_suite(suite)
    original = read(root, "configs/experiments/phase16_imputation_restricted_boosting.json")
    for key in original:
        if key not in {"experiment_prefix", "artifact_dir"} and suite[key] != original[key]:
            raise ValueError(f"Original pipeline changed: {key}")
    variant = runner._variant(read(root, suite["preprocessing_config_path"]), suite["preprocessing_variant"])
    return suite, variant


def verify_rows(arrays, development, labels, outer):
    """Reject misaligned, incomplete or differently partitioned reusable predictions."""
    np.testing.assert_array_equal(arrays["development_indices"], development)
    np.testing.assert_array_equal(arrays["labels"], labels[development])
    ids = np.zeros(development.size, dtype=np.int8)
    for fold, (_, valid) in enumerate(outer, 1):
        ids[valid] = fold
    np.testing.assert_array_equal(arrays["fold_ids"], ids)
    probabilities = arrays["probabilities"]
    if (probabilities.shape != ids.shape or not np.all(np.isfinite(probabilities))
            or np.any((probabilities < 0) | (probabilities > 1))):
        raise ValueError("Reusable scores must be finite probabilities aligned with development rows.")


def collapsed_binary_flags(names, binner):
    """Count distinguishable binary states rather than allocated histogram bins."""
    return [names[i] for i in range(295, len(names))
            if np.unique(np.searchsorted(binner.edges_[i], [0.0, 1.0], side="right")).size == 1]


def preflight(root):
    suite, variant = configuration(root)
    old = read(root, suite["reuse_record_path"])
    baseline = read(root, BASELINE)
    original_suite = "configs/experiments/phase16_imputation_restricted_boosting_optimistic.json"
    expected_old = old["outcome"]["input_sha256"]
    if _sha256(root / original_suite) != expected_old["boosting_suite_config"]:
        raise ValueError("Original outer configuration changed.")
    parameters = {**suite["models"][0]["parameters"], "n_estimators": 200}
    if parameters != old["config"]["model"]["parameters"] or parameters != baseline["config"]["model"]["parameters"]:
        raise ValueError("Model parameters differ from original or baseline.")
    if old["config"]["split"] != baseline["config"]["split"]:
        raise ValueError("Original and baseline split protocols differ.")
    fingerprints = {key: _sha256(root / suite["data"][field]) for key, field in (
        ("features", "features_path"), ("labels", "labels_path"), ("split_indices", "split_indices_path"))}
    fingerprints["feature_metadata"] = _sha256(root / suite["feature_metadata_path"])
    for key, value in fingerprints.items():
        if value != expected_old[key] or value != baseline["outcome"]["input_sha256"][key]:
            raise ValueError(f"Historical inputs changed: {key}")
    fingerprints["preprocessing_config"] = _sha256(root / suite["preprocessing_config_path"])
    if fingerprints["preprocessing_config"] != expected_old["preprocessing_config"]:
        raise ValueError("Original preprocessing changed.")
    fingerprints["boosting_suite_config"] = _sha256(root / SUITE)
    x, y, development, names, metadata = _input_data(root, suite)
    outer = list(_stratified_folds(y[development], 3, suite["seed"]))
    arrays = _arrays(root, old)
    verify_rows(arrays, development, y, outer)
    verify_rows(_arrays(root, baseline), development, y, outer)
    report = {"scope": "Verify original outer models; no new disease model training.",
        "input_sha256": fingerprints, "original_record_sha256": _sha256(root / suite["reuse_record_path"]),
        "original_suite_sha256": expected_old["boosting_suite_config"],
        "source_sha256": {p: _sha256(root / p) for p in OWN_SOURCES},
        "final_config_sha256": _sha256(root / "configs/final_model.json"), "outer_checks": []}
    base_plan = runner._variant(read(root, baseline["config"]["features"]["preprocessing_source"]),
                               baseline["config"]["features"]["preprocessing_variant"])["preprocessing"]
    expected_plan = {**base_plan, "imputation": variant["preprocessing"]["imputation"]}
    if variant["preprocessing"] != expected_plan or not expected_plan["imputation"]["fill_values"]:
        raise ValueError("Expected Phase 14 plus the original restricted imputation.")
    for fold, (train, valid) in enumerate(outer, 1):
        checkpoint = old["outcome"]["model_checkpoints"][fold - 1]
        if checkpoint["fold"] != fold or _sha256(root / checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError("Original checkpoint changed or is assigned to the wrong fold.")
        model, saved = HistogramGradientBoostingClassifier.load_checkpoint(root / checkpoint["path"])
        np.testing.assert_array_equal(saved["training_row_indices"], development[train])
        for key, actual in (("training_row_indices_sha256", runner._array_sha256(development[train])),
                            ("training_labels_sha256", runner._array_sha256(y[development[train]]))):
            if saved[key] != actual:
                raise ValueError(f"Historical training boundary changed: {key}")
        if saved["input_sha256"] != expected_old or saved["outer_fold"] != fold or saved["fitted_tree_count"] != 200:
            raise ValueError("Historical checkpoint provenance differs from the record.")
        if any(getattr(model, key) != value for key, value in parameters.items()):
            raise ValueError("Actual checkpoint parameters differ.")
        if model.binning_strategy != "quantile" or len(model.trees_) != 200:
            raise ValueError("Expected original quantile model with 200 trees.")
        ct, cv, cn = tree_feature_matrices(x[development[train]], x[development[valid]], names, metadata,
                                         variant["preprocessing"])
        if len(cn) != 303 or model.n_features_in_ != 303:
            raise ValueError("Original imputation must produce 303 inputs.")
        binner = HistogramBinner(64).fit(ct)
        np.testing.assert_array_equal(binner.bin_counts_, model.binner_.bin_counts_)
        for now, previous in zip(binner.edges_, model.binner_.edges_):
            np.testing.assert_array_equal(now, previous)
        probability_difference = float(np.max(np.abs(model.predict_proba(cv)[:, 1] - arrays["probabilities"][valid])))
        scores = next(model.staged_decision_function(ct, [200]))[1]
        score_difference = float(np.max(np.abs(scores - model.training_scores_)))
        if probability_difference != 0 or score_difference != 0:
            raise ValueError("Original model predictions are not reproduced exactly.")
        report["outer_checks"].append({"fold": fold, "training_rows": int(train.size),
            "validation_rows": int(valid.size), "feature_names": cn,
            "training_row_indices_sha256": saved["training_row_indices_sha256"],
            "training_labels_sha256": saved["training_labels_sha256"],
            "maximum_validation_probability_difference": probability_difference,
            "maximum_training_score_difference": score_difference,
            "collapsed_flags": collapsed_binary_flags(cn, binner)})
        print(f"PREFLIGHT fold={fold}: original bins, training scores and outer predictions identical", flush=True)
        del model, ct, cv, binner, scores
        gc.collect()
    (root / PREFLIGHT).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def verify_frozen(root, check, suite):
    for path, expected in check["source_sha256"].items():
        if _sha256(root / path) != expected:
            raise ValueError(f"Source changed after verification: {path}")
    if _sha256(root / suite["reuse_record_path"]) != check["original_record_sha256"]:
        raise ValueError("Original record changed.")
    if _sha256(root / "configs/final_model.json") != check["final_config_sha256"]:
        raise ValueError("Final model configuration changed.")
    current = {key: _sha256(root / suite["data"][field]) for key, field in (
        ("features", "features_path"), ("labels", "labels_path"), ("split_indices", "split_indices_path"))}
    current.update(feature_metadata=_sha256(root / suite["feature_metadata_path"]),
                   preprocessing_config=_sha256(root / suite["preprocessing_config_path"]),
                   boosting_suite_config=_sha256(root / SUITE))
    if current != check["input_sha256"]:
        raise ValueError("Configuration or training inputs changed after verification.")


def run(root):
    suite, variant = configuration(root)
    check = read(root, PREFLIGHT)
    verify_frozen(root, check, suite)
    directory = root / suite["artifact_dir"]
    if list((directory / "records").glob("*.json")) or list((directory / "inner_oof_predictions").glob("*.npz")):
        raise ValueError("Phase 24 artifacts already exist; inspect them before retraining.")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run_stdout.log").open("w", encoding="utf-8") as log:
        tee = _Tee(sys.stdout, log)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            return run_inner(root, suite, variant, check, directory)


def run_inner(root, suite, variant, check, directory):
    old = read(root, suite["reuse_record_path"])
    arrays = _arrays(root, old)
    x, y, development, names, metadata = _input_data(root, suite)
    outer = list(_stratified_folds(y[development], 3, suite["seed"]))
    verify_rows(arrays, development, y, outer)
    name = suite["models"][0]["name"]
    key = (name, 200)
    profiles = []
    original_fit = runner._fit_checkpoint_probabilities
    runner._fit_checkpoint_probabilities = _logged_fit(original_fit, profiles, 0)
    started = time.perf_counter()
    try:
        results, thresholds = runner._nested_evaluations(
            {name: [200]}, {name: suite["models"][0]}, variant, x, development, y[development], names,
            metadata, [valid for _, valid in outer], {key: arrays["probabilities"]}, 2,
            suite["nested_threshold"]["inner_seed"], directory / "inner_oof_predictions", PREFIX, root)
    finally:
        runner._fit_checkpoint_probabilities = original_fit
    elapsed = time.perf_counter() - started
    verify_frozen(root, check, suite)
    if len(profiles) != 6 or any(p["stage"] != "inner" or p["binning_strategy"] != "quantile" for p in profiles):
        raise ValueError("Expected six original-quantile inner fits.")
    config = copy.deepcopy(old["config"])
    config["experiment_name"] = f"{PREFIX}_{name}_200trees"
    config["hypothesis"] = suite["hypothesis"]
    config["threshold"]["selection_method"] = "Exploratory pooled threshold retained for provenance; comparison uses inner-only thresholds in nested_threshold_evaluation."
    config["missing_values"] = {"method": "Original fold-fitted restricted imputation plus eight original indicators and quantile binning; remaining NaNs routed by trees.",
                                "fit_partition": "corresponding training rows only; imputer receives no disease labels"}
    outcome = copy.deepcopy(old["outcome"])
    path, fingerprint = runner._save_oof_predictions(directory / "oof_predictions", config["experiment_name"],
        development, arrays["fold_ids"], y[development], arrays["probabilities"],
        old["config"]["threshold"]["value"], thresholds[key])
    outcome.update(nested_threshold_evaluation=results[key], input_sha256=check["input_sha256"],
        source_sha256=check["source_sha256"], environment={"python": platform.python_version(), "numpy": np.__version__},
        individual_fit_profiles=profiles, runtime_seconds=elapsed,
        runtime_basis="Six new inner fits including preprocessing; three historical outer fits reused, not included in this duration.",
        historical_outer_runtime_seconds=old["outcome"]["runtime_seconds"],
        original_outer_reuse={"record": suite["reuse_record_path"], "record_sha256": check["original_record_sha256"],
            "input_sha256": old["outcome"]["input_sha256"], "preflight": PREFLIGHT,
            "preflight_sha256": _sha256(root / PREFLIGHT), "outer_checks": check["outer_checks"]},
        oof_predictions_artifact={"path": str(path.relative_to(root)), "sha256": fingerprint,
            "arrays": ["development_indices", "fold_ids", "labels", "probabilities", "exploratory_threshold", "nested_thresholds", "nested_predictions"]})
    record = save_experiment(config, outcome, directory / "records")
    print(f"COMPLETE six inner fits; record={record.name}", flush=True)
    return compare(root, record)


def compare(root, source):
    suite, _ = configuration(root)
    check = read(root, PREFLIGHT)
    verify_frozen(root, check, suite)
    record, baseline, old = read(root, source), read(root, BASELINE), read(root, suite["reuse_record_path"])
    ca, ba, oa = _arrays(root, record), _arrays(root, baseline), _arrays(root, old)
    x, y, development, _, _ = _input_data(root, suite)
    outer = list(_stratified_folds(y[development], 3, suite["seed"]))
    for arrays in (ca, ba, oa):
        verify_rows(arrays, development, y, outer)
    np.testing.assert_array_equal(ca["probabilities"], oa["probabilities"])
    np.testing.assert_array_equal(ca["nested_predictions"], ca["probabilities"] >= ca["nested_thresholds"])
    nested = record["outcome"]["nested_threshold_evaluation"]
    checks = []
    for fold, (train, valid) in enumerate(outer, 1):
        detail = nested["folds"][fold - 1]
        info = detail["inner_oof_predictions_artifact"]
        if _sha256(root / info["path"]) != info["sha256"]:
            raise ValueError("Inner predictions fingerprint differs.")
        with np.load(root / info["path"]) as inner:
            for key, expected in info["array_sha256"].items():
                if runner._array_sha256(inner[key]) != expected:
                    raise ValueError(f"Inner array changed: {key}")
            np.testing.assert_array_equal(inner["training_row_indices"], development[train])
            np.testing.assert_array_equal(inner["outer_validation_row_indices"], development[valid])
            np.testing.assert_array_equal(inner["labels"], y[development[train]])
            if np.intersect1d(inner["training_row_indices"], inner["outer_validation_row_indices"]).size:
                raise ValueError("Outer validation rows entered threshold selection.")
            ids = np.zeros(train.size, dtype=np.int8)
            for number, (_, iv) in enumerate(_stratified_folds(y[development[train]], 2, 20260922 + fold), 1):
                ids[iv] = number
            np.testing.assert_array_equal(inner["fold_ids"], ids)
            threshold = best_f1_threshold(inner["labels"], inner["probabilities"]).threshold
        if threshold != detail["inner_threshold"]:
            raise ValueError("Inner threshold cannot be reproduced.")
        np.testing.assert_array_equal(ca["nested_thresholds"][valid], np.full(valid.size, threshold))
        fm, fc = runner._metrics(ca["labels"][valid], ca["probabilities"][valid], threshold)
        if fm != detail["outer_evaluation_metrics"] or fc != detail["outer_evaluation_confusion_counts"]:
            raise ValueError("Fold metrics cannot be reproduced.")
        checks.append({"fold": fold, "threshold_reproduced": threshold, "outer_rows_excluded": True})
        print(f"VERIFIED fold={fold}: inner rows, threshold and external metrics identical", flush=True)
    from src.evaluation import classification_metrics, average_precision, binary_log_loss
    counts, metrics = classification_metrics(ca["labels"], ca["nested_predictions"])
    metrics.update(average_precision=average_precision(ca["labels"], ca["probabilities"]),
                   log_loss=binary_log_loss(ca["labels"], ca["probabilities"]))
    if metrics != nested["pooled_metrics"] or vars(counts) != nested["pooled_confusion_counts"]:
        raise ValueError("Pooled metrics cannot be reproduced.")
    bn = baseline["outcome"]["nested_threshold_evaluation"]
    deltas = {k: v - bn["pooled_metrics"][k] for k, v in metrics.items()}
    fold_deltas = [a["outer_evaluation_metrics"]["f1"] - b["outer_evaluation_metrics"]["f1"]
                   for a, b in zip(nested["folds"], bn["folds"])]
    published = _publish(source, root)
    summary = {"phase": 24, "baseline_record": BASELINE, "original_record": suite["reuse_record_path"],
        "candidate_record": str(published.relative_to(root)).replace("\\", "/"),
        "candidate_nested": nested, "baseline_nested": bn, "metric_deltas": deltas, "fold_f1_deltas": fold_deltas,
        "positive_fold_count": sum(d > 0 for d in fold_deltas),
        "meets_plan_promotion_rule": deltas["f1"] > 0 and sum(d > 0 for d in fold_deltas) >= 2,
        "paired_conditional_f1_delta_interval_95": _paired_interval(ca["labels"], ba["nested_predictions"], ca["nested_predictions"]),
        "bootstrap_repetitions": 20000, "bootstrap_seed": 20260929,
        "bootstrap_scope": "Fixed models and thresholds; excludes retraining and repeated candidate selection uncertainty.",
        "runtime_seconds": record["outcome"]["runtime_seconds"], "new_inner_fit_count": 6, "reused_outer_fit_count": 3,
        "environment": record["outcome"]["environment"], "verification": checks, "final_model_changed": False,
        "original_exploratory_f1": old["outcome"]["metrics"]["f1"]}
    (root / COMPARISON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(root, summary)
    print(json.dumps({"phase": 24, "f1": metrics["f1"], "delta_f1": deltas["f1"], "fold_deltas": fold_deltas,
                      "meets_plan_rule": summary["meets_plan_promotion_rule"]}), flush=True)
    return summary


def write_report(root, summary):
    c, b = summary["candidate_nested"], summary["baseline_nested"]
    lines = ["# Phase 24 — conferma annidata dell'imputazione ristretta originale", "",
        "## Configurazione e protocollo", "",
        "Replica della pipeline Phase 16 originale: 303 input, otto indicatori con binning a quantili, imputazione ristretta di INCOME2, PNEUVAC3, EMPLOY1, altezza/peso e BMI derivabile. I due indicatori rari restano accorpati dal binning originale. Non è la variante con binning esatto della Phase 23.", "",
        "Stessi dati development e tre fold esterni di Phase 14. Booster: 200 alberi, profondità 5, learning rate 0,05, minimo foglia 200, 64 bin, 128 feature candidate, L2=1, seed 20260920. Soglie da due fold interni con seed 20260922 + fold esterno. Imputatori rifatti esclusivamente sul training della rispettiva partizione, senza target di malattia.", "",
        "## Verifiche e riutilizzo", "",
        "Riutilizzati i tre modelli esterni originali dopo confronto di hash, righe e label di training, parametri e bin. Le predizioni esterne e gli score di training ricalcolati coincidono esattamente con quelli storici. Addestrati sei nuovi modelli interni; salvate le loro predizioni con righe, fold e hash. Soglie e risultati ricostruiti dagli artefatti.", "",
        f"Tempo della nuova CV interna, inclusi imputatori: {summary['runtime_seconds'] / 60:.2f} minuti. Il tempo non comprende i tre training esterni storici né il preflight. Ambiente: Python {summary['environment']['python']}, NumPy {summary['environment']['numpy']}.", "",
        "## Risultati", "", "| Metrica | Phase 14 | Imputazione originale | Differenza |",
        "| --- | ---: | ---: | ---: |"]
    for key in ("f1", "precision", "recall", "average_precision", "log_loss"):
        lines.append(f"| {key} | {b['pooled_metrics'][key]:.9f} | {c['pooled_metrics'][key]:.9f} | {summary['metric_deltas'][key]:+.9f} |")
    lines += ["", "| Fold | F1 Phase 14 | F1 imputazione | Differenza | Soglia interna |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for index, (cf, bf) in enumerate(zip(c["folds"], b["folds"]), 1):
        lines.append(f"| {index} | {bf['outer_evaluation_metrics']['f1']:.9f} | {cf['outer_evaluation_metrics']['f1']:.9f} | {summary['fold_f1_deltas'][index-1]:+.9f} | {cf['inner_threshold']:.9f} |")
    lines += ["", f"Matrice di confusione aggregata: {json.dumps(c['pooled_confusion_counts'])}.", "",
              "## Interpretazione e decisione", "",
              f"Il precedente F1 esplorativo {summary['original_exploratory_f1']:.9f} sceglieva e valutava la soglia sulle stesse label OOF. L'F1 annidato qui riportato separa questa scelta dalla valutazione esterna; il riutilizzo di predizioni già consultate e la selezione ripetuta dei candidati mantengono un limite di selezione del modello.", "",
              f"Fold favorevoli: {summary['positive_fold_count']}/3. Regola operativa del piano soddisfatta: {summary['meets_plan_promotion_rule']}. Configurazione finale lasciata invariata in questa prova.", "",
              f"Intervallo descrittivo bootstrap appaiato 95% del delta F1: {summary['paired_conditional_f1_delta_interval_95']}. Modelli e soglie fissi; non comprende incertezza da riaddestramento o selezione dei candidati.", "",
              f"Record: [{Path(summary['candidate_record']).name}](../experiments/{Path(summary['candidate_record']).name}).",
              f"Confronto completo: [JSON](../experiments/{Path(COMPARISON).name}).", ""]
    (root / REPORT).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run", "compare"))
    parser.add_argument("record", nargs="?", type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    if args.action == "preflight":
        preflight(root)
    elif args.action == "run":
        run(root)
    elif args.record is not None:
        compare(root, args.record)
    else:
        parser.error("compare requires a record path")


if __name__ == "__main__":
    main()

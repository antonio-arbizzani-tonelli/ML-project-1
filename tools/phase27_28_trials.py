"""Run independent Phase 14 learning-rate and ALCDAY5 trials safely in parallel."""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import json
import sys
from itertools import zip_longest
from pathlib import Path

import numpy as np

from src.evaluation import average_precision, best_f1_threshold, binary_log_loss, classification_metrics
from src.feature_preprocessing import tree_feature_matrices
from src.numpy_boosting import HistogramBinner, HistogramGradientBoostingClassifier
from src.run_boosting_cv import _array_sha256, _metrics, _validate_suite, _variant
from src.run_logged_boosting_cv import run_logged_suite
from src.run_preprocessing_cv import _sha256, _stratified_folds
from src.summarize_phase19_transfer import BASELINE, _arrays, _paired_interval
from tools.phase23_missing_reason_flags import SOURCES
from tools.run_targeted_preprocessing_trials import _Tee
from tools.targeted_preprocessing_analysis import _input_data, _partitions


REFERENCE = "configs/experiments/phase14_boosting_codebook_corrections_only.json"
PREFIXES = {27: "phase27_phase14_lr0025_400trees", 28: "phase28_phase14_ablate_alcday5"}


def _zip_equal(first, second):
    """Pair iterables lazily and reject unequal lengths on Python 3.9."""
    missing = object()
    for left, right in zip_longest(first, second, fillvalue=missing):
        if left is missing or right is missing:
            raise ValueError("Paired iterables must have equal lengths.")
        yield left, right


def read(root, path):
    return json.loads((root / path).read_text(encoding="utf-8"))


def paths(phase):
    prefix = PREFIXES[phase]
    return f"configs/experiments/{prefix}.json", f"results/eda/{prefix}_preflight.json"


def validate_change(candidate, reference, phase):
    """Reject changes beyond the requested independent intervention."""
    _validate_suite(candidate)
    expected = copy.deepcopy(reference)
    if phase == 27:
        expected["models"][0]["parameters"]["learning_rate"] = 0.025
        expected["models"][0]["checkpoints"] = [400]
        expected["nested_threshold"]["model_checkpoints"][0]["checkpoints"] = [400]
    elif phase == 28:
        expected["preprocessing_config_path"] = "configs/experiments/phase28_ablate_alcday5_preprocessing.json"
        expected["preprocessing_variant"] = "phase28_ablate_alcday5"
    else:
        raise ValueError("Only phases 27 and 28 are supported.")
    ignored = {"experiment_prefix", "artifact_dir", "hypothesis"}
    if {k: v for k, v in candidate.items() if k not in ignored} != {k: v for k, v in expected.items() if k not in ignored}:
        raise ValueError("Candidate includes an unintended change from Phase 14.")
    if candidate["experiment_prefix"] != PREFIXES[phase] or candidate["artifact_dir"] != f"results/boosting_artifacts/{PREFIXES[phase]}":
        raise ValueError("Each trial must have its own artifact directory and name.")


def configuration(root, phase):
    suite = read(root, paths(phase)[0])
    reference = read(root, REFERENCE)
    validate_change(suite, reference, phase)
    plan = _variant(read(root, suite["preprocessing_config_path"]), suite["preprocessing_variant"])["preprocessing"]
    base_plan = _variant(read(root, reference["preprocessing_config_path"]), reference["preprocessing_variant"])["preprocessing"]
    expected = copy.deepcopy(base_plan)
    if phase == 28:
        expected["exclude_features"].append("ALCDAY5")
    if plan != expected:
        raise ValueError("Unexpected preprocessing change.")
    return suite, plan, base_plan


def frozen_hashes(root, phase, suite):
    names = [*SOURCES, "tools/phase27_28_trials.py", REFERENCE, BASELINE, "configs/final_model.json",
             paths(phase)[0], suite["feature_metadata_path"], suite["preprocessing_config_path"],
             *[suite["data"][k] for k in ("features_path", "labels_path", "split_indices_path", "x_train_csv_path")]]
    return {name: _sha256(root / name) for name in names}


def preflight(root, phase):
    suite, plan, base_plan = configuration(root, phase)
    directory = root / suite["artifact_dir"]
    if (directory / "executed_runner.py").exists():
        raise ValueError("Preserve the original preflight after training starts.")
    baseline = read(root, BASELINE)
    cached = read(root, "results/eda/phase26_two_seeds_preflight.json")
    hashes = frozen_hashes(root, phase, suite)
    for name, expected in cached["source_sha256"].items():
        if hashes[name] != expected:
            raise ValueError(f"Verified baseline source changed: {name}")
    for name, key in ((BASELINE, "baseline_record_sha256"), (REFERENCE, "reference_suite_sha256"),
                      ("configs/final_model.json", "final_config_sha256")):
        if hashes[name] != cached[key]:
            raise ValueError(f"Verified baseline reference changed: {name}")
    for key, name in (("features", suite["data"]["features_path"]), ("labels", suite["data"]["labels_path"]),
                      ("split_indices", suite["data"]["split_indices_path"]), ("feature_metadata", suite["feature_metadata_path"])):
        if hashes[name] != baseline["outcome"]["input_sha256"][key]:
            raise ValueError(f"Historical input changed: {key}")
    x, y, development, names, metadata = _input_data(root, suite)
    ba = _arrays(root, baseline)
    np.testing.assert_array_equal(ba["development_indices"], development)
    np.testing.assert_array_equal(ba["labels"], y[development])
    result = {"phase": phase, "frozen_sha256": hashes, "partitions": [],
              "baseline_verification_reused_from": "results/eda/phase26_two_seeds_preflight.json"}
    for (partition, train, valid), previous in _zip_equal(_partitions(y, development, suite), cached["partitions"]):
        bt, bv, bn = tree_feature_matrices(x[development[train]], x[development[valid]], names, metadata, base_plan)
        if phase == 27:
            ct, cv, cn = bt, bv, bn
        else:
            ct, cv, cn = tree_feature_matrices(x[development[train]], x[development[valid]], names, metadata, plan)
        expected_names = [name for name in bn if phase == 27 or name != "ALCDAY5"]
        if cn != expected_names or len(cn) != (295 if phase == 27 else 294) or "DROCDY3_" not in cn:
            raise ValueError("Incorrect candidate feature set or order.")
        for ci, name in enumerate(cn):
            bi = bn.index(name)
            np.testing.assert_array_equal(ct[:, ci], bt[:, bi])
            np.testing.assert_array_equal(cv[:, ci], bv[:, bi])
        binner = HistogramBinner(64).fit(ct)
        item = {"partition": partition, "feature_count": len(cn), "output_names": cn,
                "training_rows": int(train.size), "validation_rows": int(valid.size),
                "training_row_indices_sha256": _array_sha256(development[train]),
                "training_labels_sha256": _array_sha256(y[development[train]]),
                "bin_counts": binner.bin_counts_.tolist(), "exact_column_indices": []}
        for key in ("partition", "training_rows", "validation_rows", "training_row_indices_sha256", "training_labels_sha256"):
            if item[key] != previous[key]:
                raise ValueError(f"Baseline partition differs: {partition}/{key}")
        if bn != previous["output_names"] or item["bin_counts"] != [previous["bin_counts"][bn.index(n)] for n in cn]:
            raise ValueError("Quantile binning differs from the verified baseline.")
        if "inner" not in partition:
            np.testing.assert_array_equal(ba["fold_ids"][valid], np.full(valid.size, int(partition[-1])))
        result["partitions"].append(item)
        print(f"PREFLIGHT phase={phase} {partition}: {len(cn)} inputs, rows and bins verified", flush=True)
        del bt, bv, ct, cv, binner
        gc.collect()
    (root / paths(phase)[1]).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def verify_frozen(root, phase, suite, check):
    if frozen_hashes(root, phase, suite) != check["frozen_sha256"]:
        raise ValueError("Inputs, code or configuration changed since preflight.")


def compare(root, phase, source):
    suite, plan, _ = configuration(root, phase)
    check = read(root, paths(phase)[1])
    verify_frozen(root, phase, suite, check)
    candidate, baseline = read(root, source), read(root, BASELINE)
    outcome = candidate["outcome"]
    expected_parameters = {**suite["models"][0]["parameters"], "n_estimators": suite["models"][0]["checkpoints"][-1]}
    expected_features = {"preprocessing_source": suite["preprocessing_config_path"],
                         "preprocessing_variant": suite["preprocessing_variant"],
                         "output_feature_count": 295 if phase == 27 else 294}
    if candidate["config"]["model"]["parameters"] != expected_parameters or candidate["config"]["features"] != expected_features or candidate["config"]["split"] != baseline["config"]["split"]:
        raise ValueError("Completed record changed the intended experiment.")
    for name, value in outcome["source_sha256"].items():
        if value != check["frozen_sha256"][name]:
            raise ValueError("Executed training code differs from preflight.")
    for key, name in (("features", suite["data"]["features_path"]), ("labels", suite["data"]["labels_path"]),
                      ("split_indices", suite["data"]["split_indices_path"]), ("feature_metadata", suite["feature_metadata_path"]),
                      ("preprocessing_config", suite["preprocessing_config_path"]), ("boosting_suite_config", paths(phase)[0])):
        if outcome["input_sha256"][key] != check["frozen_sha256"][name]:
            raise ValueError(f"Executed input differs: {key}")
    profiles = outcome["individual_fit_profiles"]
    if len(profiles) != 9:
        raise ValueError("Expected nine completed fits.")
    for i, (actual, expected) in enumerate(_zip_equal(profiles, check["partitions"])):
        if actual["stage"] != ("outer" if i < 3 else "inner") or actual["binning_strategy"] != "quantile":
            raise ValueError("Wrong training stage or binning strategy.")
        for key in ("feature_count", "training_rows", "validation_rows", "bin_counts", "exact_column_indices"):
            if actual[key] != expected[key]:
                raise ValueError(f"Fit differs from preflight: {key}")
    ca, ba = _arrays(root, candidate), _arrays(root, baseline)
    for key in ("development_indices", "labels", "fold_ids"):
        np.testing.assert_array_equal(ca[key], ba[key])
    np.testing.assert_array_equal(ca["nested_predictions"], ca["probabilities"] >= ca["nested_thresholds"])
    nested = outcome["nested_threshold_evaluation"]
    counts, metrics = classification_metrics(ca["labels"], ca["nested_predictions"])
    metrics.update(average_precision=average_precision(ca["labels"], ca["probabilities"]),
                   log_loss=binary_log_loss(ca["labels"], ca["probabilities"]))
    if metrics != nested["pooled_metrics"] or vars(counts) != nested["pooled_confusion_counts"]:
        raise ValueError("Pooled metrics cannot be reproduced.")
    x, y, development, names, metadata = _input_data(root, suite)
    verified = []
    for fold, (train, valid) in enumerate(_stratified_folds(y[development], 3, suite["seed"]), 1):
        cp = outcome["model_checkpoints"][fold - 1]
        if cp["fold"] != fold or _sha256(root / cp["path"]) != cp["sha256"]:
            raise ValueError("Checkpoint fingerprint or fold differs.")
        model, saved = HistogramGradientBoostingClassifier.load_checkpoint(root / cp["path"])
        np.testing.assert_array_equal(saved["training_row_indices"], development[train])
        expected = check["partitions"][fold - 1]
        for key in ("training_row_indices_sha256", "training_labels_sha256"):
            if saved[key] != expected[key]:
                raise ValueError("Checkpoint training boundary differs.")
        if saved["input_sha256"] != outcome["input_sha256"]:
            raise ValueError("Checkpoint input hashes differ.")
        for key, value in expected_parameters.items():
            if getattr(model, key) != value:
                raise ValueError(f"Checkpoint parameter differs: {key}")
        ct, cv, cn = tree_feature_matrices(x[development[train]], x[development[valid]], names, metadata, plan)
        if cn != saved["output_feature_names"] or cn != expected["output_names"]:
            raise ValueError("Checkpoint feature order differs.")
        np.testing.assert_array_equal(model.predict_proba(cv)[:, 1], ca["probabilities"][valid])
        detail = nested["folds"][fold - 1]
        info = detail["inner_oof_predictions_artifact"]
        if _sha256(root / info["path"]) != info["sha256"]:
            raise ValueError("Inner OOF fingerprint differs.")
        with np.load(root / info["path"]) as inner:
            for key, value in info["array_sha256"].items():
                if _array_sha256(inner[key]) != value:
                    raise ValueError(f"Inner OOF array differs: {key}")
            np.testing.assert_array_equal(inner["training_row_indices"], development[train])
            np.testing.assert_array_equal(inner["outer_validation_row_indices"], development[valid])
            np.testing.assert_array_equal(inner["labels"], y[development[train]])
            ids = np.zeros(train.size, dtype=np.int8)
            for number, (_, iv) in enumerate(_stratified_folds(y[development[train]], 2, suite["nested_threshold"]["inner_seed"] + fold), 1):
                ids[iv] = number
            np.testing.assert_array_equal(inner["fold_ids"], ids)
            threshold = best_f1_threshold(inner["labels"], inner["probabilities"]).threshold
        if threshold != detail["inner_threshold"]:
            raise ValueError("Inner threshold differs.")
        np.testing.assert_array_equal(ca["nested_thresholds"][valid], np.full(valid.size, threshold))
        fm, fc = _metrics(ca["labels"][valid], ca["probabilities"][valid], threshold)
        if fm != detail["outer_evaluation_metrics"] or fc != detail["outer_evaluation_confusion_counts"]:
            raise ValueError("Outer fold metrics differ.")
        verified.append({"fold": fold, "maximum_probability_difference": 0.0, "reproduced_inner_threshold": threshold})
        print(f"VERIFIED phase={phase} fold={fold}: checkpoint, OOF and threshold identical", flush=True)
        del model, ct, cv
        gc.collect()
    bn = baseline["outcome"]["nested_threshold_evaluation"]
    deltas = {key: value - bn["pooled_metrics"][key] for key, value in metrics.items()}
    fold_deltas = [cf["outer_evaluation_metrics"]["f1"] - bf["outer_evaluation_metrics"]["f1"]
                   for cf, bf in _zip_equal(nested["folds"], bn["folds"])]
    summary = {"phase": phase, "baseline_record": BASELINE, "candidate_record": str(source.relative_to(root)).replace("\\", "/"),
               "baseline_nested": bn, "candidate_nested": nested, "metric_deltas": deltas, "fold_f1_deltas": fold_deltas,
               "positive_fold_count": sum(d > 0 for d in fold_deltas),
               "meets_plan_promotion_rule": deltas["f1"] > 0 and sum(d > 0 for d in fold_deltas) >= 2,
               "paired_conditional_f1_delta_interval_95": _paired_interval(ca["labels"], ba["nested_predictions"], ca["nested_predictions"]),
               "bootstrap_seed": 20260929, "bootstrap_repetitions": 20000,
               "bootstrap_scope": "Fixed fits and thresholds; excludes retraining and repeated model selection.",
               "runtime_seconds": outcome["runtime_seconds"], "environment": outcome["environment"],
               "individual_fit_profiles": profiles, "checkpoint_and_inner_oof_checks": verified, "final_model_changed": False}
    (root / suite["artifact_dir"] / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(root, phase, suite, summary)
    print(json.dumps({"phase": phase, "f1": metrics["f1"], "delta_f1": deltas["f1"], "fold_deltas": fold_deltas}), flush=True)


def write_report(root, phase, suite, summary):
    title = "Learning rate 0,025 e 400 alberi" if phase == 27 else "Rimozione della sola ALCDAY5"
    lines = [f"# Phase {phase} — {title}", "", suite["hypothesis"], "",
             "Controllo: Phase 14. Stesse righe development, tre fold esterni e due interni per scegliere ogni soglia. Seed invariati. Nove nuovi fit del candidato; baseline verificata riutilizzata.", "",
             "| Metrica | Phase 14 | Candidato | Delta |", "| --- | ---: | ---: | ---: |"]
    for key, delta in summary["metric_deltas"].items():
        lines.append(f"| {key} | {summary['baseline_nested']['pooled_metrics'][key]:.9f} | {summary['candidate_nested']['pooled_metrics'][key]:.9f} | {delta:+.9f} |")
    lines += ["", "| Fold | F1 candidato | Delta F1 | Soglia interna |", "| --- | ---: | ---: | ---: |"]
    for i, detail in enumerate(summary["candidate_nested"]["folds"]):
        lines.append(f"| {i+1} | {detail['outer_evaluation_metrics']['f1']:.9f} | {summary['fold_f1_deltas'][i]:+.9f} | {detail['inner_threshold']:.9f} |")
    lines += ["", "| Esito | Phase 14 | Candidato |", "| --- | ---: | ---: |"]
    for key, count in summary["candidate_nested"]["pooled_confusion_counts"].items():
        lines.append(f"| {key} | {summary['baseline_nested']['pooled_confusion_counts'][key]} | {count} |")
    ci = summary["paired_conditional_f1_delta_interval_95"]
    lines += ["", f"Criterio operativo del piano: {'soddisfatto' if summary['meets_plan_promotion_rule'] else 'non soddisfatto'}; fold favorevoli {summary['positive_fold_count']}/3. Configurazione finale invariata.", "",
              f"Bootstrap descrittivo del delta F1: [{ci[0]:+.9f}; {ci[1]:+.9f}], 20.000 ripetizioni con modelli e soglie fissi. Non misura l'incertezza da riaddestramento o selezione ripetuta.", "",
              f"Tempo della CV: {summary['runtime_seconds']/60:.2f} minuti, con un altro esperimento eseguito in parallelo. Preflight e verifiche finali esclusi. Ambiente: {summary['environment']}.", "",
              "Nove partizioni verificate; colonne e bin invariati salvo la modifica richiesta. Checkpoint, probabilità, righe, hash, soglie interne e metriche riprodotti. Il 20% iniziale non è utilizzato; i fold development sono già stati consultati per più esperimenti.", "",
              f"Record: {summary['candidate_record']}", f"Artefatti e confronto: {suite['artifact_dir']}/", ""]
    (root / f"results/eda/{PREFIXES[phase]}_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", type=int, choices=(27, 28))
    parser.add_argument("action", choices=("preflight", "run", "compare"))
    args = parser.parse_args()
    root = Path.cwd()
    if args.action == "preflight":
        preflight(root, args.phase)
        return
    suite, _, _ = configuration(root, args.phase)
    directory = root / suite["artifact_dir"]
    records = list((directory / "records").glob("*.json"))
    if args.action == "compare":
        if len(records) != 1:
            raise ValueError("Expected one completed record.")
        compare(root, args.phase, records[0])
        return
    if (directory / "executed_runner.py").exists() or records:
        raise ValueError("Existing training artifacts: inspect before rerunning.")
    check = read(root, paths(args.phase)[1])
    verify_frozen(root, args.phase, suite, check)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "executed_runner.py").write_bytes(Path(__file__).read_bytes())
    (directory / "training_preflight.json").write_bytes((root / paths(args.phase)[1]).read_bytes())
    with (directory / "run_stdout.log").open("w", encoding="utf-8") as log:
        tee = _Tee(sys.stdout, log)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            records = run_logged_suite(root / paths(args.phase)[0], root, directory / "records")
            if len(records) != 1:
                raise ValueError("Expected one completed candidate.")
            compare(root, args.phase, records[0])


if __name__ == "__main__":
    main()

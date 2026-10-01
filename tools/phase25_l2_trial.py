"""Run and verify the isolated L2=5 comparison on Phase 14 preprocessing."""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import json
import sys
from pathlib import Path

import numpy as np

from src.evaluation import average_precision, best_f1_threshold, binary_log_loss, classification_metrics
from src.feature_preprocessing import tree_feature_matrices
from src.numpy_boosting import HistogramBinner, HistogramGradientBoostingClassifier
from src.run_boosting_cv import _array_sha256, _metrics, _validate_suite, _variant
from src.run_logged_boosting_cv import run_logged_suite
from src.run_preprocessing_cv import _sha256, _stratified_folds
from src.summarize_phase19_transfer import BASELINE, _arrays, _paired_interval, _publish
from tools.phase23_missing_reason_flags import SOURCES
from tools.run_targeted_preprocessing_trials import _Tee
from tools.targeted_preprocessing_analysis import _input_data, _partitions


PREFIX = "phase25_phase14_l2_5"
SUITE = f"configs/experiments/{PREFIX}.json"
PREFLIGHT = "results/eda/phase25_l2_preflight.json"
COMPARISON = "results/experiments/phase25_l2_comparison.json"
REPORT = "results/eda/phase25_l2_summary.md"


def read(root, path):
    return json.loads((root / path).read_text(encoding="utf-8"))


def validate_change(candidate, reference):
    """Reject any change except the requested L2 and output/provenance fields."""
    _validate_suite(candidate)
    expected = copy.deepcopy(reference)
    expected["models"][0]["parameters"]["l2_regularization"] = 5.0
    ignored = {"experiment_prefix", "artifact_dir", "hypothesis"}
    if ({k: v for k, v in candidate.items() if k not in ignored}
            != {k: v for k, v in expected.items() if k not in ignored}):
        raise ValueError("Candidate must change only L2 from 1 to 5.")
    if reference["models"][0]["parameters"]["l2_regularization"] != 1.0:
        raise ValueError("Expected Phase 14 L2=1 reference.")


def configuration(root):
    suite = read(root, SUITE)
    validate_change(suite, read(root, "configs/experiments/phase14_boosting_codebook_corrections_only.json"))
    variant = _variant(read(root, suite["preprocessing_config_path"]), suite["preprocessing_variant"])
    return suite, variant["preprocessing"]


def preflight(root):
    suite, plan = configuration(root)
    baseline = read(root, BASELINE)
    parameters = {**suite["models"][0]["parameters"], "n_estimators": 200}
    if parameters != {**baseline["config"]["model"]["parameters"], "l2_regularization": 5.0}:
        raise ValueError("Candidate parameters do not match the recorded reference.")
    fingerprints = {key: _sha256(root / suite["data"][field]) for key, field in (
        ("features", "features_path"), ("labels", "labels_path"), ("split_indices", "split_indices_path"))}
    fingerprints.update(feature_metadata=_sha256(root / suite["feature_metadata_path"]),
                        preprocessing_config=_sha256(root / suite["preprocessing_config_path"]))
    for key, value in fingerprints.items():
        if value != baseline["outcome"]["input_sha256"][key]:
            raise ValueError(f"Reference input changed: {key}")
    fingerprints["boosting_suite_config"] = _sha256(root / SUITE)
    x, y, development, names, metadata = _input_data(root, suite)
    ba = _arrays(root, baseline)
    np.testing.assert_array_equal(ba["development_indices"], development)
    np.testing.assert_array_equal(ba["labels"], y[development])
    check = {"scope": "Development only; baseline checkpoint inference, no new training.",
        "input_sha256": fingerprints, "source_sha256": {p: _sha256(root / p) for p in SOURCES},
        "helper_sha256": _sha256(Path(__file__)), "final_config_sha256": _sha256(root / "configs/final_model.json"),
        "baseline_record_sha256": _sha256(root / BASELINE), "partitions": [], "baseline_checkpoint_checks": []}
    for partition, train, valid in _partitions(y, development, suite):
        ct, cv, cn = tree_feature_matrices(x[development[train]], x[development[valid]], names, metadata, plan)
        if len(cn) != 295:
            raise ValueError("Expected unchanged 295-column representation.")
        binner = HistogramBinner(64).fit(ct)
        check["partitions"].append({"partition": partition, "feature_count": 295, "output_names": cn,
            "training_rows": int(train.size), "validation_rows": int(valid.size),
            "bin_counts": binner.bin_counts_.tolist(), "exact_column_indices": [],
            "training_row_indices_sha256": _array_sha256(development[train]),
            "training_labels_sha256": _array_sha256(y[development[train]])})
        if "inner" not in partition:
            fold = int(partition[-1])
            np.testing.assert_array_equal(ba["fold_ids"][valid], np.full(valid.size, fold))
            cp = baseline["outcome"]["model_checkpoints"][fold - 1]
            if _sha256(root / cp["path"]) != cp["sha256"]:
                raise ValueError("Baseline checkpoint fingerprint differs.")
            model, saved = HistogramGradientBoostingClassifier.load_checkpoint(root / cp["path"])
            np.testing.assert_array_equal(saved["training_row_indices"], development[train])
            np.testing.assert_array_equal(binner.bin_counts_, model.binner_.bin_counts_)
            for current, previous in zip(binner.edges_, model.binner_.edges_):
                np.testing.assert_array_equal(current, previous)
            delta = float(np.max(np.abs(model.predict_proba(cv)[:, 1] - ba["probabilities"][valid])))
            if delta != 0:
                raise ValueError("Baseline probabilities cannot be reproduced exactly.")
            check["baseline_checkpoint_checks"].append({"fold": fold, "maximum_probability_difference": delta})
            del model
        print(f"PREFLIGHT {partition}: 295 Phase 14 inputs and quantile bins verified", flush=True)
        del ct, cv, binner
        gc.collect()
    (root / PREFLIGHT).write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
    return check


def verify_frozen(root, suite, check):
    for path, expected in check["source_sha256"].items():
        if _sha256(root / path) != expected:
            raise ValueError(f"Source changed after verification: {path}")
    if _sha256(Path(__file__)) != check["helper_sha256"]:
        raise ValueError("Trial helper changed after preflight.")
    if (_sha256(root / "configs/final_model.json") != check["final_config_sha256"]
            or _sha256(root / BASELINE) != check["baseline_record_sha256"]):
        raise ValueError("Final configuration or reference record changed.")
    current = {key: _sha256(root / suite["data"][field]) for key, field in (
        ("features", "features_path"), ("labels", "labels_path"), ("split_indices", "split_indices_path"))}
    current.update(feature_metadata=_sha256(root / suite["feature_metadata_path"]),
        preprocessing_config=_sha256(root / suite["preprocessing_config_path"]), boosting_suite_config=_sha256(root / SUITE))
    if current != check["input_sha256"]:
        raise ValueError("Training inputs or configuration changed.")


def compare(root, source):
    suite, plan = configuration(root)
    check = read(root, PREFLIGHT)
    verify_frozen(root, suite, check)
    candidate, baseline = read(root, source), read(root, BASELINE)
    outcome = candidate["outcome"]
    if outcome["input_sha256"] != check["input_sha256"] or outcome["source_sha256"] != check["source_sha256"]:
        raise ValueError("Actual training provenance differs from preflight.")
    if (candidate["config"]["features"] != baseline["config"]["features"]
            or candidate["config"]["split"] != baseline["config"]["split"]
            or candidate["config"]["model"]["parameters"] != {**baseline["config"]["model"]["parameters"], "l2_regularization": 5.0}):
        raise ValueError("Actual experiment changed more than L2.")
    profiles = outcome["individual_fit_profiles"]
    if len(profiles) != 9:
        raise ValueError("Expected nine complete fits.")
    for index, (profile, expected) in enumerate(zip(profiles, check["partitions"]), 1):
        if profile["stage"] != ("outer" if index <= 3 else "inner") or profile["binning_strategy"] != "quantile":
            raise ValueError("Wrong fit stage or binning strategy.")
        for key in ("feature_count", "training_rows", "validation_rows", "bin_counts", "exact_column_indices"):
            if profile[key] != expected[key]:
                raise ValueError(f"Actual fit differs from preflight: {key}")
    ca, ba = _arrays(root, candidate), _arrays(root, baseline)
    for key in ("development_indices", "fold_ids", "labels"):
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
            raise ValueError("Candidate checkpoint fingerprint or fold differs.")
        model, saved = HistogramGradientBoostingClassifier.load_checkpoint(root / cp["path"])
        np.testing.assert_array_equal(saved["training_row_indices"], development[train])
        expected = check["partitions"][fold - 1]
        for key in ("training_row_indices_sha256", "training_labels_sha256"):
            if saved[key] != expected[key]:
                raise ValueError(f"Checkpoint training boundary differs: {key}")
        if saved["input_sha256"] != check["input_sha256"]:
            raise ValueError("Checkpoint training provenance differs.")
        for key, value in candidate["config"]["model"]["parameters"].items():
            if getattr(model, key) != value:
                raise ValueError(f"Actual checkpoint parameter differs: {key}")
        ct, cv, cn = tree_feature_matrices(x[development[train]], x[development[valid]], names, metadata, plan)
        if cn != saved["output_feature_names"] or cn != expected["output_names"]:
            raise ValueError("Checkpoint feature order differs.")
        delta = float(np.max(np.abs(model.predict_proba(cv)[:, 1] - ca["probabilities"][valid])))
        if delta != 0:
            raise ValueError("Checkpoint probabilities differ from saved OOF.")
        detail = nested["folds"][fold - 1]
        info = detail["inner_oof_predictions_artifact"]
        if _sha256(root / info["path"]) != info["sha256"]:
            raise ValueError("Inner OOF fingerprint differs.")
        with np.load(root / info["path"]) as inner:
            for key, expected_hash in info["array_sha256"].items():
                if _array_sha256(inner[key]) != expected_hash:
                    raise ValueError(f"Inner OOF array differs: {key}")
            np.testing.assert_array_equal(inner["training_row_indices"], development[train])
            np.testing.assert_array_equal(inner["outer_validation_row_indices"], development[valid])
            np.testing.assert_array_equal(inner["labels"], y[development[train]])
            if np.intersect1d(inner["training_row_indices"], inner["outer_validation_row_indices"]).size:
                raise ValueError("Outer evaluation rows entered inner threshold selection.")
            ids = np.zeros(train.size, dtype=np.int8)
            for number, (_, iv) in enumerate(_stratified_folds(y[development[train]], 2, suite["nested_threshold"]["inner_seed"] + fold), 1):
                ids[iv] = number
            np.testing.assert_array_equal(inner["fold_ids"], ids)
            threshold = best_f1_threshold(inner["labels"], inner["probabilities"]).threshold
        if threshold != detail["inner_threshold"]:
            raise ValueError("Inner threshold cannot be reproduced.")
        np.testing.assert_array_equal(ca["nested_thresholds"][valid], np.full(valid.size, threshold))
        fm, fc = _metrics(ca["labels"][valid], ca["probabilities"][valid], threshold)
        if fm != detail["outer_evaluation_metrics"] or fc != detail["outer_evaluation_confusion_counts"]:
            raise ValueError("External fold metrics cannot be reproduced.")
        verified.append({"fold": fold, "maximum_probability_difference": delta, "reproduced_inner_threshold": threshold})
        print(f"VERIFIED fold={fold}: L2=5 checkpoint, rows, threshold and metrics identical", flush=True)
        del model, ct, cv
        gc.collect()
    bn = baseline["outcome"]["nested_threshold_evaluation"]
    deltas = {k: v - bn["pooled_metrics"][k] for k, v in metrics.items()}
    fold_deltas = [c["outer_evaluation_metrics"]["f1"] - b["outer_evaluation_metrics"]["f1"]
                   for c, b in zip(nested["folds"], bn["folds"])]
    published = _publish(source, root)
    bp, cp, labels = ba["nested_predictions"], ca["nested_predictions"], ca["labels"]
    summary = {"phase": 25, "baseline_record": BASELINE,
        "candidate_record": str(published.relative_to(root)).replace("\\", "/"),
        "changed_parameters": {"l2_regularization": [1.0, 5.0]}, "baseline_nested": bn, "candidate_nested": nested,
        "metric_deltas": deltas, "fold_f1_deltas": fold_deltas, "positive_fold_count": sum(d > 0 for d in fold_deltas),
        "meets_plan_promotion_rule": deltas["f1"] > 0 and sum(d > 0 for d in fold_deltas) >= 2,
        "paired_conditional_f1_delta_interval_95": _paired_interval(labels, bp, cp),
        "bootstrap_seed": 20260929, "bootstrap_repetitions": 20000,
        "bootstrap_scope": "Fixed models and thresholds; excludes training and repeated candidate-selection uncertainty.",
        "decision_changes": {"recovered_false_negatives": int(np.sum((labels == 1) & (bp == 0) & (cp == 1))),
            "lost_true_positives": int(np.sum((labels == 1) & (bp == 1) & (cp == 0))),
            "added_false_positives": int(np.sum((labels == 0) & (bp == 0) & (cp == 1))),
            "removed_false_positives": int(np.sum((labels == 0) & (bp == 1) & (cp == 0)))},
        "runtime_seconds": outcome["runtime_seconds"], "environment": outcome["environment"],
        "individual_fit_profiles": profiles, "checkpoint_and_inner_oof_checks": verified, "final_model_changed": False,
        "exploratory_same_oof_threshold_f1": outcome["metrics"]["f1"]}
    (root / COMPARISON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(root, summary)
    print(json.dumps({"phase": 25, "f1": metrics["f1"], "delta_f1": deltas["f1"], "fold_deltas": fold_deltas,
                      "meets_plan_rule": summary["meets_plan_promotion_rule"]}), flush=True)
    return summary


def write_report(root, summary):
    b, c = summary["baseline_nested"], summary["candidate_nested"]
    lines = ["# Phase 25 — L2 delle foglie da 1 a 5", "", "## Prova e protocollo", "",
        "Una sola modifica a Phase 14: l2_regularization=5 anziché 1. Stesse 295 feature, incluso HAREHAB1, preprocessing codebook, binning a quantili e gestione dei NaN. Modello: 200 alberi, profondità 5, learning rate 0,05, minimo foglia 200, 64 bin e 128 feature candidate. Seed booster 20260920.", "",
        "262.508 righe development; split esterni seed 20260919 e due fold interni per esterno, con seed 20260922 + fold. Nove nuovi training del candidato. Ogni soglia è scelta solo negli OOF interni. Il controllo usa gli artefatti Phase 14 verificati, senza nuovo training della baseline.", "",
        "## Risultati annidati", "", "| Metrica | Phase 14 | L2=5 | Differenza |", "| --- | ---: | ---: | ---: |"]
    for key in ("f1", "precision", "recall", "average_precision", "log_loss"):
        lines.append(f"| {key} | {b['pooled_metrics'][key]:.9f} | {c['pooled_metrics'][key]:.9f} | {summary['metric_deltas'][key]:+.9f} |")
    lines += ["", "## Fold e soglie", "", "| Fold | F1 Phase 14 | F1 L2=5 | Delta | Soglia Phase 14 | Soglia L2=5 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for i, (bf, cf) in enumerate(zip(b["folds"], c["folds"]), 1):
        lines.append(f"| {i} | {bf['outer_evaluation_metrics']['f1']:.9f} | {cf['outer_evaluation_metrics']['f1']:.9f} | {summary['fold_f1_deltas'][i-1]:+.9f} | {bf['inner_threshold']:.9f} | {cf['inner_threshold']:.9f} |")
    lines += ["", "## Matrice di confusione", "", "| Esito | Phase 14 | L2=5 | Differenza |", "| --- | ---: | ---: | ---: |"]
    for key in ("true_positives", "false_negatives", "false_positives", "true_negatives"):
        bv, cv = b["pooled_confusion_counts"][key], c["pooled_confusion_counts"][key]
        lines.append(f"| {key} | {bv} | {cv} | {cv-bv:+d} |")
    ci = summary["paired_conditional_f1_delta_interval_95"]
    passed = summary["meets_plan_promotion_rule"]
    lines += ["", "## Interpretazione e decisione", "",
        f"F1 aggregato cambia di {summary['metric_deltas']['f1']:+.9f}; fold favorevoli {summary['positive_fold_count']}/3. Il criterio operativo richiede F1 superiore e almeno due fold favorevoli. Criterio {'soddisfatto' if passed else 'non soddisfatto'}.", "",
        ("L2=5 è un candidato favorevole secondo il piano; la configurazione finale resta invariata durante questo confronto. La selezione e l'eventuale training finale sono un passo successivo."
         if passed else "L2=5 non viene adottata. Conservare Phase 14 con L2=1; questa configurazione è ora una prova conclusa."), "",
        f"Intervallo descrittivo bootstrap appaiato 95% del delta F1: [{ci[0]:+.9f}, {ci[1]:+.9f}], 20.000 ripetizioni, seed 20260929. Modelli e soglie fissi; non comprende riaddestramento o selezione ripetuta dei candidati.", "",
        f"F1 esplorativo, con soglia scelta sugli stessi OOF esterni valutati: {summary['exploratory_same_oof_threshold_f1']:.9f}. La decisione usa l'F1 annidato sopra riportato.", "",
        "## Verifiche, tempi e artefatti", "",
        "Preflight su tutte le nove partizioni; hash degli input uguali a Phase 14 e output delle tre baseline riprodotti esattamente. Dopo training: parametri L2=5 dei checkpoint, ordine delle feature, righe e label di training verificati; output dei tre checkpoint identici agli OOF. Soglie e metriche ricostruite dai tre artefatti OOF interni con righe, fold e hash. Sorgenti e configurazioni congelate durante la prova.", "",
        f"Nove fit: {summary['runtime_seconds']/60:.2f} minuti, esclusi preflight e verifiche finali. Ambiente: Python {summary['environment']['python']}, NumPy {summary['environment']['numpy']}. Ambiente ufficiale Python 3.9 / NumPy 1.23.1 ancora da verificare.", "",
        "Il 20% iniziale non è utilizzato, ma era già stato consultato storicamente; anche la selezione ripetuta sui fold development limita la stima finale. Nessuna nuova submission prodotta da questo esperimento.", "",
        f"- [Record](../experiments/{Path(summary['candidate_record']).name}).",
        f"- [Confronto](../experiments/{Path(COMPARISON).name}).",
        "- [Preflight](phase25_l2_preflight.json).",
        f"- Checkpoint, OOF esterni e interni, versione del helper e log: results/boosting_artifacts/{PREFIX}/.", ""]
    (root / REPORT).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run", "compare"))
    args = parser.parse_args()
    root = Path.cwd()
    if args.action == "preflight":
        preflight(root)
        return
    suite, _ = configuration(root)
    directory = root / suite["artifact_dir"]
    records = list((directory / "records").glob("*.json"))
    if args.action == "compare":
        if len(records) != 1:
            raise ValueError("Expected one complete candidate record.")
        compare(root, records[0])
        return
    if records or list((directory / "model_checkpoints").glob("*.pkl")) or list((directory / "inner_oof_predictions").glob("*.npz")):
        raise ValueError("Existing Phase 25 artifacts: inspect before retraining.")
    check = read(root, PREFLIGHT)
    verify_frozen(root, suite, check)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "executed_runner.py").write_bytes(Path(__file__).read_bytes())
    (directory / "training_preflight.json").write_bytes((root / PREFLIGHT).read_bytes())
    with (directory / "run_stdout.log").open("w", encoding="utf-8") as log:
        tee = _Tee(sys.stdout, log)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            records = run_logged_suite(root / SUITE, root, directory / "records")
            if len(records) != 1:
                raise ValueError("Expected one complete L2 candidate.")
            compare(root, records[0])


if __name__ == "__main__":
    main()

"""Verify, run and compare the indicators-only experiment against Phase 14."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
from pathlib import Path

import numpy as np

from src.conditional_imputation import _response_code_groups
from src.evaluation import best_f1_threshold, classification_metrics, average_precision, binary_log_loss
from src.feature_preprocessing import FeaturePreprocessor, tree_feature_matrices
from src.numpy_boosting import HistogramBinner, HistogramGradientBoostingClassifier
from src.run_boosting_cv import _resolved_parameters, _array_sha256, _validate_suite
from src.run_logged_boosting_cv import run_logged_suite
from src.run_preprocessing_cv import _sha256, _stratified_folds
from src.summarize_phase19_transfer import BASELINE, _arrays, _paired_interval, _publish
from tools.targeted_preprocessing_analysis import _input_data, _partitions
from tools.run_targeted_preprocessing_trials import _Tee


PREFIX = "phase23_phase14_missing_reason_flags"
SUITE = f"configs/experiments/{PREFIX}.json"
PREFLIGHT = "results/eda/phase23_missing_reason_flags_preflight.json"
FLAGS = {
    "INCOME2__was_unknown", "INCOME2__was_refused", "EMPLOY1__was_refused",
    "PNEUVAC3__was_unknown", "PNEUVAC3__was_refused", "HTM4__was_blank",
    "WTKG3__was_blank", "_BMI5__was_blank",
}
SOURCES = (
    "src/numpy_boosting.py", "src/feature_preprocessing.py", "src/conditional_imputation.py",
    "src/run_boosting_cv.py", "src/run_logged_boosting_cv.py", "src/evaluation.py",
)


def _read(root, path):
    return json.loads((root / path).read_text(encoding="utf-8"))


def _configuration(root):
    suite = _read(root, SUITE)
    _validate_suite(suite)
    prep = _read(root, suite["preprocessing_config_path"])
    plan = next(v["preprocessing"] for v in prep["variants"]
                if v["experiment_name"] == suite["preprocessing_variant"])
    return suite, plan


def preflight(root):
    suite, plan = _configuration(root)
    baseline = _read(root, BASELINE)
    base_suite = _read(root, "configs/experiments/phase14_boosting_codebook_corrections_only.json")
    base_prep = _read(root, base_suite["preprocessing_config_path"])
    base_plan = next(v["preprocessing"] for v in base_prep["variants"]
                     if v["experiment_name"] == base_suite["preprocessing_variant"])
    restricted = _read(root, "configs/experiments/phase16_selected_imputation_preprocessing.json")
    settings = next(v["preprocessing"]["imputation"] for v in restricted["variants"]
                    if v["experiment_name"] == "phase16_imputation_restricted")
    if plan != {**base_plan, "imputation": {**settings, "fill_values": False}}:
        raise ValueError("Unexpected change to Phase 14 preprocessing.")
    for key in ("data", "feature_metadata_path", "split_id", "fold_count", "seed",
                "fixed_threshold", "nested_threshold"):
        if suite[key] != base_suite[key]:
            raise ValueError(f"Changed protocol: {key}")
    parameters = dict(suite["models"][0]["parameters"])
    if parameters.pop("binning_strategy") != "exact_selected":
        raise ValueError("The indicators must use selective exact binning.")
    if set(parameters.pop("exact_feature_names")) != FLAGS:
        raise ValueError("Expected exactly the eight restricted indicators.")
    if {**parameters, "n_estimators": 200} != baseline["config"]["model"]["parameters"]:
        raise ValueError("Other model parameters changed.")
    x, y, development, names, metadata = _input_data(root, suite)
    barray = _arrays(root, baseline)
    np.testing.assert_array_equal(development, barray["development_indices"])
    np.testing.assert_array_equal(y[development], barray["labels"])
    fingerprints = {key: _sha256(root / suite["data"][field]) for key, field in
        (("features", "features_path"), ("labels", "labels_path"), ("split_indices", "split_indices_path"))}
    fingerprints["feature_metadata"] = _sha256(root / suite["feature_metadata_path"])
    for key, value in fingerprints.items():
        if value != baseline["outcome"]["input_sha256"][key]:
            raise ValueError(f"Baseline input changed: {key}")
    processor = FeaturePreprocessor(names, metadata, plan)
    result = {"scope": "Development only; no predictive model training.",
        "input_sha256": fingerprints, "suite_sha256": _sha256(root / SUITE),
        "preprocessing_sha256": _sha256(root / suite["preprocessing_config_path"]),
        "source_sha256": {p: _sha256(root / p) for p in SOURCES},
        "final_config_sha256": _sha256(root / "configs/final_model.json"),
        "partitions": [], "baseline_checkpoint_checks": []}
    for partition, train, valid in _partitions(y, development, suite):
        rt, rv = x[development[train]], x[development[valid]]
        bt, bv, bn = tree_feature_matrices(rt, rv, names, metadata, base_plan)
        ct, cv, cn = tree_feature_matrices(rt, rv, names, metadata, plan)
        if len(bn) != 295 or cn[:295] != bn or len(cn) != 303 or set(cn[295:]) != FLAGS:
            raise ValueError("The source representation or indicator list changed.")
        np.testing.assert_array_equal(ct[:, :295], bt)
        np.testing.assert_array_equal(cv[:, :295], bv)
        params = _resolved_parameters(suite["models"][0]["parameters"], cn)
        original = HistogramBinner(64).fit(bt)
        candidate = HistogramBinner(64, params["binning_strategy"], params["exact_feature_indices"]).fit(ct)
        encoded = candidate.transform(ct)
        np.testing.assert_array_equal(encoded == 255, np.isnan(ct))
        np.testing.assert_array_equal(candidate.exact_columns_, np.r_[np.zeros(295, bool), np.ones(8, bool)])
        for column in range(295):
            np.testing.assert_array_equal(original.edges_[column], candidate.edges_[column])
        flags = {}
        for column, flag in enumerate(cn[295:], 295):
            source, reason = flag.split("__was_")
            for raw, cleaned, output in ((rt, bt, ct), (rv, bv, cv)):
                if source == "_BMI5":
                    expected = ~np.isfinite(cleaned[:, bn.index(source)])
                elif reason == "blank":
                    expected = np.isnan(raw[:, names.index(source)])
                else:
                    codes = _response_code_groups(processor, source, processor.metadata_by_name[source])[reason]
                    expected = np.isin(raw[:, names.index(source)], codes)
                np.testing.assert_array_equal(output[:, column], expected.astype(np.float32))
            np.testing.assert_array_equal(np.unique(ct[:, column]), [0, 1])
            np.testing.assert_array_equal(np.unique(encoded[:, column]), [0, 1])
            flags[flag] = {"training_ones": int(ct[:, column].sum()),
                           "validation_ones": int(cv[:, column].sum())}
        detail = {"partition": partition, "training_rows": int(train.size), "validation_rows": int(valid.size),
            "output_names": cn, "feature_count": 303, "flags": flags,
            "exact_column_indices": params["exact_feature_indices"], "bin_counts": candidate.bin_counts_.tolist(),
            "training_row_indices_sha256": _array_sha256(development[train]),
            "training_labels_sha256": _array_sha256(y[development[train]])}
        result["partitions"].append(detail)
        if "inner" not in partition:
            fold = int(partition[-1])
            cp = baseline["outcome"]["model_checkpoints"][fold - 1]
            if _sha256(root / cp["path"]) != cp["sha256"]:
                raise ValueError("Baseline checkpoint changed.")
            model, _ = HistogramGradientBoostingClassifier.load_checkpoint(root / cp["path"])
            delta = float(np.max(np.abs(model.predict_proba(bv)[:, 1] - barray["probabilities"][valid])))
            if delta != 0:
                raise ValueError("Phase 14 predictions changed.")
            result["baseline_checkpoint_checks"].append({"fold": fold, "maximum_probability_difference": delta})
            del model
        print(f"PREFLIGHT {partition}: 295 source columns unchanged; 8 exact flags OK", flush=True)
        del rt, rv, bt, bv, ct, cv, original, candidate, encoded
        gc.collect()
    (root / PREFLIGHT).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def compare(root, source):
    check = _read(root, PREFLIGHT)
    suite, plan = _configuration(root)
    record, baseline = _read(root, source), _read(root, BASELINE)
    for path, expected in check["source_sha256"].items():
        if _sha256(root / path) != expected or record["outcome"]["source_sha256"][path] != expected:
            raise ValueError(f"Source changed during training: {path}")
    if (_sha256(root / SUITE) != check["suite_sha256"]
            or _sha256(root / suite["preprocessing_config_path"]) != check["preprocessing_sha256"]
            or _sha256(root / "configs/final_model.json") != check["final_config_sha256"]):
        raise ValueError("Configuration changed during training.")
    expected_inputs = {**check["input_sha256"], "boosting_suite_config": check["suite_sha256"],
                       "preprocessing_config": check["preprocessing_sha256"]}
    if record["outcome"]["input_sha256"] != expected_inputs:
        raise ValueError("Training inputs differ from preflight.")
    if (record["config"]["model"]["parameters"] != {**suite["models"][0]["parameters"], "n_estimators": 200}
            or record["config"]["split"] != baseline["config"]["split"]):
        raise ValueError("Model settings or split differ from the requested experiment.")
    profiles = record["outcome"]["individual_fit_profiles"]
    if len(profiles) != 9 or len(check["partitions"]) != 9:
        raise ValueError("Expected nine complete fits.")
    for profile, detail in zip(profiles, check["partitions"]):
        for key in ("feature_count", "exact_column_indices", "bin_counts"):
            if profile[key] != detail[key]:
                raise ValueError(f"Actual binning differs from preflight: {key}")
        if profile["binning_strategy"] != "exact_selected":
            raise ValueError("Unexpected training binning strategy.")
    ba, ca = _arrays(root, baseline), _arrays(root, record)
    for key in ("development_indices", "fold_ids", "labels"):
        np.testing.assert_array_equal(ba[key], ca[key])
    np.testing.assert_array_equal(ca["nested_predictions"], ca["probabilities"] >= ca["nested_thresholds"])
    nested = record["outcome"]["nested_threshold_evaluation"]
    counts, metrics = classification_metrics(ca["labels"], ca["nested_predictions"])
    metrics.update(average_precision=average_precision(ca["labels"], ca["probabilities"]),
                   log_loss=binary_log_loss(ca["labels"], ca["probabilities"]))
    for key, value in metrics.items():
        if not np.isclose(value, nested["pooled_metrics"][key], rtol=0, atol=1e-15):
            raise ValueError(f"Recomputed metric differs: {key}")
    x, y, development, names, metadata = _input_data(root, suite)
    verified = []
    for fold, (train, valid) in enumerate(_stratified_folds(y[development], 3, suite["seed"]), 1):
        cp = record["outcome"]["model_checkpoints"][fold - 1]
        if _sha256(root / cp["path"]) != cp["sha256"]:
            raise ValueError("Candidate checkpoint changed.")
        model, saved = HistogramGradientBoostingClassifier.load_checkpoint(root / cp["path"])
        detail = check["partitions"][fold - 1]
        for key in ("training_row_indices_sha256", "training_labels_sha256"):
            if saved[key] != detail[key]:
                raise ValueError(f"Checkpoint boundary mismatch: {key}")
        ct, cv, cn = tree_feature_matrices(x[development[train]], x[development[valid]], names, metadata, plan)
        if cn != saved["output_feature_names"] or cn != detail["output_names"]:
            raise ValueError("Checkpoint feature order differs.")
        delta = float(np.max(np.abs(model.predict_proba(cv)[:, 1] - ca["probabilities"][valid])))
        if delta != 0:
            raise ValueError("Checkpoint probabilities differ from OOF.")
        fold_result = nested["folds"][fold - 1]
        info = fold_result["inner_oof_predictions_artifact"]
        if _sha256(root / info["path"]) != info["sha256"]:
            raise ValueError("Inner OOF artifact hash differs.")
        with np.load(root / info["path"]) as inner:
            for key, expected in info["array_sha256"].items():
                if _array_sha256(inner[key]) != expected:
                    raise ValueError(f"Inner OOF array hash differs: {key}")
            np.testing.assert_array_equal(inner["training_row_indices"], development[train])
            np.testing.assert_array_equal(inner["outer_validation_row_indices"], development[valid])
            np.testing.assert_array_equal(inner["labels"], y[development[train]])
            ids = np.zeros(train.size, dtype=np.int8)
            for index, (_, iv) in enumerate(_stratified_folds(y[development[train]], 2,
                    suite["nested_threshold"]["inner_seed"] + fold), 1):
                ids[iv] = index
            np.testing.assert_array_equal(inner["fold_ids"], ids)
            threshold = best_f1_threshold(inner["labels"], inner["probabilities"]).threshold
        if threshold != fold_result["inner_threshold"]:
            raise ValueError("Stored threshold cannot be reproduced from inner OOF.")
        np.testing.assert_array_equal(ca["nested_thresholds"][valid], np.full(valid.size, threshold))
        _, fm = classification_metrics(ca["labels"][valid], ca["nested_predictions"][valid])
        if fm["f1"] != fold_result["outer_evaluation_metrics"]["f1"]:
            raise ValueError("Fold F1 differs.")
        verified.append({"fold": fold, "maximum_probability_difference": delta,
                         "inner_threshold_reproduced": threshold})
        del model, ct, cv
        gc.collect()
        print(f"VERIFIED fold={fold}: checkpoint and inner threshold identical", flush=True)
    bn = baseline["outcome"]["nested_threshold_evaluation"]
    deltas = {k: v - bn["pooled_metrics"][k] for k, v in nested["pooled_metrics"].items()}
    fd = [c["outer_evaluation_metrics"]["f1"] - b["outer_evaluation_metrics"]["f1"]
          for c, b in zip(nested["folds"], bn["folds"])]
    bp, cp, labels = ba["nested_predictions"], ca["nested_predictions"], ca["labels"]
    published = _publish(source, root)
    summary = {"phase": 23, "baseline_record": BASELINE,
        "candidate_record": str(published.relative_to(root)).replace("\\", "/"),
        "baseline_nested": bn, "candidate_nested": nested, "metric_deltas": deltas,
        "fold_f1_deltas": fd, "positive_fold_count": sum(d > 0 for d in fd),
        "meets_plan_promotion_rule": deltas["f1"] > 0 and sum(d > 0 for d in fd) >= 2,
        "decision_changes": {
            "recovered_false_negatives": int(np.sum((labels == 1) & (bp == 0) & (cp == 1))),
            "lost_true_positives": int(np.sum((labels == 1) & (bp == 1) & (cp == 0))),
            "added_false_positives": int(np.sum((labels == 0) & (bp == 0) & (cp == 1))),
            "removed_false_positives": int(np.sum((labels == 0) & (bp == 1) & (cp == 0)))},
        "paired_conditional_f1_delta_interval_95": _paired_interval(labels, bp, cp),
        "bootstrap_repetitions": 20000, "bootstrap_seed": 20260929,
        "bootstrap_scope": "Fixed models and thresholds; excludes retraining and repeated selection uncertainty.",
        "runtime_seconds": record["outcome"]["runtime_seconds"], "environment": record["outcome"]["environment"],
        "checkpoint_and_inner_oof_checks": verified, "final_model_changed": False,
        "exploratory_same_oof_threshold_f1": record["outcome"]["metrics"]["f1"]}
    (root / "results/experiments/phase23_missing_reason_flags_comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(root, summary)
    print(json.dumps({"phase": 23, "f1": nested["pooled_metrics"]["f1"], "delta_f1": deltas["f1"],
                      "fold_deltas": fd, "meets_plan_rule": summary["meets_plan_promotion_rule"]}), flush=True)
    return summary


def write_report(root, summary):
    b, c = summary["baseline_nested"], summary["candidate_nested"]
    lines = ["# Phase 23 — soli otto indicatori dei mancanti", "", "## Cosa è stato provato", "",
        "Phase 14 più gli otto indicatori della Phase 16 ristretta: non so/rifiuto di INCOME2 e PNEUVAC3, rifiuto di EMPLOY1 e mancanza di HTM4, WTKG3 e _BMI5. Nessun valore è imputato. Le 295 colonne originali e i loro quantili sono invariati; soltanto gli otto indicatori usano binning esatto. Totale: 303 input.", "",
        "Stessi dati development (262.508 righe), seed e parametri Phase 14: 200 alberi, profondità 5, learning rate 0,05, minimo foglia 200, L2=1, 64 bin e 128 feature candidate per nodo. HAREHAB1 resta inclusa. Tre fold esterni e due interni: nove fit. Ogni soglia è scelta soltanto sugli OOF interni e valutata sul fold esterno.", "",
        "Il controllo riutilizza gli artefatti Phase 14 verificati e non viene riaddestrato. L'audit verifica tutte le nove partizioni, l'identità delle colonne sorgente e i bin degli indicatori. Dopo il training si verificano i tre checkpoint e si ricalcolano le soglie dagli OOF interni ora salvati.", "",
        "## Risultati annidati", "", "| Modello | F1 | Precision | Recall | AP | Log loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, result in (("Phase 14", b), ("Phase 23: soli indicatori", c)):
        m = result["pooled_metrics"]
        lines.append(f"| {name} | {m['f1']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['average_precision']:.6f} | {m['log_loss']:.6f} |")
    lines += ["", "## Risultati per fold", "", "| Fold | F1 Phase 14 | F1 indicatori | Delta F1 | Soglia indicatori |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for index, (bf, cf, delta) in enumerate(zip(b["folds"], c["folds"], summary["fold_f1_deltas"]), 1):
        lines.append(f"| {index} | {bf['outer_evaluation_metrics']['f1']:.6f} | {cf['outer_evaluation_metrics']['f1']:.6f} | {delta:+.6f} | {cf['inner_threshold']:.6f} |")
    lines += ["", "## Matrice di confusione", "", "| Modello | TP | FN | FP | TN |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name, result in (("Phase 14", b), ("Phase 23", c)):
        counts = result["pooled_confusion_counts"]
        lines.append(f"| {name} | {counts['true_positives']} | {counts['false_negatives']} | {counts['false_positives']} | {counts['true_negatives']} |")
    interval = summary["paired_conditional_f1_delta_interval_95"]
    flips = summary["decision_changes"]
    lines += ["", "## Interpretazione e decisione", "",
        f"Delta F1: {summary['metric_deltas']['f1']:+.6f}; fold favorevoli: {summary['positive_fold_count']}/3. Tempo della CV: {summary['runtime_seconds']/60:.1f} minuti.", "",
        f"Variazione netta: {c['pooled_confusion_counts']['true_positives'] - b['pooled_confusion_counts']['true_positives']:+d} veri positivi e {c['pooled_confusion_counts']['false_positives'] - b['pooled_confusion_counts']['false_positives']:+d} falsi positivi. L'aumento di recall non compensa la perdita di precision.", "",
        f"Recuperati {flips['recovered_false_negatives']} falsi negativi e persi {flips['lost_true_positives']} veri positivi; aggiunti {flips['added_false_positives']} falsi positivi ed eliminati {flips['removed_false_positives']}.", "",
        f"AP cambia di {summary['metric_deltas']['average_precision']:+.6f}; log loss cambia di {summary['metric_deltas']['log_loss']:+.6f}. La priorità resta l'F1. Anche l'F1 esplorativo con una soglia globale ottimizzata sugli stessi OOF ({summary['exploratory_same_oof_threshold_f1']:.6f}) resta sotto il controllo calcolato nello stesso modo (0,442078).", "",
        f"Intervallo bootstrap condizionato 95% per delta F1: [{interval[0]:+.6f}, {interval[1]:+.6f}], a modelli e soglie fissi; non include variabilità del training o selezione ripetuta.", "",
        ("Il candidato soddisfa il filtro operativo del piano: F1 aggregato superiore e almeno due fold favorevoli. Il risultato va mantenuto come controllo per isolare il contributo dell'imputazione."
         if summary["meets_plan_promotion_rule"] else
         "Il candidato non soddisfa il filtro operativo del piano. Phase 14 resta il riferimento; il risultato non esclude un beneficio quando agli indicatori si aggiunge l'imputazione."), "",
        "La successiva prova con imputazione ristretta resta da eseguire, anche se gli indicatori soli perdono. Il precedente F1 esplorativo 0,442674 riguardava indicatori e imputazione insieme, con una soglia scelta sugli stessi OOF valutati e due indicatori annullati dal binning: questa prova non è la conferma di quella combinazione.", "",
        "## Frequenze degli indicatori nei training esterni", "",
        "| Indicatore | Minimo % | Massimo % |", "| --- | ---: | ---: |"]
    outer = [p for p in _read(root, PREFLIGHT)["partitions"] if "inner" not in p["partition"]]
    for flag in outer[0]["flags"]:
        rates = [100 * p["flags"][flag]["training_ones"] / p["training_rows"] for p in outer]
        lines.append(f"| {flag} | {min(rates):.3f} | {max(rates):.3f} |")
    lines += ["", "La frequenza descrive gli indicatori e non misura la loro utilità predittiva. Aggiungere otto colonne modifica anche il campionamento delle feature, pur conservando 128 candidate per nodo: il risultato riguarda la pipeline completa.", "",
        "## Artefatti e limiti", "",
        "Prima del training sono passati 104 test, inclusi conservazione degli OOF interni, confini dei fold e identità dei risultati con/senza salvataggio. Dopo il training le tre predizioni dei checkpoint e le tre soglie interne coincidono esattamente con gli artefatti registrati.", "",
        f"Ambiente: Python {summary['environment']['python']}, NumPy {summary['environment']['numpy']}; ambiente ufficiale Python 3.9 / NumPy 1.23.1 ancora da verificare.", "",
        "Il 20% iniziale non è utilizzato nella prova, ma era già stato consultato storicamente. Configurazione finale e soglia di submission restano invariate; nessun nuovo modello sull'intero training o submission.", "",
        f"- [Record completo](../experiments/{Path(summary['candidate_record']).name}).",
        "- [Confronto verificato](../experiments/phase23_missing_reason_flags_comparison.json).",
        "- [Audit prima del training](phase23_missing_reason_flags_preflight.json).",
        "- Predizioni esterne, tre checkpoint e tre artefatti OOF interni in results/boosting_artifacts/phase23_phase14_missing_reason_flags/.", ""]
    (root / "results/eda/phase23_missing_reason_flags_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run", "compare"))
    args = parser.parse_args()
    root = Path.cwd()
    if args.action == "preflight":
        preflight(root)
        return
    directory = root / "results/boosting_artifacts" / PREFIX
    records = list((directory / "records").glob("*.json"))
    if args.action == "run":
        if records:
            raise ValueError("A Phase 23 record already exists; use compare instead of retraining.")
        check = _read(root, PREFLIGHT)
        suite, _ = _configuration(root)
        for path, expected in {**check["source_sha256"], SUITE: check["suite_sha256"],
                suite["preprocessing_config_path"]: check["preprocessing_sha256"],
                "configs/final_model.json": check["final_config_sha256"]}.items():
            if _sha256(root / path) != expected:
                raise ValueError(f"Preflight is stale: {path}")
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "run_stdout.log").open("w", encoding="utf-8") as log:
            import sys
            tee = _Tee(sys.stdout, log)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                records = run_logged_suite(root / SUITE, root, directory / "records")
                if len(records) != 1:
                    raise ValueError("Expected one completed Phase 23 record.")
                compare(root, records[0])
        return
    if len(records) != 1:
        raise ValueError("Expected one completed record for comparison.")
    compare(root, records[0])


if __name__ == "__main__":
    main()

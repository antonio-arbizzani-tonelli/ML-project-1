"""Verify and report the controlled Phase 20 binning experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.evaluation import classification_metrics
from src.feature_preprocessing import load_feature_metadata, tree_feature_matrices
from src.numpy_boosting import HistogramGradientBoostingClassifier
from src.run_preprocessing_cv import _as_binary_labels, _load_feature_names, _sha256, _stratified_folds
from src.summarize_phase19_transfer import BASELINE, _arrays, _paired_interval, _publish


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    baseline = json.loads((root / BASELINE).read_text(encoding="utf-8"))
    record = json.loads(args.record.read_text(encoding="utf-8"))
    preflight = json.loads((root / "results/eda/phase20_binning_preflight.json").read_text(encoding="utf-8"))
    for key in ("features", "labels", "split_indices", "feature_metadata", "preprocessing_config"):
        if record["outcome"]["input_sha256"][key] != baseline["outcome"]["input_sha256"][key]:
            raise ValueError(f"Candidate changed baseline input {key}.")
    for key in ("features", "split"):
        if record["config"][key] != baseline["config"][key]:
            raise ValueError(f"Candidate changed {key}.")
    bparams = baseline["config"]["model"]["parameters"]
    cparams = dict(record["config"]["model"]["parameters"])
    if cparams.pop("binning_strategy", None) != "exact_low_cardinality" or cparams != bparams:
        raise ValueError("Candidate does not isolate the binning policy.")
    for path, expected in record["outcome"]["source_sha256"].items():
        if _sha256(root / path) != expected:
            raise ValueError(f"Source changed while fitting: {path}.")
    if _sha256(root / "configs/final_model.json") != preflight["final_config_sha256"]:
        raise ValueError("The configured final model changed during the experiment.")
    profiles = record["outcome"]["individual_fit_profiles"]
    if len(profiles) != 9:
        raise ValueError("Expected nine completed predictive fits.")
    partitions = {p["partition"]: p for p in preflight["partitions"]}
    for number, profile in enumerate(profiles, 1):
        partition = (f"outer_{number}" if number <= 3
                     else f"outer_{(number - 4) // 2 + 1}_inner_{(number - 4) % 2 + 1}")
        expected = partitions[partition]
        if profile["binning_strategy"] != "exact_low_cardinality":
            raise ValueError("A fit did not use the requested policy.")
        if profile["bin_counts"] != expected["bin_counts"]:
            raise ValueError(f"Actual bin counts differ from preflight: {partition}.")
        if len(profile["exact_column_indices"]) != expected["exact_columns"]:
            raise ValueError(f"Actual exact columns differ: {partition}.")
    barray, carray = _arrays(root, baseline), _arrays(root, record)
    for key in ("development_indices", "fold_ids", "labels"):
        if not np.array_equal(barray[key], carray[key]):
            raise ValueError(f"Candidate OOF alignment differs: {key}.")
    for arrays, rec in ((barray, baseline), (carray, record)):
        if not np.array_equal(arrays["nested_predictions"], arrays["probabilities"] >= arrays["nested_thresholds"]):
            raise ValueError("Nested predictions differ from their thresholds.")
        _, checked = classification_metrics(arrays["labels"], arrays["nested_predictions"])
        for key, value in checked.items():
            saved = rec["outcome"]["nested_threshold_evaluation"]["pooled_metrics"].get(key)
            if saved is not None and not np.isclose(value, saved, rtol=0, atol=1e-15):
                raise ValueError(f"Recomputed nested metric differs: {key}.")
    for checkpoint in record["outcome"]["model_checkpoints"]:
        if _sha256(root / checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError("Candidate checkpoint hash differs.")
    suite = json.loads((root / "configs/experiments/phase20_phase14_exact_low_cardinality.json").read_text(encoding="utf-8"))
    preprocessing = json.loads((root / suite["preprocessing_config_path"]).read_text(encoding="utf-8"))
    plan = next(v["preprocessing"] for v in preprocessing["variants"] if v["experiment_name"] == suite["preprocessing_variant"])
    x = np.load(root / suite["data"]["features_path"], mmap_mode="r")
    y = _as_binary_labels(np.load(root / suite["data"]["labels_path"]))
    names = _load_feature_names(root / suite["data"]["x_train_csv_path"])
    metadata = load_feature_metadata(root / suite["feature_metadata_path"])
    development = carray["development_indices"]
    prediction_checks = []
    folds = _stratified_folds(y[development], 3, suite["seed"])
    for fold, (checkpoint, reference, (train, valid)) in enumerate(zip(
            record["outcome"]["model_checkpoints"], baseline["outcome"]["model_checkpoints"], folds), 1):
        model, checkpoint_metadata = HistogramGradientBoostingClassifier.load_checkpoint(root / checkpoint["path"])
        if _sha256(root / reference["path"]) != reference["sha256"]:
            raise ValueError("Reference checkpoint hash differs.")
        reference_model, reference_metadata = HistogramGradientBoostingClassifier.load_checkpoint(root / reference["path"])
        for key in ("training_row_indices_sha256", "training_labels_sha256"):
            if checkpoint_metadata[key] != reference_metadata[key]:
                raise ValueError(f"Checkpoint training inputs differ: {key}.")
        del reference_model
        clean_train, clean_valid, _ = tree_feature_matrices(np.asarray(x[development[train]], dtype=np.float32),
            np.asarray(x[development[valid]], dtype=np.float32), names, metadata, plan)
        if model.binning_strategy != "exact_low_cardinality":
            raise ValueError("Checkpoint lacks the new binning policy.")
        difference = float(np.max(np.abs(model.predict_proba(clean_valid)[:, 1] - carray["probabilities"][valid])))
        if difference != 0.0:
            raise ValueError("Checkpoint predictions differ from the OOF artifact.")
        prediction_checks.append({"fold": fold, "maximum_probability_difference": difference})
        print(f"Verified checkpoint fold={fold}: probability difference={difference}", flush=True)
        del model, clean_train, clean_valid
    bnest = baseline["outcome"]["nested_threshold_evaluation"]
    cnest = record["outcome"]["nested_threshold_evaluation"]
    if len(cnest["folds"]) != 3:
        raise ValueError("Candidate lacks three completed outer folds.")
    deltas = {key: value - bnest["pooled_metrics"][key] for key, value in cnest["pooled_metrics"].items()}
    fold_deltas = [c["outer_evaluation_metrics"]["f1"] - b["outer_evaluation_metrics"]["f1"]
                   for b, c in zip(bnest["folds"], cnest["folds"])]
    labels, bp, cp = carray["labels"], barray["nested_predictions"], carray["nested_predictions"]
    flips = {
        "recovered_false_negatives": int(np.sum((labels == 1) & (bp == 0) & (cp == 1))),
        "lost_true_positives": int(np.sum((labels == 1) & (bp == 1) & (cp == 0))),
        "added_false_positives": int(np.sum((labels == 0) & (bp == 0) & (cp == 1))),
        "removed_false_positives": int(np.sum((labels == 0) & (bp == 1) & (cp == 0))),
    }
    promote = deltas["f1"] > 0 and sum(d > 0 for d in fold_deltas) >= 2
    published = _publish(args.record, root)
    summary = {
        "baseline_record": BASELINE, "candidate_record": str(published.relative_to(root)).replace("\\", "/"),
        "baseline_nested": bnest, "candidate_nested": cnest, "metric_deltas": deltas,
        "fold_f1_deltas": fold_deltas, "positive_fold_count": sum(d > 0 for d in fold_deltas),
        "meets_plan_promotion_rule": promote, "decision_changes": flips,
        "paired_conditional_f1_delta_interval_95": _paired_interval(labels, bp, cp),
        "bootstrap_seed": 20260929, "bootstrap_repetitions": 20000,
        "bootstrap_scope": "Paired stratified rows, fixed fitted models and thresholds; excludes training and model-selection uncertainty.",
        "probability_correlation": float(np.corrcoef(barray["probabilities"], carray["probabilities"])[0, 1]),
        "mean_absolute_probability_difference": float(np.mean(np.abs(barray["probabilities"] - carray["probabilities"]))),
        "runtime_seconds": record["outcome"]["runtime_seconds"], "individual_fit_profiles": profiles,
        "checkpoint_prediction_checks": prediction_checks,
        "exploratory_same_oof_threshold_f1": {
            "baseline": baseline["outcome"]["metrics"]["f1"],
            "candidate": record["outcome"]["metrics"]["f1"],
            "scope": "Optimistic diagnostic: threshold chosen on the labels being scored; not used for promotion.",
        },
        "environment": record["outcome"]["environment"], "final_model_changed": False,
    }
    out = root / "results/experiments/phase20_binning_comparison.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# Phase 20 — binning esatto delle colonne con poche modalità", "",
             "## Protocollo e controlli", "",
             "Confronto isolato con Phase 14: stesse 295 feature, stessa pulizia dei codici, stessi parametri del booster e stessi split. L'unica opzione aggiunta è `binning_strategy=exact_low_cardinality`. Fino a 64 valori finiti distinti nel training si conservano tutte le modalità; oltre 64 si riusa il binning a quantili. I NaN restano nel bin 255.", "",
             "Per le colonne esatte si valuta anche la divisione fra tutti gli osservati e tutti i mancanti: senza questo confine, eliminare i bin vuoti potrebbe ridurre la capacità precedente del modello. Le colonne che restano a quantili mantengono la ricerca dei confini originale.", "",
             "Validazione sui 262.508 esempi development: tre fold esterni e due interni per scegliere ciascuna soglia. Nove fit completi da 200 alberi; Phase 14 riutilizzata senza riaddestramento. I test unitari addestrano piccoli modelli sintetici. Il primo avvio della CV, interrotto per un problema di logging prima del completamento del primo fit, non contribuisce ai risultati e al runtime del record.", "",
             "95 test passati prima del training. Il preflight sui nove training conferma tutte le modalità esatte, il fallback a quantili identico e lo stato dei mancanti invariato. I tre checkpoint Phase 14 ricalcolano tutte le predizioni salvate con differenza massima zero. I profili dei nove training reali coincidono con i conteggi dei bin del preflight.", "",
             "Hash degli input, righe, etichette e fold OOF verificati; predizioni annidate e metriche ricalcolate. Anche le predizioni dei tre checkpoint candidati sono ricalcolate, con differenza massima zero dagli OOF. Gli hash delle righe e delle etichette usate per il training coincidono con Phase 14. Il 20% iniziale non è utilizzato in questa prova; era già stato consultato storicamente.", "",
             "## Metriche annidate aggregate", "",
             "| Modello | F1 | Precision | Recall | AP | Log loss |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, nested in (("Phase 14", bnest), ("Binning esatto", cnest)):
        m = nested["pooled_metrics"]
        lines.append(f"| {name} | {m['f1']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['average_precision']:.6f} | {m['log_loss']:.6f} |")
    lines += ["", f"Differenza F1 assoluta: **{deltas['f1']:+.6f}**. Fold favorevoli: **{sum(d > 0 for d in fold_deltas)}/3**.", "",
              "## Risultati per fold", "", "| Fold | F1 Phase 14 | F1 candidato | Differenza | Soglia Phase 14 | Soglia candidato |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for number, (b, c, delta) in enumerate(zip(bnest["folds"], cnest["folds"], fold_deltas), 1):
        lines.append(f"| {number} | {b['outer_evaluation_metrics']['f1']:.6f} | {c['outer_evaluation_metrics']['f1']:.6f} | {delta:+.6f} | {b['inner_threshold']:.6f} | {c['inner_threshold']:.6f} |")
    lines += ["", "## Matrici di confusione", "", "| Modello | TP | FN | FP | TN |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, nested in (("Phase 14", bnest), ("Binning esatto", cnest)):
        c = nested["pooled_confusion_counts"]
        lines.append(f"| {name} | {c['true_positives']} | {c['false_negatives']} | {c['false_positives']} | {c['true_negatives']} |")
    interval = summary["paired_conditional_f1_delta_interval_95"]
    lines += ["", "## Cambiamenti nelle decisioni", "",
              f"Il candidato recupera {flips['recovered_false_negatives']} falsi negativi e perde {flips['lost_true_positives']} veri positivi. Introduce {flips['added_false_positives']} nuovi falsi positivi e ne elimina {flips['removed_false_positives']}.", "",
              f"Intervallo bootstrap appaiato condizionato al 95% della differenza F1: [{interval[0]:+.6f}, {interval[1]:+.6f}]. Usa 20.000 ricampionamenti stratificati per classe e seed 20260929, a modelli e soglie fissati. Non comprende la variabilità del training o la selezione ripetuta dei candidati: è un indicatore diagnostico, non una conferma indipendente.", "",
              "## Costo e ambiente", "",
              f"Runtime della CV completata: {summary['runtime_seconds']:.1f} secondi ({summary['runtime_seconds']/60:.1f} minuti). Ambiente effettivo: Python {summary['environment']['python']}, NumPy {summary['environment']['numpy']}. Non è stato eseguito il controllo nell'ambiente ufficiale Python 3.9 / NumPy 1.23.1.", "",
              "Il candidato usa 250, 251 e 250 colonne esatte nei tre training esterni, includendo le colonne costanti. Le colonne che recuperano distinzioni sono 79, 82 e 79. Nei training interni si usano 251 colonne esatte. La matrice resta uint8; il numero maggiore di confini può aumentare il costo della ricerca degli split.", "",
              "## Decisione", "",
              ("Il candidato soddisfa la regola operativa del piano: F1 aggregato maggiore e almeno due fold favorevoli. È il candidato raccomandato per il seguito, con conferma della compatibilità ufficiale e valutazione delle successive trasformazioni sul controllo corrispondente."
               if promote else "Il candidato non soddisfa la regola operativa del piano. Conservare Phase 14 come riferimento; questo confronto non supporta l'adozione generale del binning esatto."), "",
              f"La variazione AP è {deltas['average_precision']:+.6f} e quella della log loss è {deltas['log_loss']:+.6f}: sono differenze molto piccole. La correlazione delle probabilità è {summary['probability_correlation']:.6f}. Anche scegliendo esplorativamente la soglia sulle stesse etichette OOF, l'F1 del candidato è {record['outcome']['metrics']['f1']:.6f}, contro {baseline['outcome']['metrics']['f1']:.6f}. Questi ultimi numeri sono ottimistici e non entrano nella decisione.", "",
              "La perdita di modalità è reale, ma aumentare la risoluzione di tutte le colonne coinvolte non migliora l'F1 in questa prova. Il risultato non dimostra che la causa sia l'overfitting, né esclude benefici su una famiglia specifica. Un eventuale seguito deve limitare l'esatta conservazione a binarie e categorie con poche modalità, mantenendo il binning numerico originale; nei futuri one-hot e indicatori va assicurata la conservazione dei loro due stati. L'intervallo condizionato include zero: non presentare la piccola differenza come una prova definitiva di inferiorità del metodo in generale.", "",
              "La configurazione di submission resta invariata: questa richiesta implementa e valuta il candidato. Non è stato addestrato un nuovo modello sull'intero training né generata una submission.", "",
              "## Artefatti", "",
              f"- [Record candidato](../experiments/{published.name}).",
              "- [Confronto completo](../experiments/phase20_binning_comparison.json).",
              "- [Preflight](phase20_binning_preflight.json).",
              "- [Audit iniziale](discrete_binning_audit.md).",
              "- [Configurazione](../../configs/experiments/phase20_phase14_exact_low_cardinality.json)."]
    (root / "results/eda/phase20_binning_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"f1": cnest["pooled_metrics"]["f1"], "delta_f1": deltas["f1"],
                      "fold_deltas": fold_deltas, "meets_plan_rule": promote, "record": str(published)}, indent=2))


if __name__ == "__main__":
    main()

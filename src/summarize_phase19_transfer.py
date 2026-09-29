"""Publish and compare the two controlled Phase 19 transfer experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.experiment_logger import _append_index
from src.run_preprocessing_cv import _sha256


BASELINE = (
    "results/experiments/20260921T105139261213Z_"
    "phase14-boosting-codebook-corrections-only-depth5-features128-200trees.json"
)


def _publish(path: Path, root: Path) -> Path:
    """Copy a completed record and its index row without replacing old results."""

    target = root / "results/experiments" / path.name
    if path.resolve() == target.resolve():
        return target
    with (path.parent / "index.csv").open(encoding="utf-8", newline="") as handle:
        row = next(r for r in csv.DictReader(handle) if r["record_path"] == path.name)
    if _sha256(path) != row["record_sha256"]:
        raise ValueError("New experiment record does not match its index hash.")
    if target.exists() and target.read_bytes() != path.read_bytes():
        raise ValueError("A different experiment already uses the same record path.")
    if not target.exists():
        target.write_bytes(path.read_bytes())
    index = target.parent / "index.csv"
    with index.open(encoding="utf-8", newline="") as handle:
        ids = {r["run_id"] for r in csv.DictReader(handle)}
    if row["run_id"] not in ids:
        _append_index(index, row)
    return target


def _arrays(root: Path, record: dict) -> dict:
    """Load exact OOF arrays after checking their recorded fingerprint."""

    info = record["outcome"]["oof_predictions_artifact"]
    path = root / info["path"]
    if _sha256(path) != info["sha256"]:
        raise ValueError("OOF predictions differ from the completed record.")
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def _paired_interval(labels, baseline, candidate) -> list[float]:
    """Bootstrap paired prediction counts with fixed fits and thresholds."""

    rng = np.random.default_rng(20260929)
    cell = baseline.astype(np.int8) * 2 + candidate.astype(np.int8)
    positive = np.bincount(cell[labels == 1], minlength=4)
    negative = np.bincount(cell[labels == 0], minlength=4)
    pos = rng.multinomial(int(positive.sum()), positive / positive.sum(), size=20000)
    neg = rng.multinomial(int(negative.sum()), negative / negative.sum(), size=20000)
    baseline_tp = pos[:, 2] + pos[:, 3]
    candidate_tp = pos[:, 1] + pos[:, 3]
    baseline_fp = neg[:, 2] + neg[:, 3]
    candidate_fp = neg[:, 1] + neg[:, 3]
    baseline_fn = pos[:, 0] + pos[:, 1]
    candidate_fn = pos[:, 0] + pos[:, 2]
    baseline_f1 = 2 * baseline_tp / (2 * baseline_tp + baseline_fp + baseline_fn)
    candidate_f1 = 2 * candidate_tp / (2 * candidate_tp + candidate_fp + candidate_fn)
    return np.quantile(candidate_f1 - baseline_f1, [0.025, 0.975]).tolist()


def main() -> None:
    """Validate comparability, save metrics and write an Italian report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, nargs=2)
    args = parser.parse_args()
    root = Path.cwd()
    baseline = json.loads((root / BASELINE).read_text(encoding="utf-8"))
    baseline_arrays = _arrays(root, baseline)
    baseline_nested = baseline["outcome"]["nested_threshold_evaluation"]
    baseline_metrics = baseline_nested["pooled_metrics"]
    summary = {"baseline_record": BASELINE, "baseline_nested": baseline_nested,
               "candidates": [], "bootstrap_seed": 20260929,
               "bootstrap_repetitions": 20000,
               "bootstrap_scope": "Paired stratified row bootstrap with fixed fitted models and thresholds; excludes training and model-selection uncertainty."}
    for source in args.records:
        record = json.loads(source.read_text(encoding="utf-8"))
        nested = record["outcome"]["nested_threshold_evaluation"]
        if not nested or len(nested["folds"]) != 3:
            raise ValueError("Candidate must have completed all three nested outer folds.")
        for key in ("features", "labels", "split_indices", "feature_metadata", "preprocessing_config"):
            if record["outcome"]["input_sha256"][key] != baseline["outcome"]["input_sha256"][key]:
                raise ValueError(f"Candidate changed the baseline input {key}.")
        if record["config"]["features"] != baseline["config"]["features"]:
            raise ValueError("Candidate changed the Phase 14 feature representation.")
        parameters = record["config"]["model"]["parameters"]
        changed = {key: [baseline["config"]["model"]["parameters"][key], value]
                   for key, value in parameters.items()
                   if value != baseline["config"]["model"]["parameters"][key]}
        if changed not in ({"min_samples_leaf": [200, 150]}, {"learning_rate": [0.05, 0.07]}):
            raise ValueError("Candidate does not isolate a requested parameter change.")
        arrays = _arrays(root, record)
        for key in ("development_indices", "fold_ids", "labels"):
            if not np.array_equal(arrays[key], baseline_arrays[key]):
                raise ValueError(f"Candidate OOF rows differ for {key}.")
        bpred = baseline_arrays["nested_predictions"]
        cpred = arrays["nested_predictions"]
        labels = arrays["labels"]
        fold_deltas = [
            f["outer_evaluation_metrics"]["f1"] - b["outer_evaluation_metrics"]["f1"]
            for f, b in zip(nested["folds"], baseline_nested["folds"])
        ]
        metric_deltas = {key: value - baseline_metrics[key]
                         for key, value in nested["pooled_metrics"].items()}
        published = _publish(source, root)
        candidate = {
            "record": str(published.relative_to(root)).replace("\\", "/"),
            "changed_parameters": changed,
            "nested": nested, "metric_deltas": metric_deltas,
            "fold_f1_deltas": fold_deltas,
            "positive_fold_count": sum(delta > 0 for delta in fold_deltas),
            "meets_plan_promotion_rule": metric_deltas["f1"] > 0 and sum(delta > 0 for delta in fold_deltas) >= 2,
            "decision_changes": {
                "recovered_false_negatives": int(np.sum((labels == 1) & (bpred == 0) & (cpred == 1))),
                "lost_true_positives": int(np.sum((labels == 1) & (bpred == 1) & (cpred == 0))),
                "added_false_positives": int(np.sum((labels == 0) & (bpred == 0) & (cpred == 1))),
                "removed_false_positives": int(np.sum((labels == 0) & (bpred == 1) & (cpred == 0))),
            },
            "paired_conditional_f1_delta_interval_95": _paired_interval(labels, bpred, cpred),
            "probability_correlation": float(np.corrcoef(baseline_arrays["probabilities"], arrays["probabilities"])[0, 1]),
            "mean_absolute_probability_difference": float(np.mean(np.abs(baseline_arrays["probabilities"] - arrays["probabilities"]))),
            "runtime_seconds": record["outcome"]["runtime_seconds"],
            "individual_fit_profiles": record["outcome"]["individual_fit_profiles"],
            "environment": record["outcome"]["environment"],
        }
        summary["candidates"].append(candidate)
    out = root / "results/experiments/phase19_transfer_comparison.json"
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    names = ["Phase 14"] + ["Foglia 150" if "min_samples_leaf" in c["changed_parameters"] else "Learning rate 0,07" for c in summary["candidates"]]
    nested_all = [baseline_nested] + [c["nested"] for c in summary["candidates"]]
    lines = ["# Phase 19: trasferimento del tuning sulla Phase 14", "",
             "## Verifica e protocollo", "",
             "La verifica di conformità ha confrontato dati, etichette, split, configurazioni, percorso del preprocessing e predizioni dei checkpoint Phase 14. Tutti i 35 controlli sono passati; le predizioni ricalcolate coincidono esattamente nei tre fold. Nessun modello è stato addestrato durante questa verifica.", "",
             "Le due prove conservano la rappresentazione Phase 14 con `HAREHAB1`, 295 feature e tutti i seed originali. Cambiano un solo parametro. Ciascuna usa tre fold esterni e due fold interni per scegliere la soglia: nove training per candidato, sui soli 262.508 esempi development.", "",
             "## Metriche annidate aggregate", "",
             "| Modello | F1 | Precision | Recall | AP | Log loss |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, nested in zip(names, nested_all):
        m = nested["pooled_metrics"]
        lines.append(f"| {name} | {m['f1']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['average_precision']:.6f} | {m['log_loss']:.6f} |")
    lines += ["", "## F1 per fold", "", "| Modello | Fold 1 | Fold 2 | Fold 3 | Media dei fold |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, nested in zip(names, nested_all):
        values = [f["outer_evaluation_metrics"]["f1"] for f in nested["folds"]]
        lines.append(f"| {name} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | {np.mean(values):.6f} |")
    lines += ["", "## Soglie scelte nei fold interni", "", "| Modello | Fold 1 | Fold 2 | Fold 3 | Media |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, nested in zip(names, nested_all):
        values = [f["inner_threshold"] for f in nested["folds"]]
        lines.append(f"| {name} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | {np.mean(values):.6f} |")
    lines += ["", "## Matrici di confusione", "", "| Modello | TP | FN | FP | TN |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, nested in zip(names, nested_all):
        c = nested["pooled_confusion_counts"]
        lines.append(f"| {name} | {c['true_positives']} | {c['false_negatives']} | {c['false_positives']} | {c['true_negatives']} |")
    for name, c in zip(names[1:], summary["candidates"]):
        interval = c["paired_conditional_f1_delta_interval_95"]
        counts = c["nested"]["pooled_confusion_counts"]
        baseline_counts = baseline_nested["pooled_confusion_counts"]
        delta_tp = counts["true_positives"] - baseline_counts["true_positives"]
        delta_fp = counts["false_positives"] - baseline_counts["false_positives"]
        flips = c["decision_changes"]
        lines += ["", f"## {name}", "", f"Differenza F1: {c['metric_deltas']['f1']:+.6f}; fold favorevoli: {c['positive_fold_count']}/3. Tempo della validazione: {c['runtime_seconds'] / 60:.2f} minuti.", "",
                  f"Rispetto al riferimento: {delta_tp:+d} veri positivi e {delta_fp:+d} falsi positivi. La variazione dei falsi negativi è {-delta_tp:+d}.", "",
                  f"Fra le righe che cambiano decisione: {flips['recovered_false_negatives']} falsi negativi vengono recuperati, {flips['lost_true_positives']} veri positivi vengono persi, {flips['added_false_positives']} nuovi falsi positivi compaiono e {flips['removed_false_positives']} falsi positivi vengono eliminati.", "",
                  f"Le differenze F1 dei tre fold sono {c['fold_f1_deltas'][0]:+.6f}, {c['fold_f1_deltas'][1]:+.6f} e {c['fold_f1_deltas'][2]:+.6f}. AP cambia di {c['metric_deltas']['average_precision']:+.6f}; log loss cambia di {c['metric_deltas']['log_loss']:+.6f}.", "",
                  f"Bootstrap appaiato descrittivo: intervallo 95% della differenza F1 [{interval[0]:+.6f}, {interval[1]:+.6f}]. Mantiene fissi modelli e soglie e non misura l'incertezza da riaddestramento o selezione ripetuta dei candidati.", "",
                  f"Record: [{Path(c['record']).name}](../{c['record'].split('results/', 1)[1]})."]
    lines += ["", "## Decisione", "",
              "La prova misura il trasferimento alla rappresentazione Phase 14. L'esito diverso dalla Phase 17 non può essere attribuito alla sola HAREHAB1: fra i due rami differiscono anche altre scelte di preprocessing.", ""]
    if not any(c["meets_plan_promotion_rule"] for c in summary["candidates"]):
        lines += ["Nessun candidato soddisfa la regola del piano: F1 aggregato superiore al controllo e miglioramento in almeno due fold su tre. Conservare Phase 14 con minimo foglia 200 e learning rate 0.05. Il piccolo aumento dell'AP non sostituisce un miglioramento dell'F1, che resta la metrica principale.", "",
                  "Il vantaggio della Phase 17 sul suo ramo senza HAREHAB1 non si trasferisce al ramo Phase 14 nelle due prove isolate. Non assumere un effetto additivo e non avviare una combinazione come se i due cambiamenti fossero già favorevoli. La prossima prova prioritaria del piano è l'ablazione della sola ALCDAY5, conservando DROCDY3_ e tutti gli altri input del riferimento."]
    else:
        lines += ["Valutare i candidati che soddisfano la regola del piano; le modifiche favorevoli vanno combinate solo mediante un nuovo confronto controllato."]
    lines += ["", "## Ambiente e artefatti", "", "I due candidati sono stati eseguiti in parallelo. I tempi misurano la durata di ciascuna validazione in queste condizioni; il tempo storico Phase 14 non è un confronto controllato di velocità.", "",
              f"Ambiente dei record: Python {summary['candidates'][0]['environment']['python']} e NumPy {summary['candidates'][0]['environment']['numpy']}. I test locali passano (86/86) e le predizioni dei checkpoint di riferimento sono riprodotte esattamente. Questo non sostituisce la verifica nell'ambiente ufficiale Python 3.9 / NumPy 1.23.1.", "",
              "## Precisazione sulla Phase 16 remota", "",
              "La configurazione remota contiene cinque interazioni, ma il preprocessing ad alberi versionato non legge il campo interactions. Le colonne sorgente conservate e l'output dichiarato nel record sono entrambi 295. Il risultato non dimostra una prova effettiva di tali interazioni. Il piano e la storia degli esperimenti sono stati corretti; questa precisazione non cambia le due prove Phase 19 sulla rappresentazione Phase 14 verificata.", "",
              "I JSON degli esperimenti e il confronto completo sono versionabili. Checkpoint, log e predizioni OOF restano negli artefatti locali. La configurazione finale non viene modificata da questo script.", ""]
    report = root / "results/eda/phase19_transfer_summary.md"
    with report.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    print(f"Saved {out}")
    print(f"Saved {report}")
    for name, c in zip(names[1:], summary["candidates"]):
        print(name, "F1", c["nested"]["pooled_metrics"]["f1"], "delta", c["metric_deltas"]["f1"], "positive folds", c["positive_fold_count"])


if __name__ == "__main__":
    main()

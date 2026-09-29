"""Collect Phase 16 OOF predictions and compare available F1 estimates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from src.evaluation import (
    average_precision,
    best_f1_threshold,
    binary_log_loss,
    classification_metrics,
    labels_from_scores,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results/experiments"
PREDICTION_PATHS = {
    "linear_restricted": ROOT / "results/linear_imputation_artifacts/phase16_imputation_restricted/oof_predictions.npz",
    "linear_expanded": ROOT / "results/linear_imputation_artifacts/phase16_imputation_expanded/oof_predictions.npz",
    "tree_restricted": ROOT / "results/boosting_artifacts/phase16_imputation_restricted/oof_predictions/phase16_imputation_restricted_depth5_features128_200trees_oof.npz",
    "tree_expanded": ROOT / "results/boosting_artifacts/phase16_imputation_expanded/oof_predictions/phase16_imputation_expanded_depth5_features128_200trees_oof.npz",
    "linear_original": ROOT / "results/oof_predictions/20260920T113608778965Z_phase7-p2-codebook-clean-bphigh4-corrected-interactions-gamma-0p2-nested-4000_oof.npz",
    "tree_original": ROOT / "results/boosting_artifacts/phase14_codebook_corrections_only/oof_predictions/phase14_boosting_codebook_corrections_only_depth5_features128_200trees_oof.npz",
}
OPTIMISTIC_PREDICTION_PATHS = {
    "linear_restricted": ROOT / "results/linear_imputation_artifacts/phase16_imputation_restricted_optimistic/oof_predictions.npz",
    "linear_expanded": ROOT / "results/linear_imputation_artifacts/phase16_imputation_expanded_optimistic/oof_predictions.npz",
    "tree_restricted": ROOT / "results/boosting_artifacts/phase16_imputation_restricted_optimistic/oof_predictions/phase16_imputation_restricted_optimistic_depth5_features128_200trees_oof.npz",
    "tree_expanded": ROOT / "results/boosting_artifacts/phase16_imputation_expanded_optimistic/oof_predictions/phase16_imputation_expanded_optimistic_depth5_features128_200trees_oof.npz",
    "linear_original": PREDICTION_PATHS["linear_original"],
    "tree_original": PREDICTION_PATHS["tree_original"],
}
LINEAR_ORIGINAL_RECORD = ROOT / "results/experiments/20260920T113608936798Z_phase7-p2-codebook-clean-bphigh4-corrected-interactions-gamma-0p2-nested-4000.json"


def _metrics(labels: np.ndarray, probabilities: np.ndarray, decisions: np.ndarray) -> dict[str, float | int]:
    counts, values = classification_metrics(labels, decisions)
    values.update({
        "average_precision": average_precision(labels, probabilities),
        "log_loss": binary_log_loss(labels, probabilities),
        "true_positive": counts.true_positives,
        "false_positive": counts.false_positives,
        "true_negative": counts.true_negatives,
        "false_negative": counts.false_negatives,
    })
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimistic-only", action="store_true", help="Summarize outer OOF runs without nested threshold selection.")
    args = parser.parse_args()
    prediction_paths = OPTIMISTIC_PREDICTION_PATHS if args.optimistic_only else PREDICTION_PATHS
    missing = [str(path) for path in prediction_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Prediction artifacts are missing: " + ", ".join(missing))
    reference: dict[str, np.ndarray] | None = None
    arrays: dict[str, np.ndarray] = {}
    reports = {}
    original_record = json.loads(LINEAR_ORIGINAL_RECORD.read_text(encoding="utf-8"))
    original_thresholds = {
        int(fold["outer_fold"]): float(fold["inner_threshold"])
        for fold in original_record["outcome"]["nested_threshold_evaluation"]["folds"]
    }
    for name, path in prediction_paths.items():
        with np.load(path, allow_pickle=False) as artifact:
            indices = np.asarray(artifact["development_indices"], dtype=np.int64)
            fold_ids = np.asarray(artifact["fold_ids"], dtype=np.int8)
            labels = np.asarray(artifact["labels"], dtype=np.int8)
            probabilities = np.asarray(artifact["probabilities"], dtype=np.float64)
            if reference is None:
                reference = {"development_indices": indices, "fold_ids": fold_ids, "labels": labels}
                arrays.update(reference)
            else:
                for key, current in (("development_indices", indices), ("fold_ids", fold_ids), ("labels", labels)):
                    if not np.array_equal(current, reference[key]):
                        raise ValueError(f"{name} differs from the reference on {key}.")
            if not np.all(np.isfinite(probabilities)):
                raise ValueError(f"{name} has nonfinite probabilities.")
            choice = best_f1_threshold(labels, probabilities)
            best_predictions = labels_from_scores(probabilities, choice.threshold)
            arrays[f"{name}_probabilities"] = probabilities
            arrays[f"{name}_fixed_0p5_predictions"] = labels_from_scores(probabilities, 0.5)
            arrays[f"{name}_same_oof_best_predictions"] = best_predictions
            arrays[f"{name}_same_oof_best_threshold"] = np.asarray([choice.threshold])
            reports[name] = {
                "source_artifact": str(path.relative_to(ROOT)),
                "fixed_threshold_0p5_metrics": _metrics(
                    labels, probabilities, labels_from_scores(probabilities, 0.5)
                ),
                "same_oof_best_threshold": choice.threshold,
                "same_oof_best_threshold_metrics_optimistic": _metrics(labels, probabilities, best_predictions),
                "per_fold_at_same_oof_best_threshold_optimistic": {
                    str(fold): _metrics(
                        labels[fold_ids == fold],
                        probabilities[fold_ids == fold],
                        best_predictions[fold_ids == fold],
                    )
                    for fold in (1, 2, 3)
                },
            }
            if name == "linear_original":
                nested_thresholds = np.asarray([original_thresholds[int(fold)] for fold in fold_ids])
            elif "nested_thresholds" in artifact:
                nested_thresholds = np.asarray(artifact["nested_thresholds"], dtype=np.float64)
            else:
                nested_thresholds = None
            if nested_thresholds is not None:
                nested_predictions = (probabilities >= nested_thresholds).astype(np.int8)
                if "nested_predictions" in artifact and not np.array_equal(nested_predictions, artifact["nested_predictions"]):
                    raise ValueError(f"{name} has inconsistent saved nested decisions.")
                arrays[f"{name}_nested_thresholds"] = nested_thresholds
                arrays[f"{name}_nested_predictions"] = nested_predictions
                reports[name]["nested_threshold_metrics"] = _metrics(labels, probabilities, nested_predictions)
                reports[name]["per_fold_nested_f1"] = {
                    str(fold): _metrics(labels[fold_ids == fold], probabilities[fold_ids == fold], nested_predictions[fold_ids == fold])["f1"]
                    for fold in (1, 2, 3)
                }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = "phase16_optimistic" if args.optimistic_only else "phase16_retrained"
    predictions_path = OUTPUT_DIR / f"{prefix}_all_oof_predictions.npz"
    temporary_path = OUTPUT_DIR / f"{prefix}_all_oof_predictions.tmp.npz"
    np.savez_compressed(temporary_path, **arrays)
    os.replace(temporary_path, predictions_path)
    report = {
        "protocol": (
            "Three outer OOF folds; each variant's feature imputers and disease classifier were retrained within each outer training fold. A single F1 threshold was then selected on the same pooled outer OOF predictions and labels."
            if args.optimistic_only else
            "Three outer OOF folds; each variant's disease classifier and feature imputers retrained inside every outer fold. Nested thresholds were chosen using two inner OOF folds within the corresponding outer training rows."
        ),
        "row_count": int(reference["labels"].size),
        "all_predictions_artifact": str(predictions_path.relative_to(ROOT)),
        "models": reports,
        "baseline_caveat": "The historical linear_original artifact used median fills for categorical missing values. The retrained Phase 16 logistic runs use the current mode-fill code, so comparison with that historical linear baseline also reflects the fill-policy change.",
        "interpretation": (
            "The reported best-threshold F1 uses the same outer OOF labels to select and evaluate the threshold, so it is optimistic."
            if args.optimistic_only else
            "Nested-threshold F1 is the primary validation estimate. The best threshold selected on the same outer OOF labels is optimistic and descriptive only."
        ),
    }
    report_path = OUTPUT_DIR / "phase16_retrained_comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {predictions_path}")
    print(f"Saved {report_path}")
    for name, item in reports.items():
        nested = item.get("nested_threshold_metrics")
        nested_text = f"nested F1={nested['f1']:.6f} " if nested else ""
        print(f"{name:20s} {nested_text}same-OOF best F1={item['same_oof_best_threshold_metrics_optimistic']['f1']:.6f} threshold={item['same_oof_best_threshold']:.6f}", flush=True)


if __name__ == "__main__":
    main()

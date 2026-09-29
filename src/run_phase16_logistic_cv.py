"""Retrain logistic regression with Phase 16 fold-fitted imputations.

The default uses inner folds to select realistic outer-fold thresholds.
--optimistic-only skips inner fits and selects the best threshold on the same
pooled outer OOF labels being scored. Probabilities and fold weights are saved.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation import (
    average_precision,
    best_f1_threshold,
    binary_log_loss,
    classification_metrics,
    labels_from_scores,
)
from src.feature_preprocessing import FeaturePreprocessor, load_feature_metadata, tree_feature_matrices
from src.run_model_cv import _add_interactions, _fit_checkpoints
from src.run_preprocessing_cv import _as_binary_labels, _sigmoid, _stratified_folds


ROOT = Path(__file__).resolve().parents[1]
VARIANT_CONFIG = ROOT / "configs/experiments/phase16_selected_imputation_preprocessing.json"
BASE_TREE_CONFIG = ROOT / "configs/experiments/phase14_codebook_corrections_only.json"
PHASE7_RECORD = ROOT / "results/experiments/20260920T113608936798Z_phase7-p2-codebook-clean-bphigh4-corrected-interactions-gamma-0p2-nested-4000.json"
FEATURES = ROOT / "data/processed/x_train_float32.npy"
LABELS = ROOT / "data/processed/y_train_int8.npy"
SPLITS = ROOT / "results/eda/analysis/splits/stratified_seed_20260918.npz"
HEADER = ROOT / "data/raw/dataset/x_train.csv"
METADATA = ROOT / "configs/feature_metadata.json"
ARTIFACT_ROOT = ROOT / "results/linear_imputation_artifacts"
INTERACTIONS = [["_AGE80", "_AGE80"], ["_AGE80", "BPHIGH4"], ["_AGE80", "DIABETE3"], ["_AGE80", "SMOKE100"]]
MODEL = {"name": "logistic_gamma_0p2_4000", "gamma": 0.2, "lambda": 0.0, "checkpoints": [4000]}
SEED = 20260919
INNER_SEED = 20260922


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _target_names(plan: dict[str, Any]) -> list[str]:
    settings = plan["imputation"]
    names = list(settings["nonresponse_features"])
    names.extend(settings["continuous_blank_features"])
    if settings["derive_bmi_from_height_weight"]:
        names.append("_BMI5")
    return names


def _replace_source_values(
    design: np.ndarray,
    original_tree: np.ndarray,
    imputed_tree: np.ndarray,
    tree_names: list[str],
    target_names: list[str],
    processor: FeaturePreprocessor,
) -> np.ndarray:
    result = design.copy()
    tree_positions = {name: index for index, name in enumerate(tree_names)}
    output_positions = {name: index + 1 for index, name in enumerate(processor.output_feature_names)}
    states = {state.name: state for state in processor._states or []}
    for name in target_names:
        if name not in tree_positions or name not in output_positions:
            raise ValueError(f"Target {name} is not represented by both preprocessors.")
        state = states[name]
        if state.one_hot_categories or state.mean is None or state.scale is None:
            raise ValueError(f"Unexpected nonnumeric target representation: {name}")
        tree_col = tree_positions[name]
        changed = ~np.isfinite(original_tree[:, tree_col]) & np.isfinite(imputed_tree[:, tree_col])
        values = imputed_tree[changed, tree_col].astype(np.float64)
        if state.binary_map and not set(np.unique(values)).issubset(set(state.binary_map.values())):
            raise ValueError(f"Binary mapping mismatch for {name}.")
        result[changed, output_positions[name]] = ((values - state.mean) / state.scale).astype(np.float32)
    return result


def _designs(
    train_raw: np.ndarray,
    validation_raw: np.ndarray,
    feature_names: list[str],
    metadata: list[dict[str, Any]],
    variant_plan: dict[str, Any],
    base_tree_plan: dict[str, Any],
    linear_plan: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    original_train, original_validation, base_names = tree_feature_matrices(
        train_raw, validation_raw, feature_names, metadata, base_tree_plan
    )
    imputed_train, imputed_validation, imputed_names = tree_feature_matrices(
        train_raw, validation_raw, feature_names, metadata, variant_plan
    )
    if imputed_names[:len(base_names)] != base_names:
        raise ValueError("Imputation changed the original source feature order.")
    processor = FeaturePreprocessor(feature_names, metadata, linear_plan).fit(train_raw)
    train = _replace_source_values(
        processor.transform(train_raw), original_train, imputed_train,
        base_names, _target_names(variant_plan), processor,
    )
    validation = _replace_source_values(
        processor.transform(validation_raw), original_validation, imputed_validation,
        base_names, _target_names(variant_plan), processor,
    )
    flag_names = imputed_names[len(base_names):]
    if flag_names:
        train = np.column_stack((train, imputed_train[:, len(base_names):]))
        validation = np.column_stack((validation, imputed_validation[:, len(base_names):]))
    train, validation, output_names = _add_interactions(
        train, validation, list(processor.output_feature_names) + flag_names, INTERACTIONS
    )
    if train.shape[1] != validation.shape[1] or not np.all(np.isfinite(train)) or not np.all(np.isfinite(validation)):
        raise ValueError("The imputed logistic design has invalid values or dimensions.")
    return train, validation, output_names


def _fit_probabilities(train: np.ndarray, y_train: np.ndarray, validation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, weights, _ = next(_fit_checkpoints(y_train, train, MODEL))
    probability = _sigmoid(validation @ weights)
    if not np.all(np.isfinite(probability)):
        raise ValueError("Logistic regression produced nonfinite probabilities.")
    return probability, weights


def _metrics(y: np.ndarray, probability: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    counts, metrics = classification_metrics(y, predicted)
    metrics.update({
        "average_precision": average_precision(y, probability),
        "log_loss": binary_log_loss(y, probability),
        "true_positive": counts.true_positives,
        "false_positive": counts.false_positives,
        "true_negative": counts.true_negatives,
        "false_negative": counts.false_negatives,
    })
    return metrics


def run_variant(variant_name: str, *, optimistic_only: bool = False) -> tuple[Path, Path]:
    suite = _read_json(VARIANT_CONFIG)
    variant = next(item for item in suite["variants"] if item["experiment_name"] == variant_name)
    tree_plan = _read_json(BASE_TREE_CONFIG)["variants"][0]["preprocessing"]
    linear_plan = _read_json(PHASE7_RECORD)["config"]["features"]["preprocessing"]
    metadata = load_feature_metadata(METADATA)
    with HEADER.open(encoding="utf-8", newline="") as handle:
        feature_names = next(csv.reader(handle))[1:]
    x = np.load(FEATURES, mmap_mode="r")
    full_y = _as_binary_labels(np.load(LABELS))
    with np.load(SPLITS) as split:
        development_indices = np.asarray(split["development_indices"], dtype=np.int64)
    y = full_y[development_indices]
    if x.shape[1] != len(feature_names):
        raise ValueError("Feature matrix and CSV header do not align.")
    artifact_dir = ARTIFACT_ROOT / (
        f"{variant_name}_optimistic" if optimistic_only else variant_name
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    probabilities = np.full(y.size, np.nan, dtype=np.float64)
    fold_ids = np.zeros(y.size, dtype=np.int8)
    thresholds = np.full(y.size, np.nan, dtype=np.float64)
    fold_weights = []
    folds = []
    output_names_reference: list[str] | None = None
    started = time.perf_counter()

    for outer_fold, (outer_train_positions, outer_validation_positions) in enumerate(
        _stratified_folds(y, 3, SEED), start=1
    ):
        fold_ids[outer_validation_positions] = outer_fold
        train_raw = np.asarray(x[development_indices[outer_train_positions]], dtype=np.float32)
        validation_raw = np.asarray(x[development_indices[outer_validation_positions]], dtype=np.float32)
        train_design, validation_design, output_names = _designs(
            train_raw, validation_raw, feature_names, metadata,
            variant["preprocessing"], tree_plan, linear_plan,
        )
        if output_names_reference is None:
            output_names_reference = output_names
        elif output_names != output_names_reference:
            raise ValueError("Logistic feature names changed between outer folds.")
        outer_probability, weights = _fit_probabilities(
            train_design, y[outer_train_positions], validation_design
        )
        probabilities[outer_validation_positions] = outer_probability
        fold_weights.append(weights)
        del train_raw, validation_raw, train_design, validation_design

        fold_path = artifact_dir / f"fold{outer_fold}_outputs.npz"
        if optimistic_only:
            np.savez_compressed(
                fold_path,
                outer_validation_positions=outer_validation_positions,
                outer_probabilities=outer_probability,
                outer_labels=y[outer_validation_positions],
                weights=weights,
            )
            folds.append({
                "outer_fold": outer_fold,
                "outer_training_rows": int(outer_train_positions.size),
                "outer_validation_rows": int(outer_validation_positions.size),
            })
            print(f"{variant_name}: outer fold {outer_fold}/3 complete", flush=True)
            continue

        inner_y = y[outer_train_positions]
        inner_probability = np.full(inner_y.size, np.nan, dtype=np.float64)
        for inner_train_positions, inner_validation_positions in _stratified_folds(
            inner_y, 2, INNER_SEED + outer_fold
        ):
            inner_train_raw = np.asarray(
                x[development_indices[outer_train_positions[inner_train_positions]]], dtype=np.float32
            )
            inner_validation_raw = np.asarray(
                x[development_indices[outer_train_positions[inner_validation_positions]]], dtype=np.float32
            )
            inner_train_design, inner_validation_design, inner_names = _designs(
                inner_train_raw, inner_validation_raw, feature_names, metadata,
                variant["preprocessing"], tree_plan, linear_plan,
            )
            if inner_names != output_names_reference:
                raise ValueError("Logistic feature names changed in an inner fold.")
            inner_probability[inner_validation_positions], _ = _fit_probabilities(
                inner_train_design, inner_y[inner_train_positions], inner_validation_design
            )
            del inner_train_raw, inner_validation_raw, inner_train_design, inner_validation_design
        if not np.all(np.isfinite(inner_probability)):
            raise ValueError("Inner OOF logistic probabilities were not fully populated.")
        choice = best_f1_threshold(inner_y, inner_probability)
        thresholds[outer_validation_positions] = choice.threshold
        fold_predicted = labels_from_scores(outer_probability, choice.threshold)
        fold_metrics = _metrics(y[outer_validation_positions], outer_probability, fold_predicted)
        folds.append({
            "outer_fold": outer_fold,
            "inner_threshold": choice.threshold,
            "outer_training_rows": int(outer_train_positions.size),
            "outer_validation_rows": int(outer_validation_positions.size),
            "outer_validation_metrics": fold_metrics,
        })
        np.savez_compressed(
            fold_path,
            outer_validation_positions=outer_validation_positions,
            outer_probabilities=outer_probability,
            outer_labels=y[outer_validation_positions],
            inner_training_positions=outer_train_positions,
            inner_probabilities=inner_probability,
            inner_labels=inner_y,
            nested_threshold=np.asarray([choice.threshold]),
            nested_predictions=fold_predicted,
            weights=weights,
        )
        print(f"{variant_name}: fold {outer_fold}/3 complete, nested F1={fold_metrics['f1']:.6f}", flush=True)

    if not np.all(np.isfinite(probabilities)) or np.any(fold_ids == 0):
        raise ValueError("Outer OOF logistic outputs were not fully populated.")
    if not optimistic_only and not np.all(np.isfinite(thresholds)):
        raise ValueError("Nested thresholds were not fully populated.")
    choice = best_f1_threshold(y, probabilities)
    best_predictions = labels_from_scores(probabilities, choice.threshold)
    if optimistic_only:
        for item in folds:
            mask = fold_ids == item["outer_fold"]
            item["metrics_at_same_oof_best_threshold_optimistic"] = _metrics(
                y[mask], probabilities[mask], best_predictions[mask]
            )
    oof_path = artifact_dir / "oof_predictions.npz"
    temporary_path = artifact_dir / "oof_predictions.tmp.npz"
    oof_arrays = {
        "development_indices": development_indices,
        "fold_ids": fold_ids,
        "labels": y,
        "probabilities": probabilities,
        "same_oof_best_threshold": np.asarray([choice.threshold]),
        "same_oof_best_predictions": best_predictions,
        "fold_weights": np.asarray(fold_weights),
        "feature_names": np.asarray(["intercept", *(output_names_reference or [])]),
    }
    if not optimistic_only:
        nested_predictions = (probabilities >= thresholds).astype(np.int8)
        oof_arrays["nested_thresholds"] = thresholds
        oof_arrays["nested_predictions"] = nested_predictions
    np.savez_compressed(temporary_path, **oof_arrays)
    os.replace(temporary_path, oof_path)
    report = {
        "variant": variant_name,
        "model": MODEL,
        "evaluation_mode": "outer_oof_same_labels_best_threshold_optimistic" if optimistic_only else "nested_threshold",
        "outer_fold_count": 3,
        "folds": folds,
        "same_oof_best_threshold": choice.threshold,
        "same_oof_best_threshold_metrics_optimistic": _metrics(y, probabilities, best_predictions),
        "oof_artifact": str(oof_path.relative_to(ROOT)),
        "fold_artifacts": [str((artifact_dir / f"fold{fold}_outputs.npz").relative_to(ROOT)) for fold in (1, 2, 3)],
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation": (
            "The threshold is selected on the same outer OOF labels used for the reported F1; this F1 is optimistic."
            if optimistic_only else
            "Nested F1 uses an inner OOF threshold fitted wholly within each outer training fold. The same-OOF best-threshold F1 is optimistic."
        ),
    }
    if not optimistic_only:
        report["inner_fold_count"] = 2
        report["inner_seed"] = INNER_SEED
        report["nested_threshold_metrics"] = _metrics(y, probabilities, nested_predictions)
    report_path = artifact_dir / "evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{variant_name}: saved {oof_path} and {report_path}", flush=True)
    return oof_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("phase16_imputation_restricted", "phase16_imputation_expanded"))
    parser.add_argument("--optimistic-only", action="store_true", help="Fit only outer folds and choose the best threshold on their pooled OOF scores.")
    args = parser.parse_args()
    run_variant(args.variant, optimistic_only=args.optimistic_only)


if __name__ == "__main__":
    main()

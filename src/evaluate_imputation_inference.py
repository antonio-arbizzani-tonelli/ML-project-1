"""Fast OOF sensitivity test for Phase 16 imputations using saved disease models.

This reuses the outer-fold logistic weights and histogram boosting checkpoints.
It fits only the feature imputers and feature preprocessors needed to construct
each validation matrix; it does not fit either disease classifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from src.feature_preprocessing import FeaturePreprocessor, load_feature_metadata, tree_feature_matrices
from src.run_model_cv import _add_interactions, _sigmoid


ROOT = Path(__file__).resolve().parents[1]
LINEAR_RECORD = ROOT / "results/experiments/20260920T113608936798Z_phase7-p2-codebook-clean-bphigh4-corrected-interactions-gamma-0p2-nested-4000.json"
LINEAR_OOF = ROOT / "results/oof_predictions/20260920T113608778965Z_phase7-p2-codebook-clean-bphigh4-corrected-interactions-gamma-0p2-nested-4000_oof.npz"
TREE_RECORD = ROOT / "results/experiments/20260921T105139261213Z_phase14-boosting-codebook-corrections-only-depth5-features128-200trees.json"
TREE_OOF = ROOT / "results/boosting_artifacts/phase14_codebook_corrections_only/oof_predictions/phase14_boosting_codebook_corrections_only_depth5_features128_200trees_oof.npz"
TREE_CHECKPOINT_DIR = ROOT / "results/boosting_artifacts/phase14_codebook_corrections_only/model_checkpoints"
TREE_PREPROCESSING = ROOT / "configs/experiments/phase14_codebook_corrections_only.json"
IMPUTATION_CONFIG = ROOT / "configs/experiments/phase16_selected_imputation_preprocessing.json"
PHASE7_PREPROCESSING = ROOT / "configs/experiments/phase3_preprocessing_cv.json"
FEATURES = ROOT / "data/processed/x_train_float32.npy"
LABELS = ROOT / "data/processed/y_train_int8.npy"
SPLITS = ROOT / "results/eda/analysis/splits/stratified_seed_20260918.npz"
METADATA = ROOT / "configs/feature_metadata.json"
OUTPUT = ROOT / "results/experiments/phase16_imputation_fast_oof_inference.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(y: np.ndarray, probability: np.ndarray, threshold: np.ndarray | float) -> dict[str, float]:
    predicted = probability >= threshold
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    tn = int(np.sum(~predicted & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    average_precision = _average_precision(y, probability)
    return {
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "precision": precision,
        "recall": recall,
        "average_precision": average_precision,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def _sorted_threshold_sweep(y: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-probability, kind="mergesort")
    sorted_probability = probability[order]
    sorted_labels = y[order].astype(np.int64)
    ends = np.r_[np.flatnonzero(sorted_probability[:-1] != sorted_probability[1:]), y.size - 1]
    true_positives = np.cumsum(sorted_labels)[ends]
    false_positives = ends + 1 - true_positives
    thresholds = sorted_probability[ends]
    recall = true_positives / max(int(np.sum(y)), 1)
    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    return thresholds, precision, recall, true_positives


def _average_precision(y: np.ndarray, probability: np.ndarray) -> float:
    _, precision, recall, _ = _sorted_threshold_sweep(y, probability)
    previous_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - previous_recall) * precision))


def _best_f1_threshold(y: np.ndarray, probability: np.ndarray) -> tuple[float, dict[str, float]]:
    thresholds, precision, recall, _ = _sorted_threshold_sweep(y, probability)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
    best = int(np.argmax(f1))
    threshold = float(thresholds[best])
    return threshold, _metrics(y, probability, threshold)


def _thresholds_by_fold(record: dict[str, Any]) -> dict[int, float]:
    return {
        int(item.get("outer_fold", item.get("fold"))): float(
            item.get("inner_threshold", item.get("threshold"))
        )
        for item in record["outcome"]["nested_threshold_evaluation"]["folds"]
    }


def _evaluate_scores(y: np.ndarray, fold_ids: np.ndarray, probability: np.ndarray, thresholds: dict[int, float]) -> dict[str, Any]:
    row_thresholds = np.asarray([thresholds[int(fold)] for fold in fold_ids])
    fixed = _metrics(y, probability, row_thresholds)
    best_threshold, best = _best_f1_threshold(y, probability)
    return {
        "nested_threshold_f1": fixed,
        "best_threshold_on_same_oof_f1_optimistic": {
            "threshold": best_threshold,
            "metrics": best,
        },
    }


def _align_oof(linear: np.lib.npyio.NpzFile, tree: np.lib.npyio.NpzFile) -> None:
    for key in ("development_indices", "fold_ids", "labels"):
        if not np.array_equal(linear[key], tree[key]):
            raise ValueError(f"Linear and tree OOF artifacts disagree on {key}.")


def _checkpoint(fold: int):
    path = TREE_CHECKPOINT_DIR / f"phase14_boosting_codebook_corrections_only_depth5_features128_fold{fold}_200trees.pkl"
    with path.open("rb") as handle:
        return pickle.load(handle)["model"]


def _imputation_target_names(settings: dict[str, Any]) -> list[str]:
    names = list(settings.get("nonresponse_features", []))
    names.extend(settings.get("continuous_blank_features", []))
    if settings.get("derive_bmi_from_height_weight"):
        names.append("_BMI5")
    return list(dict.fromkeys(names))


def _fit_saved_logistic_preprocessor(
    feature_names: list[str], metadata: list[dict[str, Any]], plan: dict[str, Any], training: np.ndarray
) -> FeaturePreprocessor:
    """Recreate the earlier median-fill representation used by saved Phase 7 weights."""
    processor = FeaturePreprocessor(feature_names, metadata, plan).fit(training)
    legacy_states = []
    for state in processor._states or []:
        if state.auxiliary_only or state.one_hot_categories:
            legacy_states.append(state)
            continue
        values, _ = processor._clean_values(
            training[:, state.index], state.missing_codes, state.zero_codes
        )
        if state.binary_map:
            converted = values.copy()
            for code, mapped in state.binary_map.items():
                converted[values == code] = mapped
            values = converted
        missing = ~np.isfinite(values)
        fill_value = float(np.nanmedian(values))
        imputed = np.where(missing, fill_value, values)
        if state.semantic_type == "binary" and state.binary_map:
            mean, scale = 0.0, 1.0
        else:
            mean = float(np.mean(imputed))
            scale = float(np.std(imputed))
            if scale == 0.0:
                scale = 1.0
        legacy_states.append(
            replace(state, fill_value=fill_value, mean=mean, scale=scale)
        )
    processor._states = legacy_states
    return processor


def _replace_linear_sources(
    baseline_design: np.ndarray,
    imputed_tree: np.ndarray,
    clean_tree: np.ndarray,
    tree_names: list[str],
    target_names: list[str],
    linear_processor: FeaturePreprocessor,
) -> np.ndarray:
    """Replace only the transformed value columns; preserve original missing flags."""
    result = baseline_design.copy()
    linear_positions = {name: index for index, name in enumerate(linear_processor.output_feature_names)}
    tree_positions = {name: index for index, name in enumerate(tree_names)}
    linear_states = {state.name: state for state in (linear_processor._states or [])}
    tree_processor = FeaturePreprocessor(
        linear_processor.feature_names,
        list(linear_processor.metadata_by_name.values()),
        linear_processor.plan,
    )
    # The tree matrix's binary source columns use the same fold-fitted mapping.
    # Its Phase 16 plan can add missing overrides, so infer mappings directly from
    # the observed raw fold values through the values represented in clean_tree.
    del tree_processor
    for name in target_names:
        if name not in linear_positions or name not in tree_positions:
            raise ValueError(f"Imputation target {name} is absent from a model representation.")
        state = linear_states[name]
        if state.one_hot_categories:
            raise ValueError(f"Unexpected one-hot imputation target: {name}")
        column = tree_positions[name]
        eligible = ~np.isfinite(clean_tree[:, column]) & np.isfinite(imputed_tree[:, column])
        if not np.any(eligible):
            continue
        values = imputed_tree[eligible, column].astype(np.float64)
        if state.binary_map:
            # Both preprocessors map the binary source to 0/1. Confirm the linear
            # mapping is complete and retain its exact scale for the saved weights.
            mapped_values = values
            if not set(np.unique(mapped_values)).issubset(set(state.binary_map.values())):
                raise ValueError(f"Binary mapping mismatch for {name}.")
        else:
            mapped_values = values
        result[eligible, linear_positions[name] + 1] = (
            (mapped_values - state.mean) / state.scale
        ).astype(np.float32)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    linear_oof = np.load(LINEAR_OOF, allow_pickle=True)
    tree_oof = np.load(TREE_OOF, allow_pickle=True)
    _align_oof(linear_oof, tree_oof)
    development_indices = linear_oof["development_indices"]
    fold_ids = linear_oof["fold_ids"].astype(int)
    y = linear_oof["labels"].astype(np.int8)
    x = np.load(FEATURES, mmap_mode="r")
    metadata = load_feature_metadata(METADATA)
    header_path = ROOT / "data/raw/dataset/x_train.csv"
    with header_path.open(encoding="utf-8", newline="") as handle:
        all_names = next(csv.reader(handle))[1:]
    if x.shape[1] != len(all_names):
        raise ValueError("Feature metadata does not align with the saved feature matrix.")

    linear_record = _read_json(LINEAR_RECORD)
    tree_record = _read_json(TREE_RECORD)
    phase16 = _read_json(IMPUTATION_CONFIG)
    phase7 = _read_json(PHASE7_PREPROCESSING)
    linear_config = linear_record["config"]
    preprocessing_variant = next(
        item for item in phase7["variants"]
        if item["experiment_name"] == linear_config["features"]["preprocessing_variant"]
    )
    linear_plan = preprocessing_variant["preprocessing"]
    interactions = linear_config["features"]["interactions"]
    linear_weights = linear_oof["fold_weights"]
    linear_thresholds = _thresholds_by_fold(linear_record)
    tree_thresholds = _thresholds_by_fold(tree_record)

    y_tree = tree_oof["labels"].astype(np.int8)
    base_outputs: dict[str, np.ndarray] = {
        "linear": np.asarray(linear_oof["probabilities"], dtype=np.float64).copy(),
        "tree": np.asarray(tree_oof["probabilities"], dtype=np.float64).copy(),
    }
    altered: dict[str, dict[str, np.ndarray]] = {
        variant["experiment_name"]: {
            "linear": np.full(y.size, np.nan, dtype=np.float64),
            "tree": np.full(y.size, np.nan, dtype=np.float64),
        }
        for variant in phase16["variants"]
    }
    variant_plans = {
        variant["experiment_name"]: variant["preprocessing"]
        for variant in phase16["variants"]
    }
    variant_targets = {
        name: _imputation_target_names(plan["imputation"])
        for name, plan in variant_plans.items()
    }

    for fold in sorted(np.unique(fold_ids)):
        fold_positions = np.flatnonzero(fold_ids == fold)
        train_positions = np.flatnonzero(fold_ids != fold)
        train_raw = np.asarray(x[development_indices[train_positions]], dtype=np.float32)
        valid_raw = np.asarray(x[development_indices[fold_positions]], dtype=np.float32)
        labels = y[fold_positions]

        # Restore and verify the exact original logistic representation. The model
        # weights are saved, so this is preprocessing plus matrix multiplication.
        processor = _fit_saved_logistic_preprocessor(all_names, metadata, linear_plan, train_raw)
        base_train = processor.transform(train_raw)
        base_valid = processor.transform(valid_raw)
        train_design, base_valid_interacted, output_names = _add_interactions(
            base_train, base_valid, list(processor.output_feature_names), interactions
        )
        if not np.array_equal(np.asarray(linear_oof["feature_names"]), np.asarray(["intercept", *output_names])):
            raise ValueError("Reconstructed logistic feature names do not match saved OOF weights.")
        reconstructed = _sigmoid(base_valid_interacted.astype(np.float64) @ linear_weights[int(fold) - 1])
        saved = base_outputs["linear"][fold_positions]
        if not np.allclose(reconstructed, saved, rtol=0, atol=2e-6):
            differences = [
                float(np.max(np.abs(_sigmoid(base_valid_interacted.astype(np.float64) @ weight) - saved)))
                for weight in linear_weights
            ]
            raise ValueError(f"Saved logistic weights did not reproduce fold {fold} OOF scores; max errors by saved weight row: {differences}.")

        model = _checkpoint(int(fold))
        phase14 = _read_json(TREE_PREPROCESSING)
        phase14_plan = phase14["variants"][0]["preprocessing"]
        base_tree_train, base_tree_valid, base_tree_names = tree_feature_matrices(
            train_raw, valid_raw, all_names, metadata, phase14_plan
        )
        if base_tree_valid.shape[1] != model.n_features_in_:
            raise ValueError("Phase 14 feature count does not match the saved tree checkpoint.")
        reconstructed_tree = model.predict_proba(base_tree_valid)[:, 1]
        saved_tree = base_outputs["tree"][fold_positions]
        if not np.allclose(reconstructed_tree, saved_tree, rtol=0, atol=2e-6):
            raise ValueError(f"Saved tree checkpoint did not reproduce fold {fold} OOF scores.")
        for variant_name, plan in variant_plans.items():
            imp_train, imp_valid, output_names_tree = tree_feature_matrices(
                train_raw, valid_raw, all_names, metadata, plan
            )
            base_count = model.n_features_in_
            if output_names_tree[:base_count] != base_tree_names:
                raise ValueError("Imputation changed the saved tree model's base feature order.")
            if imp_valid.shape[1] < base_count:
                raise ValueError("Imputed tree matrix has fewer columns than the saved model.")
            tree_probability = model.predict_proba(imp_valid[:, :base_count])[:, 1]
            altered[variant_name]["tree"][fold_positions] = tree_probability

            # Preserve the original validation masks and change only each imputed
            # source's value contribution to the saved logistic design matrix.
            changed_base = _replace_linear_sources(
                base_valid,
                imp_valid[:, :len(base_tree_names)],
                base_tree_valid,
                base_tree_names,
                variant_targets[variant_name],
                processor,
            )
            _, changed_interacted, changed_names = _add_interactions(
                base_train, changed_base, list(processor.output_feature_names), interactions
            )
            if changed_names != output_names:
                raise ValueError("Imputation changed the logistic feature layout.")
            altered[variant_name]["linear"][fold_positions] = _sigmoid(
                changed_interacted.astype(np.float64) @ linear_weights[int(fold) - 1]
            )
        print(f"completed outer fold {fold}/3", flush=True)

    results: dict[str, Any] = {
        "protocol": "3-fold OOF inference using saved Phase 7 logistic weights and Phase 14 tree checkpoints; no disease classifier retraining",
        "models": {
            "linear": {
                "source_experiment": linear_record["config"]["experiment_name"],
                "saved_oof_reproduction_verified": True,
                "original_oof": _evaluate_scores(y, fold_ids, base_outputs["linear"], linear_thresholds),
            },
            "tree_depth5_200": {
                "source_experiment": tree_record["config"]["experiment_name"],
                "saved_oof": _evaluate_scores(y_tree, fold_ids, base_outputs["tree"], tree_thresholds),
            },
        },
        "variants": {},
        "interpretation": {
            "nested_threshold_f1": "Uses the threshold selected inside each corresponding outer training fold.",
            "best_threshold_on_same_oof_f1_optimistic": "Chooses a threshold on all OOF labels being scored; exploratory and optimistic, not an unbiased validation estimate.",
            "scope": "Only validation feature values are imputed using the corresponding outer training fold. Saved disease model parameters and original feature layouts are fixed. Imputation flags are not added because these models were not trained with them.",
        },
    }
    for variant_name, predictions in altered.items():
        results["variants"][variant_name] = {
            "imputed_features": variant_targets[variant_name],
            **{
                model_name: _evaluate_scores(
                    y,
                    fold_ids,
                    values,
                    linear_thresholds if model_name == "linear" else tree_thresholds,
                )
                for model_name, values in predictions.items()
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()

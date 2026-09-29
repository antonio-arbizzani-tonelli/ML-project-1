"""Measure feature recovery before running the downstream boosting comparison.

Known answers in the saved 20% validation partition serve as a proxy task;
the imputers fit on the 80% development partition.
The resulting scores do not prove accuracy for people who declined to answer.
No cardiovascular target labels are loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.conditional_imputation import (
    _AnchorBinner,
    _categorical_predictions,
    _continuous_predictions,
    _response_code_groups,
)
from src.feature_preprocessing import (
    FeaturePreprocessor,
    load_feature_metadata,
    tree_feature_matrices,
)


def run_audit(
    project_root: Path,
    variant_name: str,
    output_path: Path,
    fold: int | None = None,
) -> list[dict]:
    """Compare each imputer with a training-only mode or median prediction."""
    config = json.loads(
        (project_root / "configs/experiments/phase15_imputation_preprocessing.json")
        .read_text(encoding="utf-8")
    )
    matches = [v for v in config["variants"] if v["experiment_name"] == variant_name]
    if len(matches) != 1:
        raise ValueError(f"Unknown imputation variant: {variant_name}")
    plan = matches[0]["preprocessing"]
    settings = plan["imputation"]
    base_plan = {key: value for key, value in plan.items() if key != "imputation"}
    with (project_root / "data/raw/dataset/x_train.csv").open(newline="") as handle:
        names = next(csv.reader(handle))[1:]
    metadata = load_feature_metadata(project_root / "configs/feature_metadata.json")
    features = np.load(project_root / "data/processed/x_train_float32.npy", mmap_mode="r")
    if fold is None:
        with np.load(project_root / "results/eda/analysis/splits/stratified_seed_20260918.npz") as split:
            training_rows = np.asarray(split["development_indices"], dtype=np.int64)
            validation_rows = np.asarray(split["validation_indices"], dtype=np.int64)
    else:
        fold_artifact = (
            project_root
            / "results/boosting_artifacts/phase14_codebook_corrections_only/oof_predictions/phase14_boosting_codebook_corrections_only_depth5_features128_200trees_oof.npz"
        )
        with np.load(fold_artifact) as saved:
            development = np.asarray(saved["development_indices"], dtype=np.int64)
            fold_ids = np.asarray(saved["fold_ids"], dtype=np.int8)
        if fold not in (1, 2, 3) or not np.all(np.isin(fold_ids, (1, 2, 3))):
            raise ValueError("fold must be 1, 2, or 3 and align with saved Phase 14 folds.")
        validation_rows = development[fold_ids == fold]
        training_rows = development[fold_ids != fold]
    raw_train = np.asarray(features[training_rows], dtype=np.float32)
    raw_validation = np.asarray(features[validation_rows], dtype=np.float32)
    clean_train, clean_validation, output_names = tree_feature_matrices(
        raw_train, raw_validation, names, metadata, base_plan
    )
    processor = FeaturePreprocessor(names, metadata, base_plan)
    positions = {name: index for index, name in enumerate(output_names)}
    anchors = settings["anchor_features"]
    anchor_positions = [positions[name] for name in anchors]
    binner = _AnchorBinner(settings["anchor_bins"]).fit(clean_train[:, anchor_positions])
    train_bins = binner.transform(clean_train[:, anchor_positions])
    validation_bins = binner.transform(clean_validation[:, anchor_positions])
    records = []
    for name in output_names:
        entry = processor.metadata_by_name[name]
        source_kind = entry.get("source_kind")
        response_codes = _response_code_groups(processor, name, entry)
        nonresponse_codes = response_codes["unknown"] + response_codes["refused"]
        should_impute_nonresponse = (
            source_kind in settings["nonresponse_source_kinds"] and bool(nonresponse_codes)
        )
        should_impute_blank = name in settings["continuous_blank_features"]
        if not should_impute_nonresponse and not should_impute_blank:
            continue
        index = positions[name]
        observed_train = np.isfinite(clean_train[:, index])
        observed_validation = np.isfinite(clean_validation[:, index])
        count = int(np.sum(observed_train))
        validation_count = int(np.sum(observed_validation))
        raw_index = processor._name_to_index[name]
        raw_column = raw_train[:, raw_index]
        nonresponse_count = int(np.sum(np.isin(raw_column, nonresponse_codes)))
        blank_count = int(np.sum(np.isnan(raw_column)))
        record = {
            "feature": name,
            "source_kind": source_kind,
            "semantic_type": entry["semantic_type"],
            "observed_training": count,
            "observed_validation": validation_count,
            "nonresponse_training": nonresponse_count,
            "nonresponse_validation": int(
                np.sum(np.isin(raw_validation[:, raw_index], nonresponse_codes))
            ),
            "blank_training": blank_count,
            "blank_validation": int(np.sum(np.isnan(raw_validation[:, raw_index]))),
            "status": "insufficient_observed",
            "metric": "",
            "imputer_score": "",
            "baseline_score": "",
            "balanced_accuracy": "",
            "baseline_balanced_accuracy": "",
            "macro_f1": "",
            "baseline_macro_f1": "",
        }
        if count < settings["minimum_observed"] or validation_count == 0:
            records.append(record)
            continue
        selected = [index for index, anchor in enumerate(anchors) if anchor != name]
        x_train = train_bins[:, selected]
        x_validation = validation_bins[observed_validation][:, selected]
        target_train = clean_train[:, index]
        actual = clean_validation[observed_validation, index]
        if entry["semantic_type"] == "continuous":
            prediction = _continuous_predictions(
                x_train, target_train, observed_train, x_validation,
                settings["ridge"], settings["maximum_continuous_training_rows"],
            )
            reference = np.median(target_train[observed_train])
            record["metric"] = "mean_absolute_error_lower_is_better"
            record["imputer_score"] = float(np.mean(np.abs(actual - prediction)))
            record["baseline_score"] = float(np.mean(np.abs(actual - reference)))
        else:
            prediction = _categorical_predictions(
                x_train, target_train, observed_train, x_validation,
                settings["smoothing"], settings["prediction_batch_size"],
            )
            values, counts = np.unique(target_train[observed_train], return_counts=True)
            reference = values[np.argmax(counts)]
            record["metric"] = "exact_accuracy_higher_is_better"
            record["imputer_score"] = float(np.mean(actual == prediction))
            record["baseline_score"] = float(np.mean(actual == reference))
            validation_classes = np.unique(actual)
            recalls = []
            class_f1 = []
            baseline_recalls = []
            baseline_f1 = []
            for response in validation_classes:
                actual_class = actual == response
                predicted_class = prediction == response
                true_positive = int(np.sum(actual_class & predicted_class))
                false_positive = int(np.sum(~actual_class & predicted_class))
                false_negative = int(np.sum(actual_class & ~predicted_class))
                recalls.append(true_positive / int(np.sum(actual_class)))
                denominator = 2 * true_positive + false_positive + false_negative
                class_f1.append(2 * true_positive / denominator if denominator else 0.0)
                baseline_tp = int(np.sum(actual_class & (reference == response)))
                baseline_fp = int(np.sum(~actual_class & (reference == response)))
                baseline_fn = int(np.sum(actual_class & (reference != response)))
                baseline_recalls.append(baseline_tp / int(np.sum(actual_class)))
                baseline_denominator = 2 * baseline_tp + baseline_fp + baseline_fn
                baseline_f1.append(
                    2 * baseline_tp / baseline_denominator
                    if baseline_denominator else 0.0
                )
            record["balanced_accuracy"] = float(np.mean(recalls))
            record["macro_f1"] = float(np.mean(class_f1))
            record["baseline_balanced_accuracy"] = float(np.mean(baseline_recalls))
            record["baseline_macro_f1"] = float(np.mean(baseline_f1))
        record["status"] = "evaluated"
        records.append(record)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("phase15_imputation_direct", "phase15_imputation_all"),
        default="phase15_imputation_direct",
    )
    parser.add_argument("--fold", type=int, choices=(1, 2, 3))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("results/eda/imputation_quality.csv"))
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else project_root / args.output
    if args.fold is not None:
        output = output.with_name(
            f"{args.variant}_quality_fold{args.fold}{output.suffix}"
        )
    records = run_audit(project_root, args.variant, output, args.fold)
    print(f"Wrote {len(records)} feature recovery checks to {output}")


if __name__ == "__main__":
    main()

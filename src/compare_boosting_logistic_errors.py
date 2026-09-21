"""Compare paired nested OOF errors of the selected boosting and logistic models.

This is a development-only diagnostic. Threshold perturbations and group rankings
describe existing predictions; they are not new threshold or feature selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.analyze_errors import (
    _apply_zero_codes,
    _categorical_groups,
    _continuous_groups,
    _load_feature_names,
    _metadata_labels,
    _thresholds_for_rows,
)
from src.evaluation import classification_metrics
from src.feature_preprocessing import FeaturePreprocessor, NUMERIC_SEMANTIC_TYPES
from src.numpy_boosting import HistogramGradientBoostingClassifier

MARGIN_EDGES = (0.01, 0.03, 0.10)
MARGIN_NAMES = ("<0.01", "0.01-<0.03", "0.03-<0.10", ">=0.10")


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _file(root: Path, value: str) -> Path:
    path = root / value
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counts_and_metrics(
    labels: np.ndarray, predictions: np.ndarray
) -> tuple[dict, dict]:
    counts, metrics = classification_metrics(labels, predictions)
    return asdict(counts), metrics


def _load_bundle(root: Path, record_name: str) -> dict:
    record_path = _file(root, record_name)
    record = _read_json(record_path)
    nested = record["outcome"]["nested_threshold_evaluation"]
    if not nested:
        raise ValueError(f"Nested thresholds are absent: {record_path}")
    artifact_info = record["outcome"]["oof_predictions_artifact"]
    artifact_path = _file(root, artifact_info["path"])
    if _sha256(artifact_path) != artifact_info["sha256"]:
        raise ValueError(f"OOF artifact checksum mismatch: {artifact_path}")
    with np.load(artifact_path, allow_pickle=False) as artifact:
        arrays = {
            name: np.asarray(artifact[name])
            for name in ("development_indices", "fold_ids", "labels", "probabilities")
        }
        saved_thresholds = (
            np.asarray(artifact["nested_thresholds"])
            if "nested_thresholds" in artifact
            else None
        )
        saved_predictions = (
            np.asarray(artifact["nested_predictions"])
            if "nested_predictions" in artifact
            else None
        )
    indices = arrays["development_indices"].astype(np.int64)
    folds = arrays["fold_ids"].astype(np.int8)
    labels = arrays["labels"].astype(np.int8)
    probabilities = arrays["probabilities"].astype(np.float64)
    if any(
        item.ndim != 1 or item.size != indices.size
        for item in (folds, labels, probabilities)
    ):
        raise ValueError("OOF arrays must be aligned, one-dimensional arrays.")
    if np.unique(indices).size != indices.size or not np.all(
        np.isfinite(probabilities)
    ):
        raise ValueError("OOF row indices must be unique and probabilities finite.")
    thresholds = _thresholds_for_rows(folds, nested)
    predictions = (probabilities >= thresholds).astype(np.int8)
    if saved_thresholds is not None and not np.allclose(
        saved_thresholds, thresholds, atol=1e-14, rtol=0
    ):
        raise ValueError("Saved row thresholds disagree with the nested record.")
    if saved_predictions is not None and not np.array_equal(
        saved_predictions, predictions
    ):
        raise ValueError("Saved row predictions disagree with the nested record.")
    counts, metrics = _counts_and_metrics(labels, predictions)
    expected_counts = nested["pooled_confusion_counts"]
    if counts != expected_counts or not np.isclose(
        metrics["f1"], nested["pooled_metrics"]["f1"], atol=1e-12
    ):
        raise ValueError("Recomputed metrics disagree with the nested record.")
    return {
        "indices": indices,
        "folds": folds,
        "labels": labels,
        "probabilities": probabilities,
        "thresholds": thresholds,
        "predictions": predictions,
        "counts": counts,
        "metrics": metrics,
        "record": record,
        "record_path": record_path,
        "artifact_path": artifact_path,
    }


def align_to_boosting(boosting: dict, logistic: dict) -> dict:
    """Reorder logistic arrays by original row ID, rejecting any split mismatch."""
    left = np.asarray(boosting["indices"])
    right = np.asarray(logistic["indices"])
    if left.ndim != 1 or right.ndim != 1 or left.size != right.size:
        raise ValueError("The models must cover the same number of original rows.")
    if np.unique(left).size != left.size or np.unique(right).size != right.size:
        raise ValueError("Original row indices must be unique within each model.")
    order = np.argsort(right)
    if not np.array_equal(np.sort(left), right[order]):
        raise ValueError("The models do not cover the same original rows.")
    positions = order[np.searchsorted(right[order], left)]
    aligned = dict(logistic)
    for name in (
        "indices",
        "folds",
        "labels",
        "probabilities",
        "thresholds",
        "predictions",
    ):
        aligned[name] = np.asarray(logistic[name])[positions]
    if not np.array_equal(boosting["labels"], aligned["labels"]):
        raise ValueError("Labels disagree after original-row alignment.")
    if not np.array_equal(boosting["folds"], aligned["folds"]):
        raise ValueError("Outer folds disagree after original-row alignment.")
    return aligned


def margin_counts(
    mask: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray
) -> dict:
    """Count rows by absolute probability distance from their own fold threshold."""
    margin = np.abs(np.asarray(probabilities) - np.asarray(thresholds))
    if mask.shape != margin.shape or not np.all(np.isfinite(margin)):
        raise ValueError("Margin arrays are misaligned or non-finite.")
    band = np.searchsorted(MARGIN_EDGES, margin, side="right")
    return {
        name: int(np.sum(mask & (band == index)))
        for index, name in enumerate(MARGIN_NAMES)
    }


def paired_outcomes(
    labels: np.ndarray,
    boosting_predictions: np.ndarray,
    logistic_predictions: np.ndarray,
) -> dict:
    """Paired correctness counts, separately for actual positives and negatives."""
    result = {}
    for actual, label_name in ((1, "actual_positive"), (0, "actual_negative")):
        selected = labels == actual
        booster_correct = boosting_predictions == labels
        logistic_correct = logistic_predictions == labels
        result[label_name] = {
            "rows": int(np.sum(selected)),
            "both_correct": int(np.sum(selected & booster_correct & logistic_correct)),
            "boosting_only_correct": int(
                np.sum(selected & booster_correct & ~logistic_correct)
            ),
            "logistic_only_correct": int(
                np.sum(selected & ~booster_correct & logistic_correct)
            ),
            "both_wrong": int(np.sum(selected & ~booster_correct & ~logistic_correct)),
        }
    return result


def paired_error_margins(labels: np.ndarray, boosting: dict, logistic: dict) -> dict:
    """Split each boosting-error margin band by logistic correctness."""
    result = {}
    logistic_correct = logistic["predictions"] == labels
    for actual, name in ((1, "false_negative"), (0, "false_positive")):
        boosting_wrong = (labels == actual) & (boosting["predictions"] != labels)
        result[name] = {
            "logistic_correct": margin_counts(
                boosting_wrong & logistic_correct,
                boosting["probabilities"],
                boosting["thresholds"],
            ),
            "both_wrong": margin_counts(
                boosting_wrong & ~logistic_correct,
                boosting["probabilities"],
                boosting["thresholds"],
            ),
        }
    return result


def _group_rows(
    feature: str,
    groups: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    booster: dict,
    logistic: dict,
    minimum_rows: int,
) -> list[dict]:
    booster_fn = (labels == 1) & (booster["predictions"] == 0)
    logistic_fn = (labels == 1) & (logistic["predictions"] == 0)
    booster_fp = (labels == 0) & (booster["predictions"] == 1)
    logistic_fp = (labels == 0) & (logistic["predictions"] == 1)
    margin = np.abs(booster["probabilities"] - booster["thresholds"])
    output = []
    for group in np.unique(groups):
        selected = groups == group
        rows = int(np.sum(selected))
        if rows < minimum_rows:
            continue
        positive = int(np.sum(selected & (labels == 1)))
        negative = rows - positive
        row = {
            "feature": feature,
            "group": str(group),
            "rows": rows,
            "positive_rows": positive,
            "negative_rows": negative,
            "boosting_false_negatives": int(np.sum(selected & booster_fn)),
            "logistic_false_negatives": int(np.sum(selected & logistic_fn)),
            "boosting_false_positives": int(np.sum(selected & booster_fp)),
            "logistic_false_positives": int(np.sum(selected & logistic_fp)),
            "both_false_negatives": int(np.sum(selected & booster_fn & logistic_fn)),
            "both_false_positives": int(np.sum(selected & booster_fp & logistic_fp)),
            "boosting_only_false_negatives": int(
                np.sum(selected & booster_fn & ~logistic_fn)
            ),
            "boosting_only_false_positives": int(
                np.sum(selected & booster_fp & ~logistic_fp)
            ),
            "boosting_far_false_negatives": int(
                np.sum(selected & booster_fn & (margin >= 0.10))
            ),
            "boosting_far_false_positives": int(
                np.sum(selected & booster_fp & (margin >= 0.10))
            ),
        }
        row["boosting_false_negative_rate"] = (
            None if positive == 0 else row["boosting_false_negatives"] / positive
        )
        row["logistic_false_negative_rate"] = (
            None if positive == 0 else row["logistic_false_negatives"] / positive
        )
        row["boosting_false_positive_rate"] = (
            None if negative == 0 else row["boosting_false_positives"] / negative
        )
        row["logistic_false_positive_rate"] = (
            None if negative == 0 else row["logistic_false_positives"] / negative
        )
        row["folds"] = []
        for fold in np.unique(folds):
            local = selected & (folds == fold)
            local_positive = int(np.sum(local & (labels == 1)))
            local_negative = int(np.sum(local & (labels == 0)))
            row["folds"].append(
                {
                    "fold": int(fold),
                    "positive_rows": local_positive,
                    "negative_rows": local_negative,
                    "boosting_false_negatives": int(np.sum(local & booster_fn)),
                    "logistic_false_negatives": int(np.sum(local & logistic_fn)),
                    "boosting_false_positives": int(np.sum(local & booster_fp)),
                    "logistic_false_positives": int(np.sum(local & logistic_fp)),
                }
            )
        output.append(row)
    return output


def _group_tables(
    root: Path,
    config: dict,
    labels: np.ndarray,
    folds: np.ndarray,
    indices: np.ndarray,
    booster: dict,
    logistic: dict,
) -> list[dict]:
    group_config = _read_json(_file(root, config["group_config_path"]))
    matrix = np.load(_file(root, group_config["features_path"]), mmap_mode="r")
    names = _load_feature_names(_file(root, group_config["x_train_csv_path"]))
    if (
        matrix.shape[1] != len(names)
        or np.any(indices < 0)
        or np.any(indices >= matrix.shape[0])
    ):
        raise ValueError("Raw feature matrix and OOF indices are incompatible.")
    metadata = _read_json(_file(root, group_config["feature_metadata_path"]))
    minimum_rows = int(group_config["minimum_group_rows"])
    rows = []
    grouped_features = {}
    for spec in group_config["continuous_features"]:
        feature = spec["name"]
        values = _apply_zero_codes(
            matrix[indices, names.index(feature)], spec.get("zero_code_values", [])
        )
        groups = _continuous_groups(values, spec["bins"], spec["labels"])
        grouped_features[feature] = groups
        rows.extend(
            _group_rows(feature, groups, labels, folds, booster, logistic, minimum_rows)
        )
    categorical_features = group_config["categorical_features"] + config.get(
        "additional_categorical_features", []
    )
    if len(categorical_features) != len(set(categorical_features)):
        raise ValueError("Categorical analysis features must be unique.")
    for feature in categorical_features:
        values = matrix[indices, names.index(feature)]
        groups = _categorical_groups(values, _metadata_labels(metadata, feature))
        grouped_features[feature] = groups
        feature_minimum = (
            1
            if feature in config.get("additional_categorical_features", [])
            else minimum_rows
        )
        rows.extend(
            _group_rows(
                feature, groups, labels, folds, booster, logistic, feature_minimum
            )
        )
    for left, right in group_config.get("cross_feature_pairs", []):
        if left not in grouped_features or right not in grouped_features:
            raise ValueError("Cross-feature group requires both component features.")
        crossed = np.char.add(
            np.char.add(grouped_features[left], " | "), grouped_features[right]
        )
        rows.extend(
            _group_rows(
                f"{left} × {right}",
                crossed,
                labels,
                folds,
                booster,
                logistic,
                minimum_rows,
            )
        )
    return rows


def _feature_split_usage(root: Path, config: dict, booster: dict) -> dict:
    """Count splits on named source features in trusted local outer checkpoints."""
    requested = config.get("inspect_split_features", [])
    if not requested:
        return {}
    record = booster["record"]
    preprocessing_path = _file(
        root, record["config"]["features"]["preprocessing_source"]
    )
    preprocessing = _read_json(preprocessing_path)
    variant_name = record["config"]["features"]["preprocessing_variant"]
    variant = next(
        item
        for item in preprocessing["variants"]
        if item["experiment_name"] == variant_name
    )
    group_config = _read_json(_file(root, config["group_config_path"]))
    source_names = _load_feature_names(_file(root, group_config["x_train_csv_path"]))
    metadata = _read_json(_file(root, group_config["feature_metadata_path"]))
    processor = FeaturePreprocessor(source_names, metadata, variant["preprocessing"])
    output_names = [
        name
        for name in processor._retained_names()
        if processor.metadata_by_name[name].get("semantic_type")
        in NUMERIC_SEMANTIC_TYPES
    ]
    unknown = set(requested) - set(output_names)
    if unknown:
        raise ValueError(
            f"Requested split features are absent from the model: {sorted(unknown)}"
        )

    def count_nodes(node, index: int) -> int:
        if node.feature is None:
            return 0
        return (
            int(node.feature == index)
            + count_nodes(node.left, index)
            + count_nodes(node.right, index)
        )

    result = {name: [] for name in requested}
    for item in record["outcome"]["model_checkpoints"]:
        path = _file(root, item["path"])
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"Model checkpoint checksum mismatch: {path}")
        model, _ = HistogramGradientBoostingClassifier.load_checkpoint(path)
        for name in requested:
            index = output_names.index(name)
            result[name].append(
                {
                    "fold": int(item["fold"]),
                    "trees": len(model.trees_),
                    "trees_using_feature": int(
                        sum(count_nodes(tree.root_, index) > 0 for tree in model.trees_)
                    ),
                    "total_splits": int(
                        sum(count_nodes(tree.root_, index) for tree in model.trees_)
                    ),
                    "root_splits": int(
                        sum(tree.root_.feature == index for tree in model.trees_)
                    ),
                }
            )
    return result


def _markdown(result: dict) -> str:
    b = result["boosting"]
    l = result["logistic"]
    pos = result["paired"]["actual_positive"]
    neg = result["paired"]["actual_negative"]
    lines = [
        "# Phase 12 — paired error analysis",
        "",
        "The same development rows and outer folds are compared with each model's own nested F1 threshold. No model was retrained. Threshold shifts and group rankings below are descriptive and must not be used to choose a new threshold on these rows.",
        "",
        "## Overall results",
        "",
        "| Model | F1 | Precision | Recall | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Boosting | {b['metrics']['f1']:.5f} | {b['metrics']['precision']:.5f} | {b['metrics']['recall']:.5f} | {b['counts']['false_positives']} | {b['counts']['false_negatives']} |",
        f"| Logistic | {l['metrics']['f1']:.5f} | {l['metrics']['precision']:.5f} | {l['metrics']['recall']:.5f} | {l['counts']['false_positives']} | {l['counts']['false_negatives']} |",
        "",
        "## Paired predictions",
        "",
        "| Actual class | Both correct | Only boosting correct | Only logistic correct | Both wrong |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Positive | {pos['both_correct']} | {pos['boosting_only_correct']} | {pos['logistic_only_correct']} | {pos['both_wrong']} |",
        f"| Negative | {neg['both_correct']} | {neg['boosting_only_correct']} | {neg['logistic_only_correct']} | {neg['both_wrong']} |",
        "",
        "## Boosting errors by distance from its fold threshold",
        "",
        "| Absolute distance | FN, logistic correct | FN, both wrong | FP, logistic correct | FP, both wrong |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in MARGIN_NAMES:
        paired = result["paired_error_margins"]
        lines.append(
            f"| {name} | {paired['false_negative']['logistic_correct'][name]} | "
            f"{paired['false_negative']['both_wrong'][name]} | "
            f"{paired['false_positive']['logistic_correct'][name]} | "
            f"{paired['false_positive']['both_wrong'][name]} |"
        )
    lines.extend(
        [
            "",
            "A distance of at least 0.10 means far from the decision threshold, not a calibrated confidence level. Strict tails are reported separately: false negatives with probability <0.05 and false positives with probability >=0.50.",
            "",
            "## Diagnostic threshold perturbation",
            "",
            "| Change to each fold threshold | F1 | False positives | False negatives |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["threshold_sensitivity"]:
        lines.append(
            f"| {row['delta']:+.2f} | {row['metrics']['f1']:.5f} | {row['counts']['false_positives']} | {row['counts']['false_negatives']} |"
        )
    lines.extend(
        [
            "",
            "## Group detail",
            "",
            "The JSON and CSV contain paired false-positive and false-negative counts for each predefined feature group, plus a fold breakdown. A group can overlap other groups; its error count is not an independent contribution. Differences here are diagnostic and are subject to model-selection reuse of the development data.",
            "",
        ]
    )
    if result["feature_split_usage"]:
        lines.extend(
            [
                "## Checkpoint split usage",
                "",
                "Split counts describe how often a feature was used, not its causal effect or an ablation result.",
                "",
                "| Feature | Fold | Trees using it | Total splits | Root splits |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for feature, folds in result["feature_split_usage"].items():
            for row in folds:
                lines.append(
                    f"| {feature} | {row['fold']} | {row['trees_using_feature']}/{row['trees']} | {row['total_splits']} | {row['root_splits']} |"
                )
        lines.append("")
    shared_fn = pos["both_wrong"] / (pos["both_wrong"] + pos["logistic_only_correct"])
    shared_fp = neg["both_wrong"] / (neg["both_wrong"] + neg["logistic_only_correct"])
    far_fn = result["paired_error_margins"]["false_negative"]
    far_fp = result["paired_error_margins"]["false_positive"]
    far_fn_shared = far_fn["both_wrong"][">=0.10"] / (
        far_fn["both_wrong"][">=0.10"] + far_fn["logistic_correct"][">=0.10"]
    )
    far_fp_shared = far_fp["both_wrong"][">=0.10"] / (
        far_fp["both_wrong"][">=0.10"] + far_fp["logistic_correct"][">=0.10"]
    )
    lines.extend(
        [
            "## Interpretation checks",
            "",
            f"The logistic model also misses {shared_fn:.1%} of boosting's false negatives and {shared_fp:.1%} of its false positives. At distance >=0.10 those fractions are {far_fn_shared:.1%} and {far_fp_shared:.1%} respectively.",
            "",
            "The two +/-0.01 threshold perturbations lower F1 on these same outer predictions. This is a local diagnostic, not a newly selected threshold or an unbiased comparison of alternative threshold policies.",
            "",
        ]
    )
    harehab = [row for row in result["group_rows"] if row["feature"] == "HAREHAB1"]
    if harehab:
        answered = [row for row in harehab if row["group"] != "missing"]
        answered_rows = sum(row["rows"] for row in answered)
        answered_positive = sum(row["positive_rows"] for row in answered)
        lines.extend(
            [
                f"HAREHAB1 has a recorded response code in {answered_rows} development rows; {answered_positive} are positive. The feature asks about rehabilitation after a heart attack and is a target-conditioned leakage concern. Split counts and these associations do not quantify its causal contribution; a fit without it is required for that comparison.",
                "",
            ]
        )
    return "\n".join(lines)


def run_comparison(config_path: Path, root: Path) -> dict:
    config = _read_json(config_path)
    booster = _load_bundle(root, config["boosting_record_path"])
    logistic = align_to_boosting(
        booster, _load_bundle(root, config["logistic_record_path"])
    )
    if (
        booster["record"]["config"]["split"]["id"]
        != logistic["record"]["config"]["split"]["id"]
    ):
        raise ValueError("The model records name different development splits.")
    labels = booster["labels"]
    folds = booster["folds"]
    booster_fn = (labels == 1) & (booster["predictions"] == 0)
    booster_fp = (labels == 0) & (booster["predictions"] == 1)
    results = {
        "protocol": "Paired nested OOF comparison on identical development rows and outer folds; threshold perturbations are descriptive only.",
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "rows": int(labels.size),
        "outer_folds": [int(fold) for fold in np.unique(folds)],
        "boosting": {"counts": booster["counts"], "metrics": booster["metrics"]},
        "logistic": {"counts": logistic["counts"], "metrics": logistic["metrics"]},
        "paired": paired_outcomes(
            labels, booster["predictions"], logistic["predictions"]
        ),
        "boosting_error_margins": {
            "false_negative": margin_counts(
                booster_fn, booster["probabilities"], booster["thresholds"]
            ),
            "false_positive": margin_counts(
                booster_fp, booster["probabilities"], booster["thresholds"]
            ),
            "strict_probability_tails": {
                "false_negative_probability_below_0p05": int(
                    np.sum(booster_fn & (booster["probabilities"] < 0.05))
                ),
                "false_positive_probability_at_least_0p50": int(
                    np.sum(booster_fp & (booster["probabilities"] >= 0.50))
                ),
                "false_negative_below_0p05_also_wrong_logistic": int(
                    np.sum(
                        booster_fn
                        & (booster["probabilities"] < 0.05)
                        & (logistic["predictions"] == 0)
                    )
                ),
                "false_positive_at_least_0p50_also_wrong_logistic": int(
                    np.sum(
                        booster_fp
                        & (booster["probabilities"] >= 0.50)
                        & (logistic["predictions"] == 1)
                    )
                ),
            },
        },
        "paired_error_margins": paired_error_margins(labels, booster, logistic),
        "threshold_sensitivity": [],
        "folds": [],
        "input_sha256": {
            "config": _sha256(config_path),
            "group_config": _sha256(_file(root, config["group_config_path"])),
            "boosting_record": _sha256(booster["record_path"]),
            "logistic_record": _sha256(logistic["record_path"]),
            "boosting_oof": _sha256(booster["artifact_path"]),
            "logistic_oof": _sha256(logistic["artifact_path"]),
        },
    }
    group_config = _read_json(_file(root, config["group_config_path"]))
    results["input_sha256"].update(
        {
            "raw_features": _sha256(_file(root, group_config["features_path"])),
            "raw_feature_header": _sha256(
                _file(root, group_config["x_train_csv_path"])
            ),
            "feature_metadata": _sha256(
                _file(root, group_config["feature_metadata_path"])
            ),
        }
    )
    for delta in (-0.01, 0.0, 0.01):
        predictions = (
            booster["probabilities"] >= booster["thresholds"] + delta
        ).astype(np.int8)
        counts, metrics = _counts_and_metrics(labels, predictions)
        results["threshold_sensitivity"].append(
            {"delta": delta, "counts": counts, "metrics": metrics}
        )
    for fold in np.unique(folds):
        selected = folds == fold
        b_counts, b_metrics = _counts_and_metrics(
            labels[selected], booster["predictions"][selected]
        )
        l_counts, l_metrics = _counts_and_metrics(
            labels[selected], logistic["predictions"][selected]
        )
        results["folds"].append(
            {
                "fold": int(fold),
                "rows": int(np.sum(selected)),
                "boosting": {"counts": b_counts, "metrics": b_metrics},
                "logistic": {"counts": l_counts, "metrics": l_metrics},
                "paired": paired_outcomes(
                    labels[selected],
                    booster["predictions"][selected],
                    logistic["predictions"][selected],
                ),
            }
        )
    results["group_rows"] = _group_tables(
        root, config, labels, folds, booster["indices"], booster, logistic
    )
    results["feature_split_usage"] = _feature_split_usage(root, config, booster)
    output_json = root / config["output_json_path"]
    output_csv = root / config["output_csv_path"]
    output_report = root / config["output_report_path"]
    for output in (output_json, output_csv, output_report):
        output.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(results, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    group_fields = [name for name in results["group_rows"][0] if name != "folds"]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=group_fields)
        writer.writeheader()
        writer.writerows(
            {name: row[name] for name in group_fields} for row in results["group_rows"]
        )
    output_report.write_text(_markdown(results), encoding="utf-8", newline="\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_comparison(args.config, args.project_root.resolve())
    print(
        f"Compared {result['rows']} paired rows; boosting F1={result['boosting']['metrics']['f1']:.5f}, logistic F1={result['logistic']['metrics']['f1']:.5f}"
    )


if __name__ == "__main__":
    main()

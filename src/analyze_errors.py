"""Analyze nested out-of-fold classification errors without touching the outer holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.evaluation import classification_metrics

ERROR_ORDER = ("true_negative", "false_positive", "false_negative", "true_positive")


def _resolve(root: Path, configured_path: str) -> Path:
    path = root / configured_path
    if not path.is_file():
        raise FileNotFoundError(f"Required analysis input does not exist: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_feature_names(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    if not header or header[0] != "Id":
        raise ValueError("x_train.csv must begin with Id.")
    return header[1:]


def _thresholds_for_rows(fold_ids: np.ndarray, nested: dict) -> np.ndarray:
    """Map each outer OOF row to the threshold selected in its own outer fold."""

    fold_ids = np.asarray(fold_ids)
    if fold_ids.ndim != 1 or np.any(fold_ids < 1):
        raise ValueError("OOF fold IDs must be a one-dimensional positive array.")
    thresholds = {}
    for detail in nested["folds"]:
        fold = int(detail["outer_fold"])
        if fold in thresholds:
            raise ValueError("Nested folds must have unique fold IDs.")
        thresholds[fold] = float(detail["inner_threshold"])
    if set(np.unique(fold_ids)) != set(thresholds):
        raise ValueError("OOF fold IDs and nested threshold folds disagree.")
    return np.asarray([thresholds[int(fold)] for fold in fold_ids], dtype=np.float64)


def _error_names(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    if labels.shape != predictions.shape or not np.all((labels == 0) | (labels == 1)):
        raise ValueError("Labels and predictions must be aligned binary arrays.")
    names = np.empty(labels.size, dtype="U14")
    names[(labels == 0) & (predictions == 0)] = "true_negative"
    names[(labels == 0) & (predictions == 1)] = "false_positive"
    names[(labels == 1) & (predictions == 0)] = "false_negative"
    names[(labels == 1) & (predictions == 1)] = "true_positive"
    return names


def _numeric_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"rows": 0, "mean": None, "median": None, "q25": None, "q75": None}
    return {
        "rows": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def _metadata_labels(metadata: list[dict], feature: str) -> dict[float, str]:
    entry = next((item for item in metadata if item["name"] == feature), None)
    if entry is None:
        return {}
    labels = {}
    for value in entry.get("documented_values", []):
        try:
            labels[float(value["code"])] = value["label"]
        except (TypeError, ValueError):
            continue
    return labels


def _group_rows(
    feature: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    error_names: np.ndarray,
    group_codes: np.ndarray,
    group_labels: list[str],
    minimum_group_rows: int,
) -> list[dict]:
    rows = []
    for code, label in zip(np.unique(group_codes), group_labels):
        del code
        mask = group_codes == label
        count = int(np.sum(mask))
        if count < minimum_group_rows:
            continue
        positive = int(np.sum(labels[mask] == 1))
        negative = count - positive
        false_negative = int(np.sum(error_names[mask] == "false_negative"))
        false_positive = int(np.sum(error_names[mask] == "false_positive"))
        rows.append(
            {
                "feature": feature,
                "group": str(label),
                "rows": count,
                "positive_rows": positive,
                "negative_rows": negative,
                "true_negative": int(np.sum(error_names[mask] == "true_negative")),
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_positive": int(np.sum(error_names[mask] == "true_positive")),
                "false_negative_rate": (
                    None if positive == 0 else false_negative / positive
                ),
                "false_positive_rate": (
                    None if negative == 0 else false_positive / negative
                ),
            }
        )
    return rows


def _continuous_groups(
    values: np.ndarray, bins: list[float], labels: list[str]
) -> np.ndarray:
    if len(bins) != len(labels) + 1:
        raise ValueError(
            "Continuous analysis bins require one more boundary than labels."
        )
    values = np.asarray(values, dtype=np.float64)
    result = np.full(values.size, "missing", dtype="U80")
    finite = np.isfinite(values)
    positions = np.digitize(values[finite], np.asarray(bins[1:-1]), right=False)
    result[finite] = np.asarray(labels, dtype="U80")[positions]
    return result


def _apply_zero_codes(values: np.ndarray, zero_code_values: list[float]) -> np.ndarray:
    """Map documented zero-day response codes before numerical grouping."""

    result = np.asarray(values, dtype=np.float64).copy()
    for code in zero_code_values:
        result[result == float(code)] = 0.0
    return result


def _categorical_groups(values: np.ndarray, labels: dict[float, str]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.full(values.size, "missing", dtype="U160")
    for value in np.unique(values[np.isfinite(values)]):
        rounded = float(value)
        name = labels.get(rounded, f"code {rounded:g}")
        result[values == value] = f"{rounded:g}: {name}"
    return result


def _cross_group_rows(
    feature_pair: tuple[str, str],
    left_groups: np.ndarray,
    right_groups: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    error_names: np.ndarray,
    fold_ids: np.ndarray,
    minimum_group_rows: int,
) -> list[dict]:
    """Summarize error rates for one pair of already cleaned feature groupings."""

    pair_name = " × ".join(feature_pair)
    groups = np.char.add(
        np.char.add(left_groups.astype("U160"), " | "), right_groups.astype("U160")
    )
    rows = []
    for group in np.unique(groups):
        mask = groups == group
        count = int(np.sum(mask))
        if count < minimum_group_rows:
            continue
        positive = int(np.sum(labels[mask] == 1))
        negative = count - positive
        row = {
            "feature_pair": pair_name,
            "group": str(group),
            "rows": count,
            "positive_rows": positive,
            "negative_rows": negative,
            "true_negative": int(np.sum(error_names[mask] == "true_negative")),
            "false_positive": int(np.sum(error_names[mask] == "false_positive")),
            "false_negative": int(np.sum(error_names[mask] == "false_negative")),
            "true_positive": int(np.sum(error_names[mask] == "true_positive")),
            "false_negative_rate": (
                None
                if positive == 0
                else int(np.sum(error_names[mask] == "false_negative")) / positive
            ),
            "false_positive_rate": (
                None
                if negative == 0
                else int(np.sum(error_names[mask] == "false_positive")) / negative
            ),
            "folds": [],
        }
        for fold in np.unique(fold_ids):
            fold_mask = mask & (fold_ids == fold)
            fold_count = int(np.sum(fold_mask))
            fold_positive = int(np.sum(labels[fold_mask] == 1))
            fold_negative = fold_count - fold_positive
            row["folds"].append(
                {
                    "fold": int(fold),
                    "rows": fold_count,
                    "positive_rows": fold_positive,
                    "negative_rows": fold_negative,
                    "false_negative": int(
                        np.sum(error_names[fold_mask] == "false_negative")
                    ),
                    "false_positive": int(
                        np.sum(error_names[fold_mask] == "false_positive")
                    ),
                    "false_negative_rate": (
                        None
                        if fold_positive == 0
                        else int(np.sum(error_names[fold_mask] == "false_negative"))
                        / fold_positive
                    ),
                    "false_positive_rate": (
                        None
                        if fold_negative == 0
                        else int(np.sum(error_names[fold_mask] == "false_positive"))
                        / fold_negative
                    ),
                }
            )
        rows.append(row)
    return rows


def _report_text(result: dict) -> str:
    counts = result["error_counts"]
    score_rows = result["score_by_outcome"]
    priority = result["priority_groups"]
    lines = [
        "# Nested out-of-fold error analysis",
        "",
        "This report uses nested out-of-fold predictions from the configured experiment. Each row is classified with the threshold selected from inner folds of its own outer fold. The 20% local validation partition is not loaded.",
        "",
        "## Error counts",
        "",
        "| Outcome | Rows |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {name.replace('_', ' ')} | {counts[name]} |" for name in ERROR_ORDER
    )
    lines.extend(
        [
            "",
            "## Probability distribution by outcome",
            "",
            "| Outcome | Rows | Mean | Median | Q25 | Q75 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in score_rows:
        lines.append(
            f"| {row['outcome'].replace('_', ' ')} | {row['rows']} | {row['mean']:.5f} | {row['median']:.5f} | {row['q25']:.5f} | {row['q75']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Groups with the highest miss rate among actual positives",
            "",
            "| Feature | Group | Rows | Positive rows | FN rate |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in priority["false_negative_rate"]:
        lines.append(
            f"| {row['feature']} | {row['group']} | {row['rows']} | {row['positive_rows']} | {row['false_negative_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Groups with the highest false-positive rate among actual negatives",
            "",
            "| Feature | Group | Rows | Negative rows | FP rate |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in priority["false_positive_rate"]:
        lines.append(
            f"| {row['feature']} | {row['group']} | {row['rows']} | {row['negative_rows']} | {row['false_positive_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Crossed groups",
            "",
            "The JSON result includes each requested pair and its three outer-fold breakdown. The paired analysis is used to decide whether an apparent single-feature error pattern persists after considering the second variable.",
        ]
    )
    lines.extend(
        [
            "",
            "The group tables are diagnostic, not causal evidence: many features overlap and the same development data guided the model search. A promising follow-up needs to change one representation element at a time and be checked across outer folds.",
            "",
            "The configured CSV output contains the full group table with false-negative and false-positive rates.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_analysis(config_path: Path, project_root: Path) -> dict:
    """Create an auditable group-level analysis from an existing nested experiment record."""

    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    record_path = _resolve(project_root, config["experiment_record_path"])
    features_path = _resolve(project_root, config["features_path"])
    header_path = _resolve(project_root, config["x_train_csv_path"])
    metadata_path = _resolve(project_root, config["feature_metadata_path"])
    with record_path.open(encoding="utf-8") as handle:
        record = json.load(handle)
    artifact_path = _resolve(
        project_root, record["outcome"]["oof_predictions_artifact"]["path"]
    )
    nested = record["outcome"].get("nested_threshold_evaluation")
    if not nested:
        raise ValueError(
            "The experiment record does not have a nested threshold evaluation."
        )
    with np.load(artifact_path) as artifact:
        development_indices = np.asarray(
            artifact["development_indices"], dtype=np.int64
        )
        fold_ids = np.asarray(artifact["fold_ids"], dtype=np.int8)
        labels = np.asarray(artifact["labels"], dtype=np.int8)
        probabilities = np.asarray(artifact["probabilities"], dtype=np.float64)
    thresholds = _thresholds_for_rows(fold_ids, nested)
    predictions = (probabilities >= thresholds).astype(np.int8)
    counts, metrics = classification_metrics(labels, predictions)
    error_names = _error_names(labels, predictions)
    features = np.load(features_path, mmap_mode="r")
    feature_names = _load_feature_names(header_path)
    if features.shape[1] != len(feature_names):
        raise ValueError("Feature matrix and CSV header have different feature counts.")
    if development_indices.size != labels.size:
        raise ValueError("OOF development indices and labels disagree.")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    group_rows = []
    grouped_features = {}
    for specification in config["continuous_features"]:
        feature = specification["name"]
        if feature not in feature_names:
            raise ValueError(f"Unknown continuous error-analysis feature: {feature}")
        values = _apply_zero_codes(
            features[development_indices, feature_names.index(feature)],
            specification.get("zero_code_values", []),
        )
        groups = _continuous_groups(
            values, specification["bins"], specification["labels"]
        )
        grouped_features[feature] = groups
        group_rows.extend(
            _group_rows(
                feature,
                labels,
                predictions,
                error_names,
                groups,
                list(np.unique(groups)),
                config["minimum_group_rows"],
            )
        )
    for feature in config["categorical_features"]:
        if feature not in feature_names:
            raise ValueError(f"Unknown categorical error-analysis feature: {feature}")
        values = features[development_indices, feature_names.index(feature)]
        groups = _categorical_groups(values, _metadata_labels(metadata, feature))
        grouped_features[feature] = groups
        group_rows.extend(
            _group_rows(
                feature,
                labels,
                predictions,
                error_names,
                groups,
                list(np.unique(groups)),
                config["minimum_group_rows"],
            )
        )
    cross_rows = []
    for pair in config.get("cross_feature_pairs", []):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(
                "Each cross_feature_pairs entry must contain exactly two feature names."
            )
        left, right = pair
        if left not in grouped_features or right not in grouped_features:
            raise ValueError(
                "Cross-feature analysis requires both features to be configured above."
            )
        cross_rows.extend(
            _cross_group_rows(
                (left, right),
                grouped_features[left],
                grouped_features[right],
                labels,
                predictions,
                error_names,
                fold_ids,
                config["minimum_group_rows"],
            )
        )
    score_by_outcome = []
    for outcome in ERROR_ORDER:
        summary = _numeric_summary(probabilities[error_names == outcome])
        score_by_outcome.append({"outcome": outcome, **summary})
    minimum_class_rows = int(
        config.get("minimum_class_rows_for_priority", config["minimum_group_rows"])
    )
    fn_candidates = [
        row
        for row in group_rows
        if row["false_negative_rate"] is not None
        and row["positive_rows"] >= minimum_class_rows
    ]
    fp_candidates = [
        row
        for row in group_rows
        if row["false_positive_rate"] is not None
        and row["negative_rows"] >= minimum_class_rows
    ]
    result = {
        "protocol": "Nested OOF error analysis. Each row uses the threshold selected by its own outer fold's inner OOF predictions.",
        "nested_metrics_recomputed": metrics,
        "error_counts": {
            "true_negative": counts.true_negatives,
            "false_positive": counts.false_positives,
            "false_negative": counts.false_negatives,
            "true_positive": counts.true_positives,
        },
        "thresholds": {
            "minimum": float(np.min(thresholds)),
            "maximum": float(np.max(thresholds)),
            "mean": float(np.mean(thresholds)),
        },
        "score_by_outcome": score_by_outcome,
        "group_rows": group_rows,
        "cross_group_rows": cross_rows,
        "priority_groups": {
            "false_negative_rate": sorted(
                fn_candidates, key=lambda row: row["false_negative_rate"], reverse=True
            )[:10],
            "false_positive_rate": sorted(
                fp_candidates, key=lambda row: row["false_positive_rate"], reverse=True
            )[:10],
        },
        "priority_group_minimum_class_rows": minimum_class_rows,
        "input_sha256": {
            "config": _sha256(config_path),
            "experiment_record": _sha256(record_path),
            "oof_artifact": _sha256(artifact_path),
            "features": _sha256(features_path),
            "feature_metadata": _sha256(metadata_path),
        },
    }
    output_json = project_root / config["output_json_path"]
    output_csv = project_root / config["output_csv_path"]
    output_cross_csv = project_root / config["output_cross_csv_path"]
    output_report = project_root / config["output_report_path"]
    for output in (output_json, output_csv, output_cross_csv, output_report):
        output.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    fields = [
        "feature",
        "group",
        "rows",
        "positive_rows",
        "negative_rows",
        *ERROR_ORDER,
        "false_negative_rate",
        "false_positive_rate",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(group_rows)
    cross_fields = [
        "feature_pair",
        "group",
        "fold",
        "rows",
        "positive_rows",
        "negative_rows",
        "false_negative",
        "false_positive",
        "false_negative_rate",
        "false_positive_rate",
    ]
    with output_cross_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cross_fields)
        writer.writeheader()
        for row in cross_rows:
            for fold in row["folds"]:
                writer.writerow(
                    {"feature_pair": row["feature_pair"], "group": row["group"], **fold}
                )
    output_report.write_text(_report_text(result), encoding="utf-8", newline="\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_analysis(args.config, args.project_root.resolve())
    print(
        f"Nested error analysis: F1={result['nested_metrics_recomputed']['f1']:.5f}, "
        f"accuracy={result['nested_metrics_recomputed']['accuracy']:.5f}"
    )


if __name__ == "__main__":
    main()

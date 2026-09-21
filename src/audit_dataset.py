"""Stream a reproducible structural audit of the official Project 1 CSV files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def update_value_counts(
    counts: Optional[Counter[str]], value: str, cardinality_cap: int
) -> Optional[Counter[str]]:
    """Count values until cardinality exceeds the reporting limit."""

    if counts is None:
        return None
    counts[value] += 1
    return None if len(counts) > cardinality_cap else counts


def scan_feature_file(path: Path, cardinality_cap: int) -> Dict[str, object]:
    """Scan one feature CSV and summarize schema, missingness, and ranges."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if not header or header[0] != "Id":
            raise ValueError(f"{path.name}: expected first column to be 'Id'")

        feature_names = header[1:]
        feature_count = len(feature_names)
        missing = [0] * feature_count
        minima = [float("inf")] * feature_count
        maxima = [float("-inf")] * feature_count
        value_counts: List[Optional[Counter[str]]] = [Counter() for _ in feature_names]
        rows = 0
        first_id: Optional[int] = None
        last_id: Optional[int] = None
        previous_id: Optional[int] = None
        duplicate_ids = 0
        non_monotonic_ids = 0

        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"{path.name}:{row_number}: expected {len(header)} fields, "
                    f"found {len(row)}"
                )

            row_id = int(row[0])
            if first_id is None:
                first_id = row_id
            if previous_id is not None:
                duplicate_ids += int(row_id == previous_id)
                non_monotonic_ids += int(row_id <= previous_id)
            previous_id = row_id
            last_id = row_id

            for index, raw_value in enumerate(row[1:]):
                if raw_value == "":
                    missing[index] += 1
                    value_counts[index] = update_value_counts(
                        value_counts[index], "<MISSING>", cardinality_cap
                    )
                    continue

                value = float(raw_value)
                if value < minima[index]:
                    minima[index] = value
                if value > maxima[index]:
                    maxima[index] = value
                value_counts[index] = update_value_counts(
                    value_counts[index], raw_value, cardinality_cap
                )

            rows += 1

    columns = []
    for index, name in enumerate(feature_names):
        counts = value_counts[index]
        columns.append(
            {
                "name": name,
                "missing_count": missing[index],
                "missing_fraction": missing[index] / rows,
                "min": None if minima[index] == float("inf") else minima[index],
                "max": None if maxima[index] == float("-inf") else maxima[index],
                "unique_count": (
                    f">{cardinality_cap}" if counts is None else len(counts)
                ),
                "constant_including_missing": counts is not None and len(counts) == 1,
                "top_values": (
                    []
                    if counts is None
                    else [
                        {
                            "value": value,
                            "count": count,
                            "fraction": count / rows,
                        }
                        for value, count in counts.most_common(10)
                    ]
                ),
            }
        )

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "row_count": rows,
        "column_count_including_id": len(header),
        "feature_count": feature_count,
        "first_id": first_id,
        "last_id": last_id,
        "adjacent_duplicate_ids": duplicate_ids,
        "non_monotonic_id_steps": non_monotonic_ids,
        "total_missing_cells": sum(missing),
        "missing_cell_fraction": sum(missing) / (rows * feature_count),
        "columns": columns,
    }


def scan_label_file(path: Path) -> Dict[str, object]:
    """Scan the training-label CSV and count labels and ID anomalies."""

    counts: Counter[str] = Counter()
    rows = 0
    first_id: Optional[int] = None
    last_id: Optional[int] = None
    previous_id: Optional[int] = None
    non_monotonic_ids = 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != ["Id", "_MICHD"]:
            raise ValueError(f"{path.name}: unexpected header {header!r}")
        for row_number, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(f"{path.name}:{row_number}: expected 2 fields")
            row_id = int(row[0])
            if first_id is None:
                first_id = row_id
            if previous_id is not None:
                non_monotonic_ids += int(row_id <= previous_id)
            previous_id = row_id
            last_id = row_id
            counts[row[1]] += 1
            rows += 1

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "row_count": rows,
        "first_id": first_id,
        "last_id": last_id,
        "non_monotonic_id_steps": non_monotonic_ids,
        "label_counts": dict(sorted(counts.items())),
        "positive_fraction": counts["1"] / rows,
    }


def scan_submission(path: Path) -> Dict[str, object]:
    """Validate the sample-submission structure and label domain."""

    counts: Counter[str] = Counter()
    rows = 0
    first_id: Optional[int] = None
    last_id: Optional[int] = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != ["Id", "Prediction"]:
            raise ValueError(f"{path.name}: unexpected header {header!r}")
        for row_number, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(f"{path.name}:{row_number}: expected 2 fields")
            row_id = int(row[0])
            if first_id is None:
                first_id = row_id
            last_id = row_id
            counts[row[1]] += 1
            rows += 1

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "row_count": rows,
        "first_id": first_id,
        "last_id": last_id,
        "prediction_counts": dict(sorted(counts.items())),
    }


def summarize(train: Dict[str, object], test: Dict[str, object]) -> Dict[str, object]:
    """Create cross-file consistency and risk summaries."""

    train_columns = train["columns"]
    test_columns = test["columns"]
    if not isinstance(train_columns, list) or not isinstance(test_columns, list):
        raise TypeError("Column summaries must be lists")

    train_names = [column["name"] for column in train_columns]
    test_names = [column["name"] for column in test_columns]
    high_missing_train = sorted(
        train_columns, key=lambda column: column["missing_fraction"], reverse=True
    )[:20]
    constants_train = [
        column["name"]
        for column in train_columns
        if column["constant_including_missing"]
    ]
    missing_shift = sorted(
        (
            {
                "name": train_column["name"],
                "train_missing_fraction": train_column["missing_fraction"],
                "test_missing_fraction": test_column["missing_fraction"],
                "absolute_difference": abs(
                    train_column["missing_fraction"] - test_column["missing_fraction"]
                ),
            }
            for train_column, test_column in zip(train_columns, test_columns)
        ),
        key=lambda row: row["absolute_difference"],
        reverse=True,
    )[:20]

    return {
        "train_test_feature_names_equal_and_ordered": train_names == test_names,
        "train_test_id_ranges_disjoint": train["last_id"] < test["first_id"],
        "constant_train_features": constants_train,
        "top_20_train_missingness": high_missing_train,
        "top_20_train_test_missingness_shift": missing_shift,
        "dense_float64_mib": {
            "x_train": train["row_count"] * train["feature_count"] * 8 / 2**20,
            "x_test": test["row_count"] * test["feature_count"] * 8 / 2**20,
        },
        "dense_float32_mib": {
            "x_train": train["row_count"] * train["feature_count"] * 4 / 2**20,
            "x_test": test["row_count"] * test["feature_count"] * 4 / 2**20,
        },
    }


def parse_args(arguments: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cardinality-cap", type=int, default=256)
    return parser.parse_args(arguments)


def main(arguments: Optional[Sequence[str]] = None) -> None:
    """Run the audit and write its JSON result."""

    args = parse_args(arguments)
    data_dir = args.data_dir.resolve()
    train = scan_feature_file(data_dir / "x_train.csv", args.cardinality_cap)
    test = scan_feature_file(data_dir / "x_test.csv", args.cardinality_cap)
    labels = scan_label_file(data_dir / "y_train.csv")
    submission = scan_submission(data_dir / "sample_submission.csv")

    if train["row_count"] != labels["row_count"]:
        raise ValueError("Training features and labels have different row counts")
    if train["first_id"] != labels["first_id"] or train["last_id"] != labels["last_id"]:
        raise ValueError("Training feature and label ID ranges do not match")
    if test["row_count"] != submission["row_count"]:
        raise ValueError(
            "Test features and sample submission have different row counts"
        )
    if (
        test["first_id"] != submission["first_id"]
        or test["last_id"] != submission["last_id"]
    ):
        raise ValueError("Test and sample-submission ID ranges do not match")

    report = {
        "data_directory": str(data_dir),
        "cardinality_cap": args.cardinality_cap,
        "x_train": train,
        "y_train": labels,
        "x_test": test,
        "sample_submission": submission,
        "cross_file_checks": summarize(train, test),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["cross_file_checks"], indent=2))


if __name__ == "__main__":
    main()

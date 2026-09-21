"""Run the reproducible statistical EDA for the Project 1 BRFSS dataset.

The script complements, rather than replaces, ``audit_dataset.py`` and
``build_feature_registry.py``.  It keeps the raw CSV files immutable, creates a
fixed stratified development/validation split, scans train and test in chunks,
and writes machine-readable tables plus a compact set of diagnostic figures.

Only the Python standard library, NumPy, and Matplotlib are used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MISSING = "<MISSING>"
DEFAULT_SEED = 20260918
LOW_CARDINALITY_CAP = 256
UNEXPECTED_VALUE_EXAMPLE_CAP = 100


def canonical_value(value: float) -> str:
    """Return a stable, human-readable representation of a numeric CSV value."""

    if math.isnan(value):
        return MISSING
    if value.is_integer() and abs(value) < 10**15:
        return str(int(value))
    return format(value, ".15g")


def safe_float(value: object) -> Optional[float]:
    """Convert a registry value to float when it is numeric."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def total_variation(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    """Return total variation distance between two discrete count mappings."""

    left_total = sum(left.values())
    right_total = sum(right.values())
    if left_total == 0 or right_total == 0:
        return 0.0
    keys = set(left) | set(right)
    return 0.5 * sum(
        abs(left.get(key, 0) / left_total - right.get(key, 0) / right_total)
        for key in keys
    )


def mean_std(
    count: int, total: float, total_sq: float
) -> Tuple[Optional[float], Optional[float]]:
    """Return population mean and standard deviation from streaming moments."""

    if count == 0:
        return None, None
    mean = total / count
    variance = max(0.0, total_sq / count - mean * mean)
    return mean, math.sqrt(variance)


def standardized_mean_difference(
    mean_left: Optional[float],
    std_left: Optional[float],
    mean_right: Optional[float],
    std_right: Optional[float],
) -> Optional[float]:
    """Return the standardized difference between two group means."""

    if None in (mean_left, std_left, mean_right, std_right):
        return None
    pooled = math.sqrt((std_left**2 + std_right**2) / 2.0)
    if pooled == 0:
        return 0.0 if mean_left == mean_right else None
    return (mean_left - mean_right) / pooled


def sha256_array(values: np.ndarray) -> str:
    """Hash a NumPy array in a platform-stable integer representation."""

    canonical = np.asarray(values, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def stratified_split(
    labels: np.ndarray, validation_fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create deterministic stratified development and validation indices."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    validation_parts = []
    development_parts = []
    for label in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == label)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        validation_size = int(round(len(shuffled) * validation_fraction))
        validation_parts.append(shuffled[:validation_size])
        development_parts.append(shuffled[validation_size:])
    validation = np.sort(np.concatenate(validation_parts)).astype(np.int64)
    development = np.sort(np.concatenate(development_parts)).astype(np.int64)
    return development, validation


def numeric_code_intervals(
    documented_values: Sequence[Mapping[str, str]],
) -> List[Tuple[float, float]]:
    """Translate documented numeric codes/ranges into conservative intervals.

    BRFSS uses labels such as ``Height in meters [2 implied decimal places]``.
    The scale is applied only when the label explicitly documents it.
    """

    intervals: List[Tuple[float, float]] = []
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    word_numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
    }
    implied_places: Optional[int] = None
    for entry in documented_values:
        label = str(entry.get("label", ""))
        implied = re.search(
            r"(\d+|one|two|three|four|five)\s+implied decimal",
            label,
            flags=re.IGNORECASE,
        )
        if implied:
            token = implied.group(1).lower()
            implied_places = int(token) if token.isdigit() else word_numbers[token]
            break
    common_scale = 10.0 ** (-implied_places) if implied_places is not None else 1.0
    for entry in documented_values:
        code = str(entry.get("code", "")).strip()
        if not code or code.upper() == "BLANK":
            continue
        range_match = re.fullmatch(rf"\s*({number})\s*-\s*({number})\s*", code)
        exact_match = re.fullmatch(rf"\s*({number})\s*", code)
        if range_match:
            lower = float(range_match.group(1)) * common_scale
            upper = float(range_match.group(2)) * common_scale
            intervals.append((min(lower, upper), max(lower, upper)))
        elif exact_match:
            value = float(exact_match.group(1)) * common_scale
            intervals.append((value, value))
    return intervals


def validation_intervals(entry: Mapping[str, object]) -> List[Tuple[float, float]]:
    """Return codebook intervals only when substantive values were parsed.

    Some codebook blocks expose only exceptional codes (for example, refused
    date codes) while the ordinary date format appears in prose.  Treating
    those exceptional codes as the complete domain would create false alarms.
    """

    documented = entry.get("documented_values", [])
    special = {
        (str(value.get("code", "")), str(value.get("label", "")))
        for value in entry.get("special_values", [])
    }
    substantive = [
        value
        for value in documented
        if str(value.get("code", "")).upper() != "BLANK"
        and (str(value.get("code", "")), str(value.get("label", ""))) not in special
    ]
    if not numeric_code_intervals(substantive):
        return []
    return numeric_code_intervals(documented)


def allowed_numeric_mask(
    values: np.ndarray, intervals: Sequence[Tuple[float, float]]
) -> np.ndarray:
    """Return which finite numeric values match at least one documented interval."""

    allowed = np.zeros(values.shape, dtype=bool)
    for lower, upper in intervals:
        tolerance = max(1e-9, abs(lower) * 1e-10, abs(upper) * 1e-10)
        allowed |= (values >= lower - tolerance) & (values <= upper + tolerance)
    return allowed


def quantile_bin_counts(values: np.ndarray, edges: np.ndarray) -> Counter[str]:
    """Count finite values in fixed bins and retain missing as a category."""

    result: Counter[str] = Counter()
    finite = np.isfinite(values)
    if np.any(finite):
        bins = np.digitize(values[finite], edges[1:-1], right=True)
        unique, counts = np.unique(bins, return_counts=True)
        for index, count in zip(unique, counts):
            lower = edges[index]
            upper = edges[index + 1]
            result[f"[{lower:.6g}, {upper:.6g}]"] += int(count)
    missing_count = int((~finite).sum())
    if missing_count:
        result[MISSING] = missing_count
    return result


def stable_quantile_edges(values: np.ndarray, bin_count: int = 10) -> np.ndarray:
    """Return unique quantile edges, extended to include all finite values."""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.array([-math.inf, math.inf], dtype=float)
    quantiles = np.quantile(finite, np.linspace(0.0, 1.0, bin_count + 1))
    edges = np.unique(quantiles.astype(float))
    if edges.size == 1:
        delta = max(1.0, abs(edges[0]) * 1e-6)
        return np.array([edges[0] - delta, edges[0] + delta])
    edges[0] = -math.inf
    edges[-1] = math.inf
    return edges


def iter_csv_chunks(
    path: Path, expected_features: Sequence[str], chunk_size: int
) -> Iterator[Tuple[int, np.ndarray, np.ndarray, List[List[str]]]]:
    """Yield ``(start, ids, numeric_features, raw_feature_rows)`` chunks."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        expected_header = ["Id", *expected_features]
        if header != expected_header:
            raise ValueError(f"{path.name}: feature header differs from registry")
        start = 0
        id_buffer: List[int] = []
        raw_buffer: List[List[str]] = []
        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"{path.name}:{row_number}: expected {len(header)} fields, found {len(row)}"
                )
            id_buffer.append(int(row[0]))
            raw_buffer.append(row[1:])
            if len(raw_buffer) == chunk_size:
                matrix = np.asarray(
                    [
                        [float(value) if value else np.nan for value in row_values]
                        for row_values in raw_buffer
                    ],
                    dtype=np.float64,
                )
                yield start, np.asarray(id_buffer, dtype=np.int64), matrix, raw_buffer
                start += len(raw_buffer)
                id_buffer = []
                raw_buffer = []
        if raw_buffer:
            matrix = np.asarray(
                [
                    [float(value) if value else np.nan for value in row_values]
                    for row_values in raw_buffer
                ],
                dtype=np.float64,
            )
            yield start, np.asarray(id_buffer, dtype=np.int64), matrix, raw_buffer


def load_labels(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load and validate ``y_train.csv``."""

    ids: List[int] = []
    labels: List[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader) != ["Id", "_MICHD"]:
            raise ValueError("Unexpected y_train.csv header")
        for row in reader:
            ids.append(int(row[0]))
            labels.append(int(row[1]))
    label_array = np.asarray(labels, dtype=np.int8)
    if set(np.unique(label_array)) != {-1, 1}:
        raise ValueError("Expected labels {-1, +1}")
    return np.asarray(ids, dtype=np.int64), label_array


@dataclass
class ScanResult:
    """Aggregates collected from one feature file."""

    row_count: int
    ids: np.ndarray
    missing: np.ndarray
    count: np.ndarray
    total: np.ndarray
    total_sq: np.ndarray
    minima: np.ndarray
    maxima: np.ndarray
    low_counts: Dict[int, Counter[str]]
    unexpected: Dict[int, Counter[str]]
    unexpected_count: np.ndarray
    sample: np.ndarray
    row_hashes: Dict[bytes, List[int]]
    target_count_pos: Optional[np.ndarray] = None
    target_total_pos: Optional[np.ndarray] = None
    target_total_sq_pos: Optional[np.ndarray] = None
    target_count_neg: Optional[np.ndarray] = None
    target_total_neg: Optional[np.ndarray] = None
    target_total_sq_neg: Optional[np.ndarray] = None
    target_missing_pos: Optional[np.ndarray] = None
    target_missing_neg: Optional[np.ndarray] = None
    target_low_counts: Optional[Dict[int, Counter[str]]] = None
    target_low_positive: Optional[Dict[int, Counter[str]]] = None


def update_counter_from_values(counter: Counter[str], values: np.ndarray) -> None:
    """Add an array of numeric/missing values to a string-keyed counter."""

    finite = values[np.isfinite(values)]
    if finite.size:
        unique, counts = np.unique(finite, return_counts=True)
        for value, count in zip(unique, counts):
            counter[canonical_value(float(value))] += int(count)
    missing = int((~np.isfinite(values)).sum())
    if missing:
        counter[MISSING] += missing


def update_bounded_counter(
    counter: Counter[str], values: np.ndarray, cap: int = UNEXPECTED_VALUE_EXAMPLE_CAP
) -> None:
    """Count unexpected examples without retaining unbounded unique values."""

    if values.size == 0:
        return
    unique, counts = np.unique(values, return_counts=True)
    for value, count in zip(unique, counts):
        key = canonical_value(float(value))
        if key in counter or len(counter) < cap:
            counter[key] += int(count)
        else:
            counter["<OTHER_UNEXPECTED_VALUES>"] += int(count)


def scan_feature_file(
    path: Path,
    feature_names: Sequence[str],
    row_count: int,
    low_indices: Sequence[int],
    validators: Sequence[Sequence[Tuple[float, float]]],
    sample_indices: np.ndarray,
    chunk_size: int,
    cache_path: Path,
    ids_cache_path: Path,
    labels: Optional[np.ndarray] = None,
    development_mask: Optional[np.ndarray] = None,
) -> ScanResult:
    """Scan a feature CSV once while creating its float32 cache."""

    feature_count = len(feature_names)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.float32, shape=(row_count, feature_count)
    )
    ids_cache = np.lib.format.open_memmap(
        ids_cache_path, mode="w+", dtype=np.int64, shape=(row_count,)
    )
    missing = np.zeros(feature_count, dtype=np.int64)
    count = np.zeros(feature_count, dtype=np.int64)
    total = np.zeros(feature_count, dtype=np.float64)
    total_sq = np.zeros(feature_count, dtype=np.float64)
    minima = np.full(feature_count, math.inf, dtype=np.float64)
    maxima = np.full(feature_count, -math.inf, dtype=np.float64)
    low_counts = {index: Counter() for index in low_indices}
    unexpected = {
        index: Counter() for index, intervals in enumerate(validators) if intervals
    }
    unexpected_count = np.zeros(feature_count, dtype=np.int64)
    sample = np.empty((len(sample_indices), feature_count), dtype=np.float32)
    row_hashes: Dict[bytes, List[int]] = {}

    supervised = labels is not None and development_mask is not None
    if supervised:
        target_count_pos = np.zeros(feature_count, dtype=np.int64)
        target_total_pos = np.zeros(feature_count, dtype=np.float64)
        target_total_sq_pos = np.zeros(feature_count, dtype=np.float64)
        target_count_neg = np.zeros(feature_count, dtype=np.int64)
        target_total_neg = np.zeros(feature_count, dtype=np.float64)
        target_total_sq_neg = np.zeros(feature_count, dtype=np.float64)
        target_missing_pos = np.zeros(feature_count, dtype=np.int64)
        target_missing_neg = np.zeros(feature_count, dtype=np.int64)
        target_low_counts = {index: Counter() for index in low_indices}
        target_low_positive = {index: Counter() for index in low_indices}
    else:
        target_count_pos = target_total_pos = target_total_sq_pos = None
        target_count_neg = target_total_neg = target_total_sq_neg = None
        target_missing_pos = target_missing_neg = None
        target_low_counts = target_low_positive = None

    seen = 0
    low_set = set(low_indices)
    for start, ids, matrix, raw_rows in iter_csv_chunks(
        path, feature_names, chunk_size
    ):
        end = start + len(matrix)
        if end > row_count:
            raise ValueError(f"{path.name}: more rows than expected")
        cache[start:end] = matrix.astype(np.float32)
        ids_cache[start:end] = ids
        finite = np.isfinite(matrix)
        missing += (~finite).sum(axis=0)
        count += finite.sum(axis=0)
        total += np.nansum(matrix, axis=0)
        total_sq += np.nansum(matrix * matrix, axis=0)
        chunk_min = np.min(np.where(finite, matrix, math.inf), axis=0)
        chunk_max = np.max(np.where(finite, matrix, -math.inf), axis=0)
        minima = np.minimum(minima, chunk_min)
        maxima = np.maximum(maxima, chunk_max)

        sample_left = np.searchsorted(sample_indices, start, side="left")
        sample_right = np.searchsorted(sample_indices, end, side="left")
        if sample_right > sample_left:
            relative = sample_indices[sample_left:sample_right] - start
            sample[sample_left:sample_right] = matrix[relative].astype(np.float32)

        for index in low_set:
            update_counter_from_values(low_counts[index], matrix[:, index])

        for index, intervals in enumerate(validators):
            if not intervals:
                continue
            values = matrix[:, index]
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                continue
            allowed = allowed_numeric_mask(finite_values, intervals)
            invalid_values = finite_values[~allowed]
            if invalid_values.size:
                unexpected_count[index] += invalid_values.size
                update_bounded_counter(unexpected[index], invalid_values)

        if supervised:
            assert labels is not None and development_mask is not None
            assert target_count_pos is not None and target_count_neg is not None
            assert target_total_pos is not None and target_total_neg is not None
            assert target_total_sq_pos is not None and target_total_sq_neg is not None
            assert target_missing_pos is not None and target_missing_neg is not None
            assert target_low_counts is not None and target_low_positive is not None
            local_development = development_mask[start:end]
            local_labels = labels[start:end]
            positive = local_development & (local_labels == 1)
            negative = local_development & (local_labels == -1)
            positive_matrix = matrix[positive]
            negative_matrix = matrix[negative]
            positive_finite = np.isfinite(positive_matrix)
            negative_finite = np.isfinite(negative_matrix)
            target_count_pos += positive_finite.sum(axis=0)
            target_count_neg += negative_finite.sum(axis=0)
            target_missing_pos += (~positive_finite).sum(axis=0)
            target_missing_neg += (~negative_finite).sum(axis=0)
            target_total_pos += np.nansum(positive_matrix, axis=0)
            target_total_neg += np.nansum(negative_matrix, axis=0)
            target_total_sq_pos += np.nansum(positive_matrix * positive_matrix, axis=0)
            target_total_sq_neg += np.nansum(negative_matrix * negative_matrix, axis=0)
            for index in low_set:
                development_values = matrix[local_development, index]
                positive_values = matrix[positive, index]
                update_counter_from_values(target_low_counts[index], development_values)
                update_counter_from_values(target_low_positive[index], positive_values)

        for offset, raw_values in enumerate(raw_rows):
            digest = hashlib.blake2b(
                "\x1f".join(raw_values).encode("utf-8"), digest_size=16
            ).digest()
            row_id = int(ids[offset])
            if digest not in row_hashes:
                row_hashes[digest] = [1, row_id, 0]
            else:
                row_hashes[digest][0] += 1
            if labels is not None:
                row_hashes[digest][2] |= 1 if labels[start + offset] == -1 else 2
        seen = end

    if seen != row_count:
        raise ValueError(f"{path.name}: expected {row_count} rows, found {seen}")
    cache.flush()
    ids_cache.flush()
    minima[minima == math.inf] = np.nan
    maxima[maxima == -math.inf] = np.nan
    return ScanResult(
        row_count=row_count,
        ids=np.asarray(ids_cache),
        missing=missing,
        count=count,
        total=total,
        total_sq=total_sq,
        minima=minima,
        maxima=maxima,
        low_counts=low_counts,
        unexpected=unexpected,
        unexpected_count=unexpected_count,
        sample=sample,
        row_hashes=row_hashes,
        target_count_pos=target_count_pos,
        target_total_pos=target_total_pos,
        target_total_sq_pos=target_total_sq_pos,
        target_count_neg=target_count_neg,
        target_total_neg=target_total_neg,
        target_total_sq_neg=target_total_sq_neg,
        target_missing_pos=target_missing_pos,
        target_missing_neg=target_missing_neg,
        target_low_counts=target_low_counts,
        target_low_positive=target_low_positive,
    )


def profile_distributions(
    train: ScanResult,
    test: ScanResult,
    registry: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    development_indices: np.ndarray,
    train_sample_indices: np.ndarray,
) -> Tuple[
    List[Dict[str, object]], List[Dict[str, object]], Dict[int, Dict[str, object]]
]:
    """Build per-feature summaries and long-form target value/bin rates."""

    development_sample_mask = np.isin(train_sample_indices, development_indices)
    sample_labels = labels[train_sample_indices]
    feature_rows: List[Dict[str, object]] = []
    target_rows: List[Dict[str, object]] = []
    distribution_details: Dict[int, Dict[str, object]] = {}
    positive_development = int((labels[development_indices] == 1).sum())
    negative_development = int((labels[development_indices] == -1).sum())
    baseline_rate = positive_development / len(development_indices)

    for index, entry in enumerate(registry):
        train_mean, train_std = mean_std(
            int(train.count[index]),
            float(train.total[index]),
            float(train.total_sq[index]),
        )
        test_mean, test_std = mean_std(
            int(test.count[index]),
            float(test.total[index]),
            float(test.total_sq[index]),
        )
        mean_shift = standardized_mean_difference(
            train_mean, train_std, test_mean, test_std
        )

        assert train.target_count_pos is not None and train.target_count_neg is not None
        assert train.target_total_pos is not None and train.target_total_neg is not None
        assert (
            train.target_total_sq_pos is not None
            and train.target_total_sq_neg is not None
        )
        assert (
            train.target_missing_pos is not None
            and train.target_missing_neg is not None
        )
        positive_mean, positive_std = mean_std(
            int(train.target_count_pos[index]),
            float(train.target_total_pos[index]),
            float(train.target_total_sq_pos[index]),
        )
        negative_mean, negative_std = mean_std(
            int(train.target_count_neg[index]),
            float(train.target_total_neg[index]),
            float(train.target_total_sq_neg[index]),
        )
        target_smd = standardized_mean_difference(
            positive_mean, positive_std, negative_mean, negative_std
        )

        if index in train.low_counts:
            train_counts = train.low_counts[index]
            test_counts = test.low_counts[index]
            shift_basis = "exact_values"
            assert train.target_low_counts is not None
            assert train.target_low_positive is not None
            development_counts = train.target_low_counts[index]
            positive_counts = train.target_low_positive[index]
            negative_counts = Counter(
                {
                    key: count - positive_counts.get(key, 0)
                    for key, count in development_counts.items()
                }
            )
            target_basis = "exact_values"
            category_order = sorted(
                development_counts,
                key=lambda key: (
                    key == MISSING,
                    safe_float(key) is None,
                    safe_float(key) or 0,
                ),
            )
            for value in category_order:
                count = development_counts[value]
                positive = positive_counts.get(value, 0)
                target_rows.append(
                    {
                        "feature": entry["name"],
                        "semantic_type": entry["semantic_type"],
                        "basis": target_basis,
                        "value_or_bin": value,
                        "development_count": count,
                        "positive_count": positive,
                        "negative_count": count - positive,
                        "positive_rate": positive / count if count else None,
                        "development_share": count / len(development_indices),
                    }
                )
        else:
            edges = stable_quantile_edges(train.sample[:, index])
            train_counts = quantile_bin_counts(train.sample[:, index], edges)
            test_counts = quantile_bin_counts(test.sample[:, index], edges)
            shift_basis = "sample_quantile_bins"
            development_values = train.sample[development_sample_mask, index]
            development_labels_sample = sample_labels[development_sample_mask]
            development_counts = quantile_bin_counts(development_values, edges)
            positive_counts = quantile_bin_counts(
                development_values[development_labels_sample == 1], edges
            )
            negative_counts = quantile_bin_counts(
                development_values[development_labels_sample == -1], edges
            )
            target_basis = "sample_quantile_bins"
            category_order = list(development_counts)
            for value in category_order:
                count = development_counts[value]
                positive = positive_counts.get(value, 0)
                target_rows.append(
                    {
                        "feature": entry["name"],
                        "semantic_type": entry["semantic_type"],
                        "basis": target_basis,
                        "value_or_bin": value,
                        "development_count": count,
                        "positive_count": positive,
                        "negative_count": count - positive,
                        "positive_rate": positive / count if count else None,
                        "development_share": count / max(1, len(development_values)),
                    }
                )

        shift_tvd = total_variation(train_counts, test_counts)
        target_tvd = total_variation(positive_counts, negative_counts)
        minimum_support = max(100, int(math.ceil(0.001 * len(development_indices))))
        supported_rates = [
            positive_counts.get(key, 0) / count
            for key, count in development_counts.items()
            if count >= minimum_support
        ]
        max_rate_lift = (
            max(abs(rate - baseline_rate) for rate in supported_rates)
            if supported_rates
            else None
        )
        train_missing_fraction = train.missing[index] / train.row_count
        test_missing_fraction = test.missing[index] / test.row_count
        positive_missing_fraction = (
            train.target_missing_pos[index] / positive_development
        )
        negative_missing_fraction = (
            train.target_missing_neg[index] / negative_development
        )
        raw_outside_count = int(train.unexpected_count[index])
        raw_outside_fraction = (
            raw_outside_count / train.count[index] if train.count[index] else 0.0
        )
        if index not in train.unexpected:
            validation_status = "not_machine_checkable"
        elif raw_outside_fraction > 0.05:
            validation_status = "documented_domain_incomplete"
        else:
            validation_status = "checked"
        reported_unexpected_count = (
            raw_outside_count if validation_status == "checked" else 0
        )
        unexpected_preview = train.unexpected.get(index, Counter()).most_common(10)
        observed_distinct = (
            len(train.low_counts[index])
            if index in train.low_counts
            else entry["train_unique_count"]
        )
        observed_constant = (
            len(train.low_counts[index]) == 1
            if index in train.low_counts
            else train.missing[index] == 0
            and train.minima[index] == train.maxima[index]
        )
        observed_constant_when_present = (
            train.count[index] > 0 and train.minima[index] == train.maxima[index]
        )
        feature_rows.append(
            {
                "position": index + 1,
                "name": entry["name"],
                "title": entry["title"],
                "semantic_type": entry["semantic_type"],
                "source_kind": entry["source_kind"],
                "train_missing_fraction": train_missing_fraction,
                "test_missing_fraction": test_missing_fraction,
                "missing_fraction_difference": abs(
                    train_missing_fraction - test_missing_fraction
                ),
                "train_unique_count": observed_distinct,
                "train_min": (
                    None if math.isnan(train.minima[index]) else train.minima[index]
                ),
                "train_max": (
                    None if math.isnan(train.maxima[index]) else train.maxima[index]
                ),
                "train_mean": train_mean,
                "train_std": train_std,
                "test_mean": test_mean,
                "test_std": test_std,
                "train_test_standardized_mean_difference": mean_shift,
                "train_test_tvd": shift_tvd,
                "train_test_tvd_basis": shift_basis,
                "constant_including_missing": observed_constant,
                "constant_when_present": observed_constant_when_present,
                "exact_duplicate_of": entry.get("exact_duplicate_of", ""),
                "codebook_validation_available": validation_status == "checked",
                "codebook_validation_status": validation_status,
                "codebook_outside_count_train": raw_outside_count,
                "codebook_outside_fraction_train": raw_outside_fraction,
                "unexpected_nonblank_count_train": reported_unexpected_count,
                "unexpected_nonblank_fraction_train": (
                    reported_unexpected_count / train.count[index]
                    if train.count[index] and validation_status == "checked"
                    else 0.0
                ),
                "unexpected_values_preview": json.dumps(unexpected_preview),
                "positive_missing_fraction_development": positive_missing_fraction,
                "negative_missing_fraction_development": negative_missing_fraction,
                "missing_target_gap": abs(
                    positive_missing_fraction - negative_missing_fraction
                ),
                "positive_mean_development": positive_mean,
                "negative_mean_development": negative_mean,
                "target_standardized_mean_difference": target_smd,
                "target_tvd": target_tvd,
                "target_tvd_basis": target_basis,
                "max_supported_positive_rate_lift": max_rate_lift,
            }
        )
        distribution_details[index] = {
            "train": train_counts,
            "test": test_counts,
            "development": development_counts,
            "positive": positive_counts,
            "negative": negative_counts,
        }
    return feature_rows, target_rows, distribution_details


def duplicate_row_summary(train: ScanResult, test: ScanResult) -> Dict[str, object]:
    """Summarize within-file and train/test feature-row duplicates."""

    train_duplicates = {
        key: value for key, value in train.row_hashes.items() if value[0] > 1
    }
    test_duplicates = {
        key: value for key, value in test.row_hashes.items() if value[0] > 1
    }
    overlap = set(train.row_hashes) & set(test.row_hashes)
    conflict_groups = sum(value[2] == 3 for value in train_duplicates.values())
    return {
        "hash_algorithm": "BLAKE2b-128 over all 321 raw feature fields; Id excluded",
        "train_duplicate_groups": len(train_duplicates),
        "train_duplicate_extra_rows": sum(
            value[0] - 1 for value in train_duplicates.values()
        ),
        "train_duplicate_groups_with_conflicting_labels": conflict_groups,
        "test_duplicate_groups": len(test_duplicates),
        "test_duplicate_extra_rows": sum(
            value[0] - 1 for value in test_duplicates.values()
        ),
        "unique_feature_rows_shared_by_train_and_test": len(overlap),
        "test_rows_matching_a_train_feature_row": sum(
            test.row_hashes[key][0] for key in overlap
        ),
        "examples_train": [
            {"first_id": value[1], "row_count": value[0], "label_mask": value[2]}
            for value in sorted(
                train_duplicates.values(), key=lambda item: item[0], reverse=True
            )[:20]
        ],
    }


def write_csv(
    path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> None:
    """Write a reviewable CSV with stable columns."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def setup_plot_style() -> None:
    """Set restrained, readable defaults for all generated figures."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save and close one Matplotlib figure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_target(labels: np.ndarray, path: Path) -> None:
    counts = [int((labels == -1).sum()), int((labels == 1).sum())]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(
        ["No reported MI/CHD (-1)", "Reported MI/CHD (+1)"],
        counts,
        color=["#4C78A8", "#E45756"],
    )
    ax.set_title("Target distribution")
    ax.set_ylabel("Respondents")
    ax.set_ylim(0, max(counts) * 1.13)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count,
            f"{count:,}\n({count / len(labels):.2%})",
            ha="center",
            va="bottom",
        )
    save_figure(fig, path)


def plot_missingness(feature_rows: Sequence[Mapping[str, object]], path: Path) -> None:
    fractions = np.asarray(
        [row["train_missing_fraction"] for row in feature_rows], dtype=float
    )
    top = sorted(
        feature_rows, key=lambda row: row["train_missing_fraction"], reverse=True
    )[:20]
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1.35]}
    )
    axes[0].hist(
        fractions, bins=np.linspace(0, 1, 21), color="#4C78A8", edgecolor="white"
    )
    axes[0].set_title("Missingness across features")
    axes[0].set_xlabel("Missing fraction in train")
    axes[0].set_ylabel("Number of features")
    names = [row["name"] for row in reversed(top)]
    values = [row["train_missing_fraction"] for row in reversed(top)]
    axes[1].barh(names, values, color="#72B7B2")
    axes[1].set_title("Features with most missing values")
    axes[1].set_xlabel("Missing fraction in train")
    axes[1].set_xlim(0, 1.08)
    axes[1].grid(axis="x", alpha=0.25)
    for position, value in enumerate(values):
        axes[1].text(value + 0.006, position, f"{value:.2%}", va="center", fontsize=8)
    fig.tight_layout()
    save_figure(fig, path)


def plot_ranked(
    feature_rows: Sequence[Mapping[str, object]],
    field: str,
    title: str,
    xlabel: str,
    path: Path,
    color: str,
) -> None:
    top = sorted(feature_rows, key=lambda row: float(row[field] or 0.0), reverse=True)[
        :20
    ]
    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    names = [row["name"] for row in reversed(top)]
    values = [float(row[field] or 0.0) for row in reversed(top)]
    ax.barh(names, values, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, path)


def plot_cardinality(registry: Sequence[Mapping[str, object]], path: Path) -> None:
    labels = ["1–2", "3–5", "6–10", "11–50", "51–256", ">256"]
    values = [0] * len(labels)
    for entry in registry:
        count = entry["train_unique_count"]
        if not isinstance(count, int):
            values[-1] += 1
        elif count <= 2:
            values[0] += 1
        elif count <= 5:
            values[1] += 1
        elif count <= 10:
            values[2] += 1
        elif count <= 50:
            values[3] += 1
        elif count <= 256:
            values[4] += 1
        else:
            values[5] += 1
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    bars = ax.bar(labels, values, color="#59A14F")
    ax.set_title("Feature cardinality")
    ax.set_xlabel("Distinct raw values in train")
    ax.set_ylabel("Number of features")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            str(value),
            ha="center",
            va="bottom",
        )
    save_figure(fig, path)


def plot_missingness_heatmap(
    sample: np.ndarray,
    feature_rows: Sequence[Mapping[str, object]],
    seed: int,
    path: Path,
) -> None:
    top_indices = [
        int(row["position"]) - 1
        for row in sorted(
            feature_rows, key=lambda row: row["train_missing_fraction"], reverse=True
        )[:60]
    ]
    rng = np.random.default_rng(seed)
    row_count = min(250, len(sample))
    selected_rows = np.sort(rng.choice(len(sample), size=row_count, replace=False))
    matrix = np.isnan(sample[selected_rows][:, top_indices])
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.imshow(
        matrix, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1
    )
    ax.set_title(
        "Missingness patterns: 250 sampled respondents × 60 most-missing features\n"
        "dark = missing, white = observed"
    )
    ax.set_xlabel("Features ordered by missing fraction")
    ax.set_ylabel("Sampled respondents")
    ax.set_xticks([])
    save_figure(fig, path)


def plot_distribution_panels(
    feature_rows: Sequence[Mapping[str, object]],
    details: Mapping[int, Mapping[str, Counter[str]]],
    registry: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    top = sorted(feature_rows, key=lambda row: row["train_test_tvd"], reverse=True)[:6]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for ax, row in zip(axes.flat, top):
        index = int(row["position"]) - 1
        train_counts = details[index]["train"]
        test_counts = details[index]["test"]
        keys = sorted(
            set(train_counts) | set(test_counts),
            key=lambda key: train_counts.get(key, 0),
            reverse=True,
        )[:10]
        train_total = sum(train_counts.values())
        test_total = sum(test_counts.values())
        positions = np.arange(len(keys))
        width = 0.4
        ax.bar(
            positions - width / 2,
            [train_counts.get(key, 0) / train_total for key in keys],
            width,
            label="train",
            color="#4C78A8",
        )
        ax.bar(
            positions + width / 2,
            [test_counts.get(key, 0) / test_total for key in keys],
            width,
            label="test",
            color="#F28E2B",
        )
        ax.set_title(f"{row['name']}  TVD={row['train_test_tvd']:.4f}")
        ax.set_xticks(positions, keys, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Fraction")
        ax.grid(axis="y", alpha=0.2)
    axes.flat[0].legend(frameon=False)
    fig.suptitle("Largest train–test distribution differences", y=1.01, fontsize=14)
    fig.tight_layout()
    save_figure(fig, path)


def plot_target_profiles(
    feature_rows: Sequence[Mapping[str, object]],
    details: Mapping[int, Mapping[str, Counter[str]]],
    path: Path,
) -> None:
    top = sorted(feature_rows, key=lambda row: row["target_tvd"], reverse=True)[:6]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for ax, row in zip(axes.flat, top):
        index = int(row["position"]) - 1
        development = details[index]["development"]
        positive = details[index]["positive"]
        keys = sorted(development, key=lambda key: development[key], reverse=True)[:10]
        rates = [positive.get(key, 0) / development[key] for key in keys]
        positions = np.arange(len(keys))
        ax.bar(positions, rates, color="#E45756")
        ax.set_title(f"{row['name']}  target TVD={row['target_tvd']:.4f}")
        ax.set_xticks(positions, keys, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Positive rate")
        ax.set_ylim(0, min(1.0, max(rates, default=0.1) * 1.15))
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Target profiles of the strongest univariate associations", y=1.01, fontsize=14
    )
    fig.tight_layout()
    save_figure(fig, path)


def markdown_table(
    rows: Sequence[Mapping[str, object]], field: str, digits: int = 4
) -> List[str]:
    lines = ["| Feature | Value | Type |", "| --- | ---: | --- |"]
    for row in rows:
        lines.append(
            f"| `{row['name']}` | {float(row[field]):.{digits}f} | `{row['semantic_type']}` |"
        )
    return lines


def write_summary(
    path: Path,
    feature_rows: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    split_summary: Mapping[str, object],
    duplicates: Mapping[str, object],
    figure_names: Sequence[str],
) -> None:
    top_shift = sorted(
        feature_rows, key=lambda row: row["train_test_tvd"], reverse=True
    )[:10]
    top_target = sorted(feature_rows, key=lambda row: row["target_tvd"], reverse=True)[
        :10
    ]
    unexpected = sorted(
        [row for row in feature_rows if row["unexpected_nonblank_count_train"]],
        key=lambda row: row["unexpected_nonblank_count_train"],
        reverse=True,
    )
    incomplete_domains = sorted(
        [
            row
            for row in feature_rows
            if row["codebook_validation_status"] == "documented_domain_incomplete"
        ],
        key=lambda row: row["codebook_outside_count_train"],
        reverse=True,
    )
    constants = [row for row in feature_rows if row["constant_including_missing"]]
    lines = [
        "# Dataset statistical analysis",
        "",
        "This report extends the structural audit and feature registry. All supplied predictor columns remain available for competition modelling; flags below describe data behaviour rather than automatic exclusions.",
        "",
        "## Target and split",
        "",
        f"- Rows: {len(labels):,}.",
        f"- Positive target: {int((labels == 1).sum()):,} ({(labels == 1).mean():.4%}).",
        f"- Development rows: {split_summary['development_rows']:,}; validation rows: {split_summary['validation_rows']:,}.",
        f"- Split seed: `{split_summary['seed']}`; validation fraction: {split_summary['validation_fraction']:.1%}.",
        "- Target-aware statistics use development rows only. Validation labels were not used in feature analysis.",
        "",
        "## Structural results",
        "",
        f"- Constant columns including missing as a state: {len(constants)}.",
        f"- Exact duplicate-column annotations already in the registry: {sum(bool(row['exact_duplicate_of']) for row in feature_rows)}.",
        f"- Duplicate train feature-row groups: {duplicates['train_duplicate_groups']:,}; conflicting-label groups: {duplicates['train_duplicate_groups_with_conflicting_labels']:,}.",
        f"- Unique feature rows shared by train and test: {duplicates['unique_feature_rows_shared_by_train_and_test']:,}.",
        "",
        "## Largest train-test distribution differences",
        "",
        *markdown_table(top_shift, "train_test_tvd"),
        "",
        "TVD is total variation distance. It is exact for features with at most 256 observed values and estimated from a fixed sample with train-defined quantile bins for higher-cardinality features.",
        "",
        "## Strongest univariate target associations",
        "",
        *markdown_table(top_target, "target_tvd"),
        "",
        "Target TVD compares the feature distribution between positive and negative development respondents. A high value indicates association, not causality or independent predictive value.",
        "",
        "## Codebook validation",
        "",
        f"- Features with at least one nonblank value outside conservatively parsed codebook intervals: {len(unexpected)}.",
    ]
    if unexpected:
        lines.extend(
            f"- `{row['name']}`: {row['unexpected_nonblank_count_train']:,} rows; examples {row['unexpected_values_preview']}."
            for row in unexpected[:20]
        )
    else:
        lines.append(
            "- No nonblank out-of-codebook values were found by the conservative numeric validator."
        )
    if incomplete_domains:
        lines.append(
            f"- Machine validation was withheld for {len(incomplete_domains)} feature(s) because the parsed documented domain covered less than 95% of observed nonblank values."
        )
        lines.extend(
            f"- `{row['name']}`: parsed-domain mismatch in {row['codebook_outside_count_train']:,} rows; this is a codebook parsing/documentation gap, not an automatic data error."
            for row in incomplete_domains
        )
    lines.extend(["", "## Figures", ""])
    lines.extend(f"- `figures/{name}`" for name in figure_names)
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- The target records whether MI or CHD was ever reported; it is not a future-event or recurrent-infarction label.",
            "- Test-set comparisons use predictors only. No test labels are available or inferred.",
            "- High-cardinality TVD and its target profile are sample estimates; exact means, standard deviations, missingness, and low-cardinality frequencies use all applicable rows.",
            "- Codebook validation is deliberately conservative. Features without machine-readable numeric codes are marked as not checkable rather than declared valid.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(arguments: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("dataset_audit", type=Path)
    parser.add_argument("feature_registry", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--chunk-size", type=int, default=2_048)
    return parser.parse_args(arguments)


def main(arguments: Optional[Sequence[str]] = None) -> None:
    args = parse_args(arguments)
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    audit = json.loads(args.dataset_audit.read_text(encoding="utf-8"))
    registry_with_id = json.loads(args.feature_registry.read_text(encoding="utf-8"))
    registry = [entry for entry in registry_with_id if entry["name"] != "Id"]
    feature_names = [entry["name"] for entry in registry]
    train_rows = int(audit["x_train"]["row_count"])
    test_rows = int(audit["x_test"]["row_count"])

    label_ids, labels = load_labels(data_dir / "y_train.csv")
    if len(labels) != train_rows:
        raise ValueError("Audit and y_train row counts differ")
    development_indices, validation_indices = stratified_split(
        labels, args.validation_fraction, args.seed
    )
    development_mask = np.zeros(train_rows, dtype=bool)
    development_mask[development_indices] = True
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        split_dir / f"stratified_seed_{args.seed}.npz",
        development_indices=development_indices,
        validation_indices=validation_indices,
    )
    split_summary = {
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "development_rows": len(development_indices),
        "validation_rows": len(validation_indices),
        "development_positive": int((labels[development_indices] == 1).sum()),
        "validation_positive": int((labels[validation_indices] == 1).sum()),
        "development_indices_sha256": sha256_array(development_indices),
        "validation_indices_sha256": sha256_array(validation_indices),
    }
    (split_dir / f"stratified_seed_{args.seed}.json").write_text(
        json.dumps(split_summary, indent=2), encoding="utf-8"
    )

    low_indices = [
        index
        for index, entry in enumerate(registry)
        if isinstance(entry["train_unique_count"], int)
        and entry["train_unique_count"] <= LOW_CARDINALITY_CAP
    ]
    validators = [validation_intervals(entry) for entry in registry]
    rng = np.random.default_rng(args.seed)
    train_sample_indices = np.sort(
        rng.choice(train_rows, size=min(args.sample_size, train_rows), replace=False)
    )
    test_sample_indices = np.sort(
        rng.choice(test_rows, size=min(args.sample_size, test_rows), replace=False)
    )

    print("Scanning x_train.csv and creating float32 cache...")
    train = scan_feature_file(
        data_dir / "x_train.csv",
        feature_names,
        train_rows,
        low_indices,
        validators,
        train_sample_indices,
        args.chunk_size,
        cache_dir / "x_train_float32.npy",
        cache_dir / "id_train_int64.npy",
        labels=labels,
        development_mask=development_mask,
    )
    if not np.array_equal(train.ids, label_ids):
        raise ValueError("x_train and y_train IDs differ or are out of order")

    print("Scanning x_test.csv and creating float32 cache...")
    test = scan_feature_file(
        data_dir / "x_test.csv",
        feature_names,
        test_rows,
        low_indices,
        validators,
        test_sample_indices,
        args.chunk_size,
        cache_dir / "x_test_float32.npy",
        cache_dir / "id_test_int64.npy",
    )
    np.save(cache_dir / "y_train_int8.npy", labels)

    feature_rows, target_rows, details = profile_distributions(
        train,
        test,
        registry,
        labels,
        development_indices,
        train_sample_indices,
    )
    duplicates = duplicate_row_summary(train, test)

    feature_fields = list(feature_rows[0])
    write_csv(output_dir / "feature_analysis.csv", feature_rows, feature_fields)
    target_fields = list(target_rows[0])
    write_csv(output_dir / "target_value_rates.csv", target_rows, target_fields)
    unexpected_rows = []
    validation_gap_rows = []
    for index, counter in train.unexpected.items():
        status = feature_rows[index]["codebook_validation_status"]
        for value, count in counter.most_common():
            destination = (
                unexpected_rows if status == "checked" else validation_gap_rows
            )
            destination.append(
                {
                    "feature": registry[index]["name"],
                    "semantic_type": registry[index]["semantic_type"],
                    "validation_status": status,
                    "value": value,
                    "train_count": count,
                    "train_fraction_among_present": count / max(1, train.count[index]),
                }
            )
    write_csv(
        output_dir / "unexpected_values.csv",
        unexpected_rows,
        [
            "feature",
            "semantic_type",
            "validation_status",
            "value",
            "train_count",
            "train_fraction_among_present",
        ],
    )
    write_csv(
        output_dir / "codebook_validation_gaps.csv",
        validation_gap_rows,
        [
            "feature",
            "semantic_type",
            "validation_status",
            "value",
            "train_count",
            "train_fraction_among_present",
        ],
    )
    (output_dir / "duplicate_rows.json").write_text(
        json.dumps(duplicates, indent=2), encoding="utf-8"
    )

    duplicate_columns_test = []
    test_cache = np.load(cache_dir / "x_test_float32.npy", mmap_mode="r")
    name_to_index = {name: index for index, name in enumerate(feature_names)}
    for entry in registry:
        reference = entry.get("exact_duplicate_of", "")
        if reference:
            left = name_to_index[entry["name"]]
            right = name_to_index[reference]
            duplicate_columns_test.append(
                {
                    "feature": entry["name"],
                    "duplicate_of": reference,
                    "equal_in_test": bool(
                        np.array_equal(
                            test_cache[:, left], test_cache[:, right], equal_nan=True
                        )
                    ),
                }
            )
    write_csv(
        output_dir / "duplicate_columns_test.csv",
        duplicate_columns_test,
        ["feature", "duplicate_of", "equal_in_test"],
    )

    setup_plot_style()
    figure_dir = output_dir / "figures"
    figures = [
        "01_target_distribution.png",
        "02_missingness_overview.png",
        "03_train_test_shift.png",
        "04_target_association.png",
        "05_cardinality_distribution.png",
        "06_missingness_heatmap.png",
        "07_top_shift_distributions.png",
        "08_top_target_profiles.png",
    ]
    plot_target(labels, figure_dir / figures[0])
    plot_missingness(feature_rows, figure_dir / figures[1])
    plot_ranked(
        feature_rows,
        "train_test_tvd",
        "Largest train–test distribution differences",
        "Total variation distance",
        figure_dir / figures[2],
        "#F28E2B",
    )
    plot_ranked(
        feature_rows,
        "target_tvd",
        "Strongest univariate target associations",
        "Total variation distance: positive vs negative",
        figure_dir / figures[3],
        "#E45756",
    )
    plot_cardinality(registry, figure_dir / figures[4])
    plot_missingness_heatmap(
        train.sample, feature_rows, args.seed, figure_dir / figures[5]
    )
    plot_distribution_panels(feature_rows, details, registry, figure_dir / figures[6])
    plot_target_profiles(feature_rows, details, figure_dir / figures[7])

    analysis_json = {
        "method": {
            "seed": args.seed,
            "sample_size_train": len(train_sample_indices),
            "sample_size_test": len(test_sample_indices),
            "low_cardinality_cap": LOW_CARDINALITY_CAP,
            "target_statistics_scope": "development split only",
            "cache_dtype": "float32 with NaN for blank",
        },
        "split": split_summary,
        "duplicates": duplicates,
        "feature_count": len(feature_rows),
        "features_with_unexpected_nonblank_values": sum(
            bool(row["unexpected_nonblank_count_train"]) for row in feature_rows
        ),
        "figures": figures,
        "cache_files": [
            "x_train_float32.npy",
            "x_test_float32.npy",
            "id_train_int64.npy",
            "id_test_int64.npy",
            "y_train_int8.npy",
        ],
    }
    (output_dir / "dataset_analysis.json").write_text(
        json.dumps(analysis_json, indent=2), encoding="utf-8"
    )
    write_summary(
        output_dir / "analysis_summary.md",
        feature_rows,
        labels,
        split_summary,
        duplicates,
        figures,
    )
    print(f"Wrote analysis to {output_dir}")


if __name__ == "__main__":
    main()

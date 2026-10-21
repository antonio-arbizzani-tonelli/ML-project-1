"""Fixed, label-independent BRFSS response counts for the A/B candidates.

The six source questions document raw code 1 as Yes and 2 as No. Unknown,
refused, blank and other invalid finite codes contribute to neither count.
Counts are derived from original responses, so conditional imputation cannot
turn an unanswered question into an observed answer. No training state is fit.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


GROUPS = {
    "A": ("BLIND", "DECIDE", "USEEQUIP"),
    "B": ("ADDEPEV2", "CHCCOPD1", "CHCKIDNY"),
}
COUNT_NAMES = {
    "A": ("FUNCTIONAL__yes_count", "FUNCTIONAL__answered_count"),
    "B": ("DIAGNOSES__yes_count", "DIAGNOSES__answered_count"),
}


def normalize_groups(groups: Sequence[str] | str) -> tuple[str, ...]:
    """Validate the requested groups and return their fixed A-then-B order."""
    requested = [groups] if isinstance(groups, str) else list(groups)
    expanded = []
    for group in requested:
        if group == "AB":
            expanded.extend(("A", "B"))
        elif isinstance(group, str) and group in GROUPS:
            expanded.append(group)
        else:
            raise ValueError("Count groups must be A, B or AB.")
    if not expanded or len(set(expanded)) != len(expanded):
        raise ValueError("Count groups must be non-empty and unique.")
    return tuple(group for group in GROUPS if group in expanded)


def _names(names: Sequence[str], columns: int, description: str) -> list[str]:
    result = list(names)
    if (len(result) != columns
            or any(not isinstance(name, str) or not name for name in result)
            or len(set(result)) != len(result)):
        raise ValueError(f"{description} must align with unique, non-empty feature names.")
    return result


def append_count_features(
    base_matrix: np.ndarray,
    output_names: Sequence[str],
    raw_features: np.ndarray,
    raw_names: Sequence[str],
    groups: Sequence[str] | str,
) -> tuple[np.ndarray, list[str]]:
    """Append yes/answered counts while retaining every existing base value.

    ``base_matrix`` is the already cleaned, floating-point tree matrix;
    ``raw_features`` contains the original response codes for the same rows.
    For each selected group, the yes count is NaN if no response is valid and
    otherwise ranges from zero to three. The answered count always ranges
    from zero to three. Existing source columns and missingness flags remain
    present. Invalid finite codes are unanswered; infinite inputs are rejected.
    """
    base = np.asarray(base_matrix)
    raw = np.asarray(raw_features)
    if base.ndim != 2 or raw.ndim != 2 or base.shape[0] != raw.shape[0]:
        raise ValueError("Base and raw feature matrices must have aligned rows and two dimensions.")
    if base.dtype.kind != "f" or raw.dtype.kind not in "ifu":
        raise ValueError("The base matrix must be floating-point and raw responses must be real numbers.")
    if np.any(np.isinf(base)) or np.any(np.isinf(raw)):
        raise ValueError("Count feature inputs must not contain infinite values.")
    existing_names = _names(output_names, base.shape[1], "Base columns")
    source_names = _names(raw_names, raw.shape[1], "Raw columns")
    selected = normalize_groups(groups)
    added_names = [name for group in selected for name in COUNT_NAMES[group]]
    if set(existing_names) & set(added_names):
        raise ValueError("Count output names must not duplicate existing feature names.")
    source_indices = {name: index for index, name in enumerate(source_names)}
    required = [name for group in selected for name in GROUPS[group]]
    missing = [name for name in required if name not in source_indices]
    if missing:
        raise ValueError("Missing count source features: " + ", ".join(missing) + ".")

    added = np.empty((len(base), len(added_names)), dtype=base.dtype)
    for offset, group in enumerate(selected):
        values = raw[:, [source_indices[name] for name in GROUPS[group]]]
        yes = np.sum(values == 1, axis=1)
        answered = np.sum((values == 1) | (values == 2), axis=1)
        added[:, 2 * offset] = yes
        added[answered == 0, 2 * offset] = np.nan
        added[:, 2 * offset + 1] = answered
    return np.concatenate((base, added), axis=1), existing_names + added_names

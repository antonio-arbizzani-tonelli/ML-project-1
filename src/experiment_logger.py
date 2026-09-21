"""Persist reproducible records for model-comparison experiments.

The logger deliberately records facts, not conclusions: the fully resolved
configuration supplied by the runner and the measured outcome.  This makes a
result auditable even when later experiments introduce new preprocessing or
model ideas.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 1
REQUIRED_CONFIG_KEYS = frozenset(
    {
        "experiment_name",
        "hypothesis",
        "data",
        "features",
        "missing_values",
        "split",
        "model",
        "threshold",
    }
)
INDEX_FIELDS = (
    "run_id",
    "recorded_at_utc",
    "experiment_name",
    "model_name",
    "split_id",
    "status",
    "threshold",
    "f1",
    "accuracy",
    "precision",
    "recall",
    "runtime_seconds",
    "estimated_memory_mebibytes",
    "record_path",
    "record_sha256",
)


def _json_default(value: Any) -> Any:
    """Convert NumPy scalar-like values while rejecting opaque objects."""

    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot store {type(value).__name__} in an experiment record.")


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject incomplete or non-serializable experiment configurations."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping.")
    missing = REQUIRED_CONFIG_KEYS - set(config)
    if missing:
        raise ValueError(
            f"config is missing required keys: {', '.join(sorted(missing))}"
        )
    if (
        not isinstance(config["experiment_name"], str)
        or not config["experiment_name"].strip()
    ):
        raise ValueError("config.experiment_name must be a non-empty string.")
    for key in ("data", "features", "missing_values", "split", "model", "threshold"):
        if not isinstance(config[key], Mapping):
            raise ValueError(f"config.{key} must be a mapping.")
    json.dumps(config, default=_json_default, allow_nan=False)


def _slug(value: str) -> str:
    """Produce a readable file-name component from an experiment name."""

    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "experiment"


def _unique_run_id(experiment_name: str, output_dir: Path) -> str:
    """Return a collision-free UTC identifier without relying on a random seed."""

    base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    prefix = f"{base}_{_slug(experiment_name)}"
    candidate, suffix = prefix, 1
    while (output_dir / f"{candidate}.json").exists():
        suffix += 1
        candidate = f"{prefix}_{suffix}"
    return candidate


def _write_json_atomically(path: Path, record: Mapping[str, Any]) -> None:
    """Write a complete record before replacing its final path."""

    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            record,
            handle,
            indent=2,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(temporary, path)


def _append_index(path: Path, row: Mapping[str, Any]) -> None:
    """Append one compact, human-sortable view of a completed record."""

    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in INDEX_FIELDS})


def save_experiment(
    config: Mapping[str, Any], outcome: Mapping[str, Any], output_dir: Path | str
) -> Path:
    """Save one completed run and return its JSON path.

    ``outcome`` must contain measured values from the caller.  No model choice,
    threshold choice, or metric is recomputed here, which keeps this module
    reusable by every future runner.
    """

    validate_config(config)
    if not isinstance(outcome, Mapping):
        raise TypeError("outcome must be a mapping.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = _unique_run_id(config["experiment_name"], output_dir)
    recorded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "recorded_at_utc": recorded_at,
        "config": dict(config),
        "outcome": dict(outcome),
    }
    record_path = output_dir / f"{run_id}.json"
    _write_json_atomically(record_path, record)
    record_bytes = record_path.read_bytes()
    metrics = outcome.get("metrics", {})
    memory = outcome.get("memory_estimate", {})
    threshold = config["threshold"].get("value")
    _append_index(
        output_dir / "index.csv",
        {
            "run_id": run_id,
            "recorded_at_utc": recorded_at,
            "experiment_name": config["experiment_name"],
            "model_name": config["model"].get("name", ""),
            "split_id": config["split"].get("id", ""),
            "status": outcome.get("status", "completed"),
            "threshold": "" if threshold is None else threshold,
            "f1": metrics.get("f1", ""),
            "accuracy": metrics.get("accuracy", ""),
            "precision": metrics.get("precision", ""),
            "recall": metrics.get("recall", ""),
            "runtime_seconds": outcome.get("runtime_seconds", ""),
            "estimated_memory_mebibytes": memory.get("arrays_mebibytes", ""),
            "record_path": record_path.name,
            "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        },
    )
    return record_path

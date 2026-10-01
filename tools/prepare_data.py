"""Prepare official CSV caches for CV without training a model.

Run from the repository root with python -m tools.prepare_data.
Use a new cache directory when changing any feature CSV: existing feature
caches are checked for dimensions and IDs, not for changes to every value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run import (
    _ensure_feature_cache,
    _ensure_label_cache,
    _read_sample_submission_ids,
)


def prepare(data_dir: Path, cache_dir: Path) -> dict:
    """Validate official ID alignment and prepare the five training/test caches."""
    train_path, train_ids_path, names = _ensure_feature_cache(
        data_dir / "x_train.csv", cache_dir, "train"
    )
    train_ids = np.load(train_ids_path, mmap_mode="r")
    label_path = _ensure_label_cache(data_dir / "y_train.csv", cache_dir, train_ids)
    test_path, test_ids_path, _ = _ensure_feature_cache(
        data_dir / "x_test.csv", cache_dir, "test", names
    )
    test_ids = np.load(test_ids_path, mmap_mode="r")
    sample_ids = _read_sample_submission_ids(data_dir / "sample_submission.csv")
    if not np.array_equal(test_ids, sample_ids):
        raise ValueError("x_test.csv and sample_submission.csv IDs do not align.")
    return {
        "training_rows": int(train_ids.size),
        "test_rows": int(test_ids.size),
        "raw_features": len(names),
        "cache_files": [str(path) for path in (
            train_path, train_ids_path, label_path, test_path, test_ids_path
        )],
    }


def main() -> None:
    """Prepare caches with the same loaders used by the final submission runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/dataset"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_dir, args.cache_dir), indent=2))


if __name__ == "__main__":
    main()

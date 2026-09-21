"""List observed response codes whose imported codebook labels need review.

This read-only audit does not recode data. A listed code is a candidate for
manual checking against the staged official codebook, not an automatic error.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.analyze_errors import _load_feature_names
from src.feature_preprocessing import FeaturePreprocessor, NUMERIC_SEMANTIC_TYPES

REVIEW_CODES = frozenset({7.0, 9.0, 77.0, 99.0, 777.0, 999.0})


def audit(root: Path) -> list[dict]:
    metadata = json.loads(
        (root / "configs/feature_metadata.json").read_text(encoding="utf-8")
    )
    variants = json.loads(
        (root / "configs/experiments/phase3_preprocessing_cv.json").read_text(
            encoding="utf-8"
        )
    )["variants"]
    plan = next(
        item["preprocessing"]
        for item in variants
        if item["experiment_name"] == "phase3_p2_codebook_clean_cv"
    )
    names = _load_feature_names(root / "data/raw/dataset/x_train.csv")
    processor = FeaturePreprocessor(names, metadata, plan)
    retained = set(processor._retained_names())
    feature_indices = {name: index for index, name in enumerate(names)}
    features = np.load(root / "data/processed/x_train_float32.npy", mmap_mode="r")
    with np.load(
        root / "results/eda/analysis/splits/stratified_seed_20260918.npz"
    ) as splits:
        development = np.asarray(splits["development_indices"], dtype=np.int64)
    rows = []
    for entry in metadata:
        name = entry["name"]
        if (
            name not in retained
            or entry.get("semantic_type") not in NUMERIC_SEMANTIC_TYPES
        ):
            continue
        mapped_missing = processor._missing_codes(name, entry)
        for item in entry.get("documented_values", []):
            if item.get("label") != "Documented value (label omitted)":
                continue
            try:
                code = float(item["code"])
            except (TypeError, ValueError):
                continue
            if code not in REVIEW_CODES or code in mapped_missing:
                continue
            observed = int(
                np.count_nonzero(features[development, feature_indices[name]] == code)
            )
            if observed:
                rows.append(
                    {
                        "feature": name,
                        "semantic_type": entry["semantic_type"],
                        "code": int(code),
                        "development_rows": observed,
                        "description": entry.get("description", ""),
                        "status": "codebook meaning requires manual verification",
                    }
                )
    return sorted(
        rows, key=lambda row: (-row["development_rows"], row["feature"], row["code"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/eda/analysis/phase13_feature_code_review_candidates.csv"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    rows = audit(root)
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "feature",
                "semantic_type",
                "code",
                "development_rows",
                "description",
                "status",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Listed {len(rows)} observed code/feature pairs for manual codebook review: {output}"
    )


if __name__ == "__main__":
    main()

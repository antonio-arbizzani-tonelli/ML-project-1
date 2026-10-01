"""Measure Phase 14 quantile collisions without fitting predictive models.

Run from the project root: python -m tools.audit_discrete_binning
Uses the actual preprocessing, binning and nine CV training partitions.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.conditional_imputation import impute_tree_pair
from src.feature_preprocessing import FeaturePreprocessor, load_feature_metadata, tree_feature_matrices
from src.numpy_boosting import HistogramBinner
from src.run_preprocessing_cv import _as_binary_labels, _load_feature_names, _sha256, _stratified_folds


REFERENCE = "results/experiments/20260921T105139261213Z_phase14-boosting-codebook-corrections-only-depth5-features128-200trees.json"
SUITE = "configs/experiments/phase14_boosting_codebook_corrections_only.json"
FLAGS_CONFIG = "configs/experiments/phase16_selected_imputation_preprocessing.json"


def _array_hash(values):
    return hashlib.sha256(memoryview(np.ascontiguousarray(values)).cast("B")).hexdigest()


def _inspect(values, edges, labels=None):
    finite = np.isfinite(values)
    unique, counts = np.unique(values[finite], return_counts=True)
    ids = np.searchsorted(edges, unique, side="right")
    groups = []
    for encoded in np.unique(ids):
        positions = np.flatnonzero(ids == encoded)
        if positions.size <= 1:
            continue
        dominant = positions[np.argmax(counts[positions])]
        group = {
            "bin": int(encoded), "value_count": int(positions.size),
            "rows": int(counts[positions].sum()),
            "nonmodal_rows": int(counts[positions].sum() - counts[dominant]),
        }
        if unique.size <= 256:
            group["values"] = [
                {"value": float(unique[p]), "rows": int(counts[p]),
                 "positive_rows": int(np.sum(labels[finite & (values == unique[p])])) if labels is not None else None}
                for p in positions
            ]
        groups.append(group)
    return {
        "known_rows": int(finite.sum()), "missing_rows": int((~finite).sum()),
        "source_values": int(unique.size), "occupied_bins": int(np.unique(ids).size),
        "lost_distinctions": int(unique.size - np.unique(ids).size),
        "low_cardinality": bool(1 < unique.size <= 64),
        "fully_collapsed": bool(unique.size > 1 and np.unique(ids).size == 1),
        "merged_bin_rows": int(sum(g["rows"] for g in groups)),
        "nonmodal_rows": int(sum(g["nonmodal_rows"] for g in groups)),
        "merged_groups": groups,
    }


def main():
    started = time.perf_counter()
    root = Path.cwd()
    read = lambda path: json.loads((root / path).read_text(encoding="utf-8"))
    suite, record = read(SUITE), read(REFERENCE)
    config = read(suite["preprocessing_config_path"])
    plan = next(v["preprocessing"] for v in config["variants"] if v["experiment_name"] == suite["preprocessing_variant"])
    flags_plan = next(v["preprocessing"] for v in read(FLAGS_CONFIG)["variants"] if v["experiment_name"] == "phase16_imputation_restricted")
    flag_settings = dict(flags_plan["imputation"], fill_values=False)
    paths = {
        "boosting_suite_config": SUITE,
        "preprocessing_config": suite["preprocessing_config_path"],
        "features": suite["data"]["features_path"], "labels": suite["data"]["labels_path"],
        "split_indices": suite["data"]["split_indices_path"],
        "feature_metadata": suite["feature_metadata_path"],
    }
    hashes = {name: _sha256(root / path) for name, path in paths.items()}
    assert hashes == record["outcome"]["input_sha256"], "Phase 14 input hashes differ"
    x = np.load(root / paths["features"], mmap_mode="r")
    labels = _as_binary_labels(np.load(root / paths["labels"]))
    with np.load(root / paths["split_indices"]) as split:
        development = split["development_indices"].copy()
    names = _load_feature_names(root / suite["data"]["x_train_csv_path"])
    metadata = load_feature_metadata(root / paths["feature_metadata"])
    by_name = {entry["name"]: entry for entry in metadata}
    flag_processor = FeaturePreprocessor(names, metadata, flags_plan)
    oof_info = record["outcome"]["oof_predictions_artifact"]
    assert _sha256(root / oof_info["path"]) == oof_info["sha256"]
    with np.load(root / oof_info["path"]) as oof:
        fold_ids = oof["fold_ids"].copy()
        assert np.array_equal(oof["development_indices"], development)
        assert np.array_equal(oof["labels"], labels[development])
    synthetic = np.r_[np.zeros(99), 1.0].astype(np.float32).reshape(-1, 1)
    toy = HistogramBinner(64)
    toy_encoded = toy.fit_transform(synthetic)
    assert np.unique(toy_encoded).size == 1
    # Preserve distinct finite states with upper-value boundaries; NaN stays 255.
    exact_edges = np.unique(synthetic)[1:]
    assert np.unique(np.searchsorted(exact_edges, synthetic, side="right")).size == 2
    rows, flag_rows, onehot_rows, partitions = [], [], [], []
    outer = list(_stratified_folds(labels[development], suite["fold_count"], suite["seed"]))
    assignments = []
    for fold, (train, valid) in enumerate(outer, 1):
        assert np.array_equal(np.flatnonzero(fold_ids == fold), valid)
        assignments.append((f"outer_{fold}", fold, None, development[train]))
        inner_seed = suite["nested_threshold"]["inner_seed"] + fold
        for inner_fold, (inner_train, _) in enumerate(_stratified_folds(labels[development[train]], 2, inner_seed), 1):
            assignments.append((f"outer_{fold}_inner_{inner_fold}", fold, inner_fold, development[train[inner_train]]))
    for partition, outer_fold, inner_fold, train_rows in assignments:
        raw = np.asarray(x[train_rows], dtype=np.float32)
        empty = np.empty((0, x.shape[1]), dtype=np.float32)
        clean, clean_empty, output_names = tree_feature_matrices(raw, empty, names, metadata, plan)
        assert len(output_names) == 295
        y = labels[train_rows]
        binner = HistogramBinner(64).fit(clean)
        encoded = binner.transform(clean)
        assert np.array_equal(encoded == 255, np.isnan(clean)), "Missing state changed"
        for column, name in enumerate(output_names):
            result = _inspect(clean[:, column], binner.edges_[column], y)
            unique = np.unique(clean[np.isfinite(clean[:, column]), column])
            assert result["occupied_bins"] == np.unique(encoded[encoded[:, column] != 255, column]).size
            if unique.size <= 64:
                assert np.unique(np.searchsorted(unique[1:], unique, side="right")).size == unique.size
            rows.append({"partition": partition, "feature": name, "semantic_type": by_name[name]["semantic_type"],
                         "title": by_name[name].get("title"), **result})
        del encoded
        with_flags, _, flag_names = impute_tree_pair(raw, empty, clean, clean_empty, output_names, flag_processor, flag_settings)
        assert np.array_equal(with_flags[:, :295], clean, equal_nan=True), "Indicator-only path changed source values"
        flags = with_flags[:, 295:]
        flag_binner = HistogramBinner(64).fit(flags)
        for column, name in enumerate(flag_names[295:]):
            result = _inspect(flags[:, column], flag_binner.edges_[column], y)
            flag_rows.append({"partition": partition, "feature": name, "ones": int(flags[:, column].sum()),
                              "ones_fraction": float(flags[:, column].mean()), **result})
        del with_flags, flags
        # Prospective one-hot: preserve original NaNs in every activity indicator.
        # These columns are diagnostics only and are not added to Phase 14.
        for name in ("EXRACT11", "EXRACT21"):
            original = clean[:, output_names.index(name)]
            known = np.isfinite(original)
            for code in np.unique(original[known]):
                indicator = np.where(known, (original == code).astype(np.float32), np.nan).reshape(-1, 1)
                onehot_binner = HistogramBinner(64).fit(indicator)
                result = _inspect(indicator[:, 0], onehot_binner.edges_[0])
                onehot_rows.append({"partition": partition, "feature": name, "code": float(code),
                                   "ones": int(np.sum(original == code)), "ones_fraction_known": float(np.mean(original[known] == code)),
                                   "source_values": result["source_values"], "occupied_bins": result["occupied_bins"],
                                   "fully_collapsed": result["fully_collapsed"]})
        subset = [r for r in rows if r["partition"] == partition]
        small = [r for r in subset if r["low_cardinality"]]
        affected = [r for r in small if r["lost_distinctions"] > 0]
        partitions.append({"partition": partition, "outer_fold": outer_fold, "inner_fold": inner_fold,
                           "training_rows": int(train_rows.size), "training_positive_rows": int(y.sum()),
                           "training_row_sha256": _array_hash(train_rows), "retained_features": len(output_names),
                           "low_cardinality_features": len(small), "low_cardinality_affected": len(affected),
                           "low_cardinality_fully_collapsed": [r["feature"] for r in affected if r["fully_collapsed"]],
                           "low_cardinality_lost_distinctions": sum(r["lost_distinctions"] for r in affected)})
        print(f"{partition}: rows={train_rows.size}, low-cardinality collisions={len(affected)}/{len(small)}, completely collapsed={partitions[-1]['low_cardinality_fully_collapsed']}", flush=True)
        del raw, clean, clean_empty, binner
        gc.collect()
    output = root / "results/eda"
    output.mkdir(parents=True, exist_ok=True)
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "environment": {"python": platform.python_version(), "numpy": np.__version__},
               "reference": REFERENCE, "input_sha256": hashes,
               "additional_sha256": {path: _sha256(root / path) for path in (FLAGS_CONFIG, "src/numpy_boosting.py", "src/feature_preprocessing.py", "src/conditional_imputation.py", "tools/audit_discrete_binning.py")},
               "predictive_models_trained": 0, "imputation_models_trained": 0, "n_bins": 64,
               "small_cardinality_limit": 64, "holdout_used": False,
               "synthetic_99_zero_1_one": {"current_edges": toy.edges_[0].tolist(), "occupied_bins": int(np.unique(toy_encoded).size), "exact_edges": exact_edges.tolist()},
               "nonmodal_rows_definition": "For each merged bin, rows belonging to all source values except its most frequent value. Counts refer to feature cells, not distinct subjects or classification errors.",
               "prospective_flags": "Exact Phase 16 restricted indicator construction with fill_values=False, using unchanged Phase 14 source matrices. No imputation predictors called.",
               "prospective_onehot": "One indicator per observed training activity category, source NaNs retained. Diagnostic only; categories are not selected using validation data.",
               "partitions": partitions, "source_features": rows, "restricted_flags": flag_rows,
               "activity_onehot": onehot_rows, "elapsed_seconds": time.perf_counter() - started}
    path = output / "discrete_binning_audit.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["partition", "feature", "semantic_type", "known_rows", "missing_rows", "source_values", "occupied_bins", "lost_distinctions", "low_cardinality", "fully_collapsed", "merged_bin_rows", "nonmodal_rows"]
    with (output / "discrete_binning_features.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}; elapsed={payload['elapsed_seconds']:.1f}s; models trained=0", flush=True)


if __name__ == "__main__":
    main()

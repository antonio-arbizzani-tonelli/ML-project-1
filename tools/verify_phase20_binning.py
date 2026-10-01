"""Verify exact bins and Phase 14 checkpoint compatibility before full CV."""

import gc
import json
from pathlib import Path

import numpy as np

from src.feature_preprocessing import load_feature_metadata, tree_feature_matrices
from src.numpy_boosting import HistogramBinner, HistogramGradientBoostingClassifier
from src.run_preprocessing_cv import _as_binary_labels, _load_feature_names, _sha256, _stratified_folds


def main():
    root = Path.cwd()
    reference = "results/experiments/20260921T105139261213Z_phase14-boosting-codebook-corrections-only-depth5-features128-200trees.json"
    record = json.loads((root / reference).read_text(encoding="utf-8"))
    suite = json.loads((root / "configs/experiments/phase20_phase14_exact_low_cardinality.json").read_text(encoding="utf-8"))
    config = json.loads((root / suite["preprocessing_config_path"]).read_text(encoding="utf-8"))
    plan = next(v["preprocessing"] for v in config["variants"] if v["experiment_name"] == suite["preprocessing_variant"])
    for key, path in (("features", suite["data"]["features_path"]), ("labels", suite["data"]["labels_path"]),
                      ("split_indices", suite["data"]["split_indices_path"]), ("feature_metadata", suite["feature_metadata_path"]),
                      ("preprocessing_config", suite["preprocessing_config_path"])):
        assert _sha256(root / path) == record["outcome"]["input_sha256"][key]
    x = np.load(root / suite["data"]["features_path"], mmap_mode="r")
    y = _as_binary_labels(np.load(root / suite["data"]["labels_path"]))
    with np.load(root / suite["data"]["split_indices_path"]) as split:
        development = split["development_indices"].copy()
    names = _load_feature_names(root / suite["data"]["x_train_csv_path"])
    metadata = load_feature_metadata(root / suite["feature_metadata_path"])
    info = record["outcome"]["oof_predictions_artifact"]
    assert _sha256(root / info["path"]) == info["sha256"]
    with np.load(root / info["path"]) as oof:
        assert np.array_equal(oof["development_indices"], development)
        probabilities = oof["probabilities"].copy()
        fold_ids = oof["fold_ids"].copy()
    report = {"reference": reference, "baseline_retrained": False, "final_config_sha256": _sha256(root / "configs/final_model.json"),
              "source_sha256": _sha256(root / "src/numpy_boosting.py"), "partitions": [], "checkpoint_prediction_checks": []}
    for fold, (train, valid) in enumerate(_stratified_folds(y[development], 3, suite["seed"]), 1):
        assert np.array_equal(np.flatnonzero(fold_ids == fold), valid)
        partitions = [(f"outer_{fold}", train)]
        for inner, (inner_train, _) in enumerate(_stratified_folds(y[development[train]], 2, suite["nested_threshold"]["inner_seed"] + fold), 1):
            partitions.append((f"outer_{fold}_inner_{inner}", train[inner_train]))
        for name, positions in partitions:
            source = np.asarray(x[development[positions]], dtype=np.float32)
            clean, _, output_names = tree_feature_matrices(source, np.empty((0, x.shape[1]), dtype=np.float32), names, metadata, plan)
            assert len(output_names) == 295
            old = HistogramBinner(64).fit(clean)
            new = HistogramBinner(64, "exact_low_cardinality").fit(clean)
            encoded = new.transform(clean)
            assert np.array_equal(encoded == 255, np.isnan(clean))
            restored = []
            for column, feature in enumerate(output_names):
                values = np.unique(clean[np.isfinite(clean[:, column]), column])
                if new.exact_columns_[column]:
                    assert values.size <= 64
                    occupied = np.unique(np.searchsorted(new.edges_[column], values, side="right")).size
                    assert occupied == values.size
                    assert int(new.bin_counts_[column]) == values.size
                    if np.unique(np.searchsorted(old.edges_[column], values, side="right")).size < values.size:
                        restored.append(feature)
                else:
                    np.testing.assert_array_equal(old.edges_[column], new.edges_[column])
            item = {"partition": name, "rows": int(positions.size), "exact_columns": int(new.exact_columns_.sum()),
                    "restored_columns": restored, "preserved_all_low_cardinality_states": True,
                    "unchanged_quantile_fallback": True, "unchanged_missing_state": True,
                    "feature_names": output_names, "bin_counts": new.bin_counts_.tolist()}
            report["partitions"].append(item)
            print(f"PASS {name}: exact={item['exact_columns']}, restored={len(restored)}", flush=True)
            del source, clean, encoded, old, new
            gc.collect()
        cp = record["outcome"]["model_checkpoints"][fold - 1]
        assert _sha256(root / cp["path"]) == cp["sha256"]
        model, _ = HistogramGradientBoostingClassifier.load_checkpoint(root / cp["path"])
        # Binary mappings are learned on the full original outer training.
        clean_train, clean_valid, _ = tree_feature_matrices(np.asarray(x[development[train]], dtype=np.float32),
            np.asarray(x[development[valid]], dtype=np.float32), names, metadata, plan)
        predicted = model.predict_proba(clean_valid)[:, 1]
        difference = float(np.max(np.abs(predicted - probabilities[valid])))
        assert difference == 0.0
        report["checkpoint_prediction_checks"].append({"fold": fold, "rows": int(valid.size), "maximum_difference": difference})
        print(f"PASS Phase14 checkpoint fold={fold}: maximum prediction difference={difference}", flush=True)
        del model, clean_train, clean_valid, predicted
        gc.collect()
    path = root / "results/eda/phase20_binning_preflight.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()

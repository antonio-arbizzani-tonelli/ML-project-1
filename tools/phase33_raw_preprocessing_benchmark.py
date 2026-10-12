"""Compare unchanged raw predictors with the selected Phase 14 pipeline."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

from src import run_boosting_cv as runner
from src.evaluation import classification_metrics
from src.feature_preprocessing import tree_feature_matrices
from src.run_logged_boosting_cv import run_logged_suite
from src.run_preprocessing_cv import _sha256, _stratified_folds
from src.summarize_phase19_transfer import BASELINE, _arrays, _paired_interval
from tools.run_targeted_preprocessing_trials import _Tee
from tools.targeted_preprocessing_analysis import _input_data


REFERENCE = "configs/experiments/phase14_boosting_codebook_corrections_only.json"
SUITE = "configs/experiments/phase33_raw_boosting_cv.json"
PLAN = "configs/experiments/phase33_raw_tree_preprocessing.json"
ARTIFACTS = "results/boosting_artifacts/phase33_raw_preprocessing"
RECORDS = "results/experiments/phase33_raw_preprocessing"
PREFLIGHT = "results/eda/phase33_raw_preprocessing_preflight.json"
COMPARISON = "results/experiments/phase33_raw_preprocessing_comparison.json"
REPORT = "results/eda/phase33_raw_preprocessing_summary.md"
HELPER = "tools/phase33_raw_preprocessing_benchmark.py"


def read(root, name):
    return json.loads((root / name).read_text(encoding="utf-8"))


def preflight(root):
    """Verify unchanged model/data/folds and a true raw passthrough."""
    suite, reference = read(root, SUITE), read(root, REFERENCE)
    expected = copy.deepcopy(reference)
    expected.update(experiment_prefix="phase33_raw_321_features",
                    artifact_dir=ARTIFACTS, preprocessing_config_path=PLAN,
                    preprocessing_variant="phase33_raw_321_features")
    ignored = {"hypothesis"}
    if ({k: v for k, v in suite.items() if k not in ignored}
            != {k: v for k, v in expected.items() if k not in ignored}):
        raise ValueError("Raw benchmark changed more than preprocessing/provenance.")
    runner._validate_suite(suite)
    plan = runner._variant(read(root, PLAN), suite["preprocessing_variant"])["preprocessing"]
    if plan != {"name": "raw_tree_passthrough"}:
        raise ValueError("Raw benchmark must disable all semantic preprocessing.")
    baseline = read(root, BASELINE)
    hashes = {key: _sha256(root / suite["data"][field]) for key, field in (
        ("features", "features_path"), ("labels", "labels_path"),
        ("split_indices", "split_indices_path"))}
    hashes["feature_metadata"] = _sha256(root / suite["feature_metadata_path"])
    for key, value in hashes.items():
        if value != baseline["outcome"]["input_sha256"][key]:
            raise ValueError(f"Historical baseline data changed: {key}")
    parameters = {**suite["models"][0]["parameters"], "n_estimators": 200}
    if parameters != baseline["config"]["model"]["parameters"]:
        raise ValueError("Historical baseline parameters differ.")
    x, y, development, names, metadata = _input_data(root, suite)
    arrays = _arrays(root, baseline)
    np.testing.assert_array_equal(arrays["development_indices"], development)
    np.testing.assert_array_equal(arrays["labels"], y[development])
    for fold, (_, valid) in enumerate(
            _stratified_folds(y[development], suite["fold_count"], suite["seed"]), 1):
        np.testing.assert_array_equal(arrays["fold_ids"][valid], np.full(valid.size, fold))
    sample = np.asarray(x[development[:512]], dtype=np.float32)
    train, valid, output_names = tree_feature_matrices(
        sample[:256], sample[256:], names, metadata, plan)
    np.testing.assert_array_equal(train, sample[:256])
    np.testing.assert_array_equal(valid, sample[256:])
    if output_names != names or len(names) != 321:
        raise ValueError("Raw benchmark must retain exactly all 321 source predictors.")
    result = {
        "scope": "One full preprocessing ablation; model, data and folds fixed.",
        "reference_record": BASELINE, "raw_suite": SUITE,
        "input_sha256": hashes, "raw_feature_count": len(output_names),
        "source_sha256": {name: _sha256(root / name) for name in (
            HELPER, "src/feature_preprocessing.py", "src/numpy_boosting.py",
            "src/evaluation.py", "src/run_boosting_cv.py",
            "src/run_logged_boosting_cv.py", "configs/final_model.json")},
        "model_parameters": parameters, "fold_count": suite["fold_count"],
        "inner_threshold": suite["nested_threshold"],
        "raw_policy": "All parsed CSV predictors; original finite codes and NaNs unchanged; no fill, scaling, recoding, exclusions or extra features. Training-fitted histogram bins and missing routing remain model components.",
        "interpretation": "Joint effect of base cleaning and exclusions; 128/321 versus 128/295 candidate fraction. Does not isolate individual preprocessing operations.",
    }
    (root / PREFLIGHT).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("PREFLIGHT: 321 unchanged columns; baseline inputs, model and folds match.", flush=True)
    return result


def compare(root, record_path):
    """Check row pairing and reconstruct nested metrics before comparison."""
    baseline, candidate = read(root, BASELINE), read(root, record_path)
    ba, ca = _arrays(root, baseline), _arrays(root, candidate)
    for key in ("development_indices", "fold_ids", "labels"):
        np.testing.assert_array_equal(ba[key], ca[key])
    if candidate["config"]["model"] != baseline["config"]["model"]:
        raise ValueError("Model configuration changed.")
    bn = baseline["outcome"]["nested_threshold_evaluation"]
    cn = candidate["outcome"]["nested_threshold_evaluation"]
    if len(cn["folds"]) != 3:
        raise ValueError("Three completed nested folds are required.")
    for record, arrays in ((baseline, ba), (candidate, ca)):
        np.testing.assert_array_equal(arrays["nested_predictions"],
                                      arrays["probabilities"] >= arrays["nested_thresholds"])
        _, metrics = classification_metrics(arrays["labels"], arrays["nested_predictions"])
        if abs(metrics["f1"] - record["outcome"]["nested_threshold_evaluation"]["pooled_metrics"]["f1"]) > 1e-14:
            raise ValueError("Recorded nested F1 does not match saved decisions.")
    bm, cm = bn["pooled_metrics"], cn["pooled_metrics"]
    result = {
        "baseline_record": BASELINE, "raw_record": str(record_path).replace("\\", "/"),
        "baseline_metrics": bm, "raw_metrics": cm,
        "baseline_confusion_counts": bn["pooled_confusion_counts"],
        "raw_confusion_counts": cn["pooled_confusion_counts"],
        "raw_minus_cleaned_metric_deltas": {key: cm[key] - bm[key] for key in bm},
        "folds": [{"fold": b["outer_fold"],
                   "cleaned_f1": b["outer_evaluation_metrics"]["f1"],
                   "raw_f1": c["outer_evaluation_metrics"]["f1"],
                   "raw_inner_threshold": c["inner_threshold"]}
                  for b, c in zip(bn["folds"], cn["folds"])],
        "paired_conditional_raw_minus_cleaned_f1_interval_95":
            _paired_interval(ca["labels"], ba["nested_predictions"], ca["nested_predictions"]),
        "interval_scope": "Fixed fitted models and thresholds; excludes training and model-selection uncertainty.",
        "raw_runtime_seconds": candidate["outcome"]["runtime_seconds"],
        "raw_runtime_basis": candidate["outcome"].get(
            "runtime_basis", "Full sequential CV wall time."),
        "comparison_source_sha256": {HELPER: _sha256(root / HELPER)},
        "interpretation": read(root, PREFLIGHT)["interpretation"],
    }
    (root / COMPARISON).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Raw predictors versus the selected preprocessing", "",
             "Same 262,508 development rows, three outer folds, two inner threshold folds, and 200-tree Phase 14 parameters. Raw input retains all 321 predictors and original response codes/NaNs. The selected input has 295 cleaned predictors.", "",
             "| Pipeline | Nested F1 | Precision | Recall | AP | Log loss |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, metrics in (("Selected preprocessing", bm), ("Raw predictors", cm)):
        lines.append(f"| {name} | " + " | ".join(f"{metrics[k]:.6f}" for k in
            ("f1", "precision", "recall", "average_precision", "log_loss")) + " |")
    lines.extend(["", f"Raw minus cleaned F1: {cm['f1'] - bm['f1']:+.6f}.", "",
                  "| Fold | Cleaned F1 | Raw F1 | Raw inner threshold |",
                  "| --- | ---: | ---: | ---: |"])
    for fold in result["folds"]:
        lines.append(f"| {fold['fold']} | {fold['cleaned_f1']:.6f} | {fold['raw_f1']:.6f} | {fold['raw_inner_threshold']:.6f} |")
    interval = result["paired_conditional_raw_minus_cleaned_f1_interval_95"]
    lines.extend(["", f"Paired 95% interval for raw minus cleaned F1: [{interval[0]:+.6f}, {interval[1]:+.6f}]. It includes zero; the small observed difference does not establish a stable gain. The interval conditions on fixed fits and thresholds, excluding training and model-selection uncertainty.", "",
                  result["interpretation"], "",
                  "This is a development comparison after repeated model selection. Both pipelines keep HAREHAB1. The benchmark measures the combined preprocessing effect, including feature exclusions; it cannot attribute the difference to one operation.", "",
                  f"Reproduce with `python -m tools.phase33_raw_preprocessing_benchmark run`.", "",
                  f"Raw record: [{Path(record_path).name}](../{Path(record_path).as_posix().removeprefix('results/')}).",
                  "Comparison: [JSON](../experiments/phase33_raw_preprocessing_comparison.json)."])
    (root / REPORT).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"cleaned_f1": bm["f1"], "raw_f1": cm["f1"],
                      "raw_minus_cleaned": cm["f1"] - bm["f1"]}), flush=True)
    return result


def run(root):
    """Run one nested raw-data candidate and preserve its provenance."""
    directory = root / ARTIFACTS
    records_dir = root / RECORDS
    if records_dir.exists() and list(records_dir.glob("*.json")):
        raise ValueError("Completed raw records already exist; use compare instead of retraining.")
    audit = preflight(root)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "executed_runner.py").write_bytes((root / HELPER).read_bytes())
    original_save = runner.save_experiment

    def save_raw(config, outcome, output_dir):
        config["missing_values"]["method"] = audit["raw_policy"]
        outcome["preprocessing_benchmark"] = audit
        outcome["environment"]["threads"] = {name: os.environ.get(name) for name in
            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
        outcome["source_sha256"][HELPER] = _sha256(root / HELPER)
        return original_save(config, outcome, output_dir)

    runner.save_experiment = save_raw
    try:
        with (directory / "run_stdout.log").open("w", encoding="utf-8") as log:
            tee = _Tee(sys.stdout, log)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                records = run_logged_suite(root / SUITE, root, records_dir)
                compare(root, records[0].relative_to(root))
    finally:
        runner.save_experiment = original_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run", "compare"))
    args = parser.parse_args()
    root = Path.cwd()
    if args.action == "preflight":
        preflight(root)
    elif args.action == "run":
        run(root)
    else:
        records = sorted((root / RECORDS).glob("*.json"))
        if len(records) != 1:
            raise ValueError("Expected exactly one completed raw record.")
        compare(root, records[0].relative_to(root))


if __name__ == "__main__":
    main()

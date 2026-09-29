"""Audit Git provenance and saved Phase 14 artifacts without fitting models.

This administrative utility uses Python's AST and Git CLI to verify source
history. It is separate from the NumPy challenge implementation in src/.
Run from the project root with python -m tools.audit_phase14_conformity.
"""

from __future__ import annotations

import ast
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.evaluation import classification_metrics
from src.feature_preprocessing import (
    FeaturePreprocessor,
    load_feature_metadata,
    tree_feature_matrices,
)
from src.numpy_boosting import HistogramGradientBoostingClassifier
from src.run_preprocessing_cv import (
    _as_binary_labels,
    _load_feature_names,
    _sha256,
    _stratified_folds,
)


REFERENCE = (
    "results/experiments/20260921T105139261213Z_"
    "phase14-boosting-codebook-corrections-only-depth5-features128-200trees.json"
)
FREQUENCY_NAMES = (
    "FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1",
    "EXEROFT1", "EXEROFT2", "EXERHMM1", "EXERHMM2",
)


def _digest_array(values: np.ndarray) -> str:
    """Hash the exact ordered bytes used by checkpoint metadata."""

    return hashlib.sha256(memoryview(np.ascontiguousarray(values)).cast("B")).hexdigest()


def _git_source(path: str) -> str:
    """Read the initial committed implementation used for the reference."""

    return subprocess.check_output(
        ["git", "show", f"4970aa6:{path}"]
    ).decode("utf-8")


def _tree_path_ast(source: str) -> dict[str, str]:
    """Compare only helpers actually executed by the Phase 14 tree path."""

    tree = ast.parse(source)
    functions = {
        "tree_feature_matrices", "_as_finite_code", "_normalise_label",
        "_missing_reason", "_metadata_code_maps", "_binary_mapping",
    }
    methods = {
        "__init__", "_missing_codes", "_zero_codes", "_clean_values",
        "_retained_names",
    }
    result = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            if node.name == "tree_feature_matrices":
                node.body = [
                    item for item in node.body
                    if not (
                        isinstance(item, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "imputation"
                                for t in item.targets)
                    ) and not (
                        isinstance(item, ast.If)
                        and isinstance(item.test, ast.Compare)
                        and isinstance(item.test.left, ast.Name)
                        and item.test.left.id == "imputation"
                    )
                ]
            result[node.name] = ast.dump(node, include_attributes=False)
        if isinstance(node, ast.ClassDef) and node.name == "FeaturePreprocessor":
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name in methods:
                    result[f"FeaturePreprocessor.{method.name}"] = ast.dump(
                        method, include_attributes=False
                    )
    return result


def main() -> None:
    """Write reproducibility checks and focused codebook findings to JSON."""

    root = Path.cwd()
    record = json.loads((root / REFERENCE).read_text(encoding="utf-8"))
    suite_path = root / "configs/experiments/phase14_boosting_codebook_corrections_only.json"
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    final = json.loads((root / "configs/final_model.json").read_text(encoding="utf-8"))
    preprocessing_path = root / suite["preprocessing_config_path"]
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    plan = next(v["preprocessing"] for v in preprocessing["variants"]
                if v["experiment_name"] == suite["preprocessing_variant"])
    checks = []

    def check(name: str, passed: bool, **details) -> None:
        checks.append({"name": name, "passed": bool(passed), **details})
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)

    inputs = {
        "boosting_suite_config": suite_path,
        "preprocessing_config": preprocessing_path,
        "features": root / suite["data"]["features_path"],
        "labels": root / suite["data"]["labels_path"],
        "split_indices": root / suite["data"]["split_indices_path"],
        "feature_metadata": root / suite["feature_metadata_path"],
    }
    for name, path in inputs.items():
        actual = _sha256(path)
        expected = record["outcome"]["input_sha256"][name]
        check(f"input_sha256:{name}", actual == expected,
              path=str(path.relative_to(root)), expected=expected, actual=actual)
    check("final_model_parameters", final["model"] == record["config"]["model"]["parameters"])
    check("final_preprocessing_paths",
          final["preprocessing"]["config"] == suite["preprocessing_config_path"]
          and final["preprocessing"]["variant"] == suite["preprocessing_variant"]
          and final["preprocessing"]["metadata"] == suite["feature_metadata_path"])
    nested = record["outcome"]["nested_threshold_evaluation"]
    check("final_threshold", final["decision_threshold"] == nested["threshold_summary"]["mean"])
    check("phase14_has_no_imputation", plan.get("imputation") is None)
    for path in ("src/numpy_boosting.py", "src/evaluation.py", "src/run_boosting_cv.py"):
        current = (root / path).read_text(encoding="utf-8")
        check(f"unchanged_source:{path}", current == _git_source(path))
    check("unchanged_tree_preprocessing_path",
          _tree_path_ast((root / "src/feature_preprocessing.py").read_text(encoding="utf-8"))
          == _tree_path_ast(_git_source("src/feature_preprocessing.py")))

    x = np.load(inputs["features"], mmap_mode="r")
    labels = _as_binary_labels(np.load(inputs["labels"]))
    with np.load(inputs["split_indices"]) as split:
        development = split["development_indices"].astype(np.int64)
    feature_names = _load_feature_names(root / suite["data"]["x_train_csv_path"])
    metadata = load_feature_metadata(inputs["feature_metadata"])
    processor = FeaturePreprocessor(feature_names, metadata, plan)
    check("data_shape", x.shape == (328135, 321) and labels.size == 328135,
          shape=list(x.shape), positive_rows=int(labels.sum()))
    check("development_rows", development.size == 262508,
          rows=int(development.size), positives=int(labels[development].sum()))
    check("harehab1_retained", "HAREHAB1" in processor._retained_names())

    oof_info = record["outcome"]["oof_predictions_artifact"]
    oof_path = root / oof_info["path"]
    oof_valid = _sha256(oof_path) == oof_info["sha256"]
    check("oof_artifact_sha256", oof_valid)
    if not oof_valid:
        raise ValueError("Reference OOF artifact differs from its trusted hash.")
    with np.load(oof_path) as oof:
        arrays = {key: oof[key].copy() for key in oof.files}
    check("oof_rows_and_labels", np.array_equal(arrays["development_indices"], development)
          and np.array_equal(arrays["labels"], labels[development]))
    counts, metrics = classification_metrics(arrays["labels"], arrays["nested_predictions"])
    check("reference_nested_metrics", all(
        np.isclose(metrics[k], v, rtol=0, atol=1e-15)
        for k, v in nested["pooled_metrics"].items() if k in metrics
    ), recomputed=metrics)
    fold_results = []
    for fold, (train, valid) in enumerate(
        _stratified_folds(labels[development], suite["fold_count"], suite["seed"]), start=1
    ):
        check(f"outer_fold_{fold}_row_assignment",
              np.array_equal(np.flatnonzero(arrays["fold_ids"] == fold), valid))
        cp = record["outcome"]["model_checkpoints"][fold - 1]
        cp_path = root / cp["path"]
        cp_valid = _sha256(cp_path) == cp["sha256"]
        check(f"outer_fold_{fold}_checkpoint_sha256", cp_valid)
        if not cp_valid:
            raise ValueError("Reference checkpoint differs from its trusted hash.")
        model, cp_metadata = HistogramGradientBoostingClassifier.load_checkpoint(cp_path)
        train_rows = development[train]
        check(f"outer_fold_{fold}_training_fingerprints",
              _digest_array(train_rows) == cp_metadata["training_row_indices_sha256"]
              and _digest_array(labels[train_rows]) == cp_metadata["training_labels_sha256"])
        train_matrix, valid_matrix, names = tree_feature_matrices(
            x[train_rows], x[development[valid]], feature_names, metadata, plan
        )
        check(f"outer_fold_{fold}_feature_count", len(names) == 295)
        predicted = model.predict_proba(valid_matrix)[:, 1]
        difference = float(np.max(np.abs(predicted - arrays["probabilities"][valid])))
        check(f"outer_fold_{fold}_saved_predictions", difference <= 1e-12,
              maximum_absolute_difference=difference)
        fold_results.append({"fold": fold, "feature_names": names,
                             "prediction_maximum_absolute_difference": difference})
        del model, train_matrix, valid_matrix

    codebook_findings = []
    for name in FREQUENCY_NAMES:
        entry = processor.metadata_by_name[name]
        missing_codes = processor._missing_codes(name, entry)
        raw = x[development, feature_names.index(name)]
        count777 = int(np.count_nonzero(raw == 777))
        clean, _ = processor._clean_values(raw, missing_codes, processor._zero_codes(name, entry))
        codebook_findings.append({
            "feature": name, "code": 777,
            "documented_meaning": "Don't know / Not sure",
            "codebook_pages": entry["codebook_pages"],
            "development_count": count777,
            "converted_to_nan": 777.0 in missing_codes,
            "still_numeric_count": int(np.count_nonzero(clean == 777)),
        })
    semantic_gaps = [f for f in codebook_findings
                     if f["development_count"] and not f["converted_to_nan"]]
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_record": REFERENCE,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "trained_models": 0,
        "conforms_to_saved_phase14": all(c["passed"] for c in checks),
        "focused_semantic_check_passed": not semantic_gaps,
        "checks": checks, "fold_results": fold_results,
        "focused_codebook_findings": codebook_findings,
        "semantic_gaps": semantic_gaps,
        "scope": "Reproduce saved inputs, tree path, checkpoints and predictions; inspect ten food/exercise nonresponse codes. No model fitting.",
    }
    out = root / "results/eda/phase19_phase14_conformity.json"
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {out}", flush=True)
    print(f"CONFORMITY={report['conforms_to_saved_phase14']} SEMANTIC_GAPS={len(semantic_gaps)}", flush=True)


if __name__ == "__main__":
    main()

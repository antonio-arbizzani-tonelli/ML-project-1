"""Compare Phase 14 seeds and their uniform mean with nested thresholds."""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from src import run_boosting_cv as runner
from src.evaluation import average_precision, best_f1_threshold, binary_log_loss, classification_metrics
from src.experiment_logger import save_experiment
from src.feature_preprocessing import tree_feature_matrices
from src.numpy_boosting import HistogramBinner, HistogramGradientBoostingClassifier
from src.run_logged_boosting_cv import _logged_fit, run_logged_suite
from src.run_preprocessing_cv import _sha256, _stratified_folds
from src.summarize_phase19_transfer import BASELINE, _arrays, _paired_interval, _publish
from tools.phase23_missing_reason_flags import SOURCES
from tools.phase24_original_restricted_imputation import verify_rows
from tools.run_targeted_preprocessing_trials import _Tee
from tools.targeted_preprocessing_analysis import _input_data, _partitions


SUITE = "configs/experiments/phase26_phase14_seed20260921.json"
REFERENCE_SUITE = "configs/experiments/phase14_boosting_codebook_corrections_only.json"
ARTIFACTS = "results/boosting_artifacts/phase26_two_seeds"
PREFLIGHT = "results/eda/phase26_two_seeds_preflight.json"
COMPARISON = "results/experiments/phase26_two_seeds_comparison.json"
REPORT = "results/eda/phase26_two_seeds_summary.md"
HELPER = "tools/phase26_two_seeds.py"


def read(root, path):
    return json.loads((root / path).read_text(encoding="utf-8"))


def validate_change(candidate, reference):
    runner._validate_suite(candidate)
    expected = copy.deepcopy(reference)
    expected["models"][0]["parameters"]["random_seed"] = 20260921
    ignored = {"experiment_prefix", "artifact_dir", "hypothesis"}
    if ({k: v for k, v in candidate.items() if k not in ignored}
            != {k: v for k, v in expected.items() if k not in ignored}):
        raise ValueError("Only the booster seed may change from Phase 14.")
    if reference["models"][0]["parameters"]["random_seed"] != 20260920:
        raise ValueError("Expected original seed 20260920.")


def configuration(root):
    suite, reference = read(root, SUITE), read(root, REFERENCE_SUITE)
    validate_change(suite, reference)
    variant = runner._variant(read(root, suite["preprocessing_config_path"]), suite["preprocessing_variant"])
    return suite, reference, variant


def fingerprints(root, suite):
    result = {key: _sha256(root / suite["data"][field]) for key, field in (
        ("features", "features_path"), ("labels", "labels_path"), ("split_indices", "split_indices_path"))}
    result.update(feature_metadata=_sha256(root / suite["feature_metadata_path"]),
        preprocessing_config=_sha256(root / suite["preprocessing_config_path"]),
        boosting_suite_config=_sha256(root / SUITE))
    return result


def preflight(root):
    if (root / ARTIFACTS / "executed_runner.py").exists():
        raise ValueError("Training snapshot exists; preserve its original preflight and use compare on the saved records.")
    suite, reference, variant = configuration(root)
    baseline = read(root, BASELINE)
    inputs = fingerprints(root, suite)
    for key, value in inputs.items():
        if key != "boosting_suite_config" and value != baseline["outcome"]["input_sha256"][key]:
            raise ValueError(f"Historical input changed: {key}")
    parameters = {**reference["models"][0]["parameters"], "n_estimators": 200}
    if parameters != baseline["config"]["model"]["parameters"]:
        raise ValueError("Reference parameters differ from the recorded Phase 14 model.")
    x, y, development, names, metadata = _input_data(root, suite)
    ba = _arrays(root, baseline)
    verify_rows(ba, development, y, list(_stratified_folds(y[development], 3, suite["seed"])))
    check = {"scope": "Development only; verify reusable Phase 14 models without new training.",
        "input_sha256": inputs, "source_sha256": {p: _sha256(root / p) for p in SOURCES},
        "helper_sha256": _sha256(root / HELPER), "reference_suite_sha256": _sha256(root / REFERENCE_SUITE),
        "baseline_record_sha256": _sha256(root / BASELINE),
        "final_config_sha256": _sha256(root / "configs/final_model.json"),
        "partitions": [], "baseline_checkpoint_checks": []}
    for partition, train, valid in _partitions(y, development, suite):
        ct, cv, cn = tree_feature_matrices(x[development[train]], x[development[valid]], names, metadata,
                                         variant["preprocessing"])
        if len(cn) != 295:
            raise ValueError("Expected 295 unchanged Phase 14 inputs.")
        binner = HistogramBinner(64).fit(ct)
        check["partitions"].append({"partition": partition, "feature_count": 295, "output_names": cn,
            "training_rows": int(train.size), "validation_rows": int(valid.size),
            "bin_counts": binner.bin_counts_.tolist(), "exact_column_indices": [],
            "training_row_indices_sha256": runner._array_sha256(development[train]),
            "training_labels_sha256": runner._array_sha256(y[development[train]])})
        if "inner" not in partition:
            fold = int(partition[-1])
            cp = baseline["outcome"]["model_checkpoints"][fold - 1]
            if cp["fold"] != fold or _sha256(root / cp["path"]) != cp["sha256"]:
                raise ValueError("Reference checkpoint changed.")
            model, saved = HistogramGradientBoostingClassifier.load_checkpoint(root / cp["path"])
            np.testing.assert_array_equal(saved["training_row_indices"], development[train])
            for key in ("training_row_indices_sha256", "training_labels_sha256"):
                if saved[key] != check["partitions"][-1][key]:
                    raise ValueError(f"Reference training boundary changed: {key}")
            if saved["input_sha256"] != baseline["outcome"]["input_sha256"]:
                raise ValueError("Reference checkpoint inputs differ from its record.")
            if any(getattr(model, key) != value for key, value in parameters.items()):
                raise ValueError("Reference checkpoint parameter differs.")
            np.testing.assert_array_equal(model.binner_.bin_counts_, binner.bin_counts_)
            for now, before in zip(binner.edges_, model.binner_.edges_):
                np.testing.assert_array_equal(now, before)
            delta = float(np.max(np.abs(model.predict_proba(cv)[:, 1] - ba["probabilities"][valid])))
            if delta != 0:
                raise ValueError("Reference probabilities cannot be reproduced exactly.")
            check["baseline_checkpoint_checks"].append({"fold": fold, "maximum_probability_difference": delta})
            del model
        print(f"PREFLIGHT {partition}: unchanged Phase 14 features and bins verified", flush=True)
        del ct, cv, binner
        gc.collect()
    (root / PREFLIGHT).write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
    return check


def verify_frozen(root, suite, check, *, allow_verification_update=False):
    for path, expected in check["source_sha256"].items():
        if _sha256(root / path) != expected:
            raise ValueError(f"Training source changed: {path}")
    if _sha256(root / HELPER) != check["helper_sha256"]:
        archive = root / ARTIFACTS / "executed_runner.py"
        if (not allow_verification_update or not archive.exists()
                or _sha256(archive) != check["helper_sha256"]):
            raise ValueError("Training helper changed; post-training checks require its intact executed snapshot.")
    for path, key in ((REFERENCE_SUITE, "reference_suite_sha256"),
                      (BASELINE, "baseline_record_sha256"), ("configs/final_model.json", "final_config_sha256")):
        if _sha256(root / path) != check[key]:
            raise ValueError(f"Frozen file changed: {path}")
    if fingerprints(root, suite) != check["input_sha256"]:
        raise ValueError("Inputs changed after preflight.")


def load_inner(root, info):
    if _sha256(root / info["path"]) != info["sha256"]:
        raise ValueError("Inner artifact fingerprint differs.")
    with np.load(root / info["path"]) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    for key, expected in info["array_sha256"].items():
        if runner._array_sha256(arrays[key]) != expected:
            raise ValueError(f"Inner array fingerprint differs: {key}")
    return arrays


def aligned_mean(first, second):
    """Require the same evaluation rows and partitions before averaging scores."""
    for key in ("training_row_indices", "outer_validation_row_indices", "fold_ids", "labels", "outer_fold", "inner_seed"):
        if not np.array_equal(first[key], second[key]):
            raise ValueError(f"Component inner arrays do not align: {key}")
    if np.intersect1d(first["training_row_indices"], first["outer_validation_row_indices"]).size:
        raise ValueError("Outer evaluation rows entered inner selection.")
    for arrays in (first, second):
        p = arrays["probabilities"]
        if p.shape != arrays["labels"].shape or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
            raise ValueError("Component probabilities are invalid.")
    return (first["probabilities"] + second["probabilities"]) / 2.0


def verify_feature_order(names, saved, expected, component_seed):
    """Older Phase 14 checkpoints predate the feature-name metadata field."""
    if names != expected:
        raise ValueError("Feature order differs from the frozen preflight.")
    stored = saved.get("output_feature_names")
    if stored is None:
        if component_seed != 20260920:
            raise ValueError("New component checkpoint must record its feature names.")
        # Legacy order is checked against frozen preprocessing and binner edges
        # in preflight, and exact checkpoint probabilities in verify_component.
        return "legacy_checkpoint_order_verified_from_preflight_and_predictions"
    if names != stored:
        raise ValueError("Feature order differs from checkpoint metadata.")
    return "checkpoint_feature_names_match"


def recover_baseline_inner(root, suite, reference, variant, check, directory):
    baseline = read(root, BASELINE)
    ba = _arrays(root, baseline)
    x, y, development, names, metadata = _input_data(root, suite)
    outer = list(_stratified_folds(y[development], 3, suite["seed"]))
    name = reference["models"][0]["name"]
    profiles = []
    original = runner._fit_checkpoint_probabilities
    runner._fit_checkpoint_probabilities = _logged_fit(original, profiles, 0)
    started = time.perf_counter()
    try:
        results, thresholds = runner._nested_evaluations(
            {name: [200]}, {name: reference["models"][0]}, variant, x, development, y[development], names,
            metadata, [valid for _, valid in outer], {(name, 200): ba["probabilities"]}, 2,
            suite["nested_threshold"]["inner_seed"], directory / "inner_oof_predictions", "phase26_baseline_recovery", root)
    finally:
        runner._fit_checkpoint_probabilities = original
    elapsed = time.perf_counter() - started
    nested = results[(name, 200)]
    old = baseline["outcome"]["nested_threshold_evaluation"]
    for current, historical in zip(nested["folds"], old["folds"]):
        for key in ("inner_threshold", "inner_selection_metrics", "outer_evaluation_metrics", "outer_evaluation_confusion_counts"):
            if current[key] != historical[key]:
                raise ValueError(f"Recovered baseline differs from historical evaluation: {key}")
    if nested["pooled_metrics"] != old["pooled_metrics"] or nested["pooled_confusion_counts"] != old["pooled_confusion_counts"]:
        raise ValueError("Recovered baseline pooled metrics differ.")
    np.testing.assert_array_equal(thresholds[(name, 200)], ba["nested_thresholds"])
    verify_frozen(root, suite, check)
    config = copy.deepcopy(baseline["config"])
    config.update(experiment_name="phase26_phase14_baseline_inner_recovery",
                  hypothesis="Recover the six missing inner fits and reproduce historical Phase 14 thresholds exactly.")
    outcome = copy.deepcopy(baseline["outcome"])
    outcome.update(nested_threshold_evaluation=nested, runtime_seconds=elapsed, individual_fit_profiles=profiles,
        runtime_basis="Six new inner fits only; original three outer fits reused.",
        source_sha256=check["source_sha256"], input_sha256=check["input_sha256"],
        environment={"python": platform.python_version(), "numpy": np.__version__},
        original_outer_reuse={"record": BASELINE, "record_sha256": check["baseline_record_sha256"],
                              "preflight": PREFLIGHT, "preflight_sha256": _sha256(root / PREFLIGHT)},
        runner_source_artifact={"path": str((root / ARTIFACTS / "executed_runner.py").relative_to(root)),
                                "sha256": check["helper_sha256"]})
    record = save_experiment(config, outcome, directory / "records")
    print("BASELINE RECOVERED: all historical thresholds and metrics identical", flush=True)
    return record


def verify_component(root, record, suite, check, reference_seed):
    ba = _arrays(root, read(root, BASELINE))
    arrays = _arrays(root, record)
    for key in ("development_indices", "fold_ids", "labels"):
        np.testing.assert_array_equal(arrays[key], ba[key])
    expected_params = {**read(root, BASELINE)["config"]["model"]["parameters"], "random_seed": reference_seed}
    if record["config"]["model"]["parameters"] != expected_params:
        raise ValueError("Component model has an unintended parameter change.")
    if record["outcome"]["source_sha256"] != check["source_sha256"] or record["outcome"]["input_sha256"] != check["input_sha256"]:
        raise ValueError("Actual component provenance differs from preflight.")
    expected_profiles = check["partitions"][3:] if reference_seed == 20260920 else check["partitions"]
    profiles = record["outcome"]["individual_fit_profiles"]
    if len(profiles) != len(expected_profiles):
        raise ValueError("Wrong number of fits in the component.")
    for profile, expected in zip(profiles, expected_profiles):
        for key in ("feature_count", "training_rows", "validation_rows", "bin_counts", "exact_column_indices"):
            if profile[key] != expected[key]:
                raise ValueError(f"Component fit differs from preflight: {key}")
        if profile["binning_strategy"] != "quantile":
            raise ValueError("Unintended binning change.")
    x, y, development, names, metadata = _input_data(root, suite)
    _, _, variant = configuration(root)
    for fold, (train, valid) in enumerate(_stratified_folds(y[development], 3, suite["seed"]), 1):
        cp = record["outcome"]["model_checkpoints"][fold - 1]
        if _sha256(root / cp["path"]) != cp["sha256"] or cp["fold"] != fold:
            raise ValueError("Component checkpoint fingerprint differs.")
        model, saved = HistogramGradientBoostingClassifier.load_checkpoint(root / cp["path"])
        np.testing.assert_array_equal(saved["training_row_indices"], development[train])
        for key in ("training_row_indices_sha256", "training_labels_sha256"):
            if saved[key] != check["partitions"][fold - 1][key]:
                raise ValueError("Checkpoint training boundary differs.")
        if any(getattr(model, key) != value for key, value in expected_params.items()):
            raise ValueError("Actual checkpoint parameter differs.")
        ct, cv, cn = tree_feature_matrices(x[development[train]], x[development[valid]], names, metadata, variant["preprocessing"])
        verify_feature_order(cn, saved, check["partitions"][fold - 1]["output_names"], reference_seed)
        np.testing.assert_array_equal(model.predict_proba(cv)[:, 1], arrays["probabilities"][valid])
        detail = record["outcome"]["nested_threshold_evaluation"]["folds"][fold - 1]
        inner = load_inner(root, detail["inner_oof_predictions_artifact"])
        np.testing.assert_array_equal(inner["training_row_indices"], development[train])
        np.testing.assert_array_equal(inner["outer_validation_row_indices"], development[valid])
        np.testing.assert_array_equal(inner["labels"], y[development[train]])
        ids = np.zeros(train.size, dtype=np.int8)
        for number, (_, iv) in enumerate(_stratified_folds(y[development[train]], 2, suite["nested_threshold"]["inner_seed"] + fold), 1):
            ids[iv] = number
        np.testing.assert_array_equal(inner["fold_ids"], ids)
        if int(inner["outer_fold"][0]) != fold or int(inner["inner_seed"][0]) != suite["nested_threshold"]["inner_seed"] + fold:
            raise ValueError("Inner partition metadata differs.")
        aligned_mean(inner, inner)
        threshold = best_f1_threshold(inner["labels"], inner["probabilities"]).threshold
        if threshold != detail["inner_threshold"]:
            raise ValueError("Component threshold cannot be reproduced.")
        np.testing.assert_array_equal(arrays["nested_thresholds"][valid], np.full(valid.size, threshold))
        fm, fc = runner._metrics(arrays["labels"][valid], arrays["probabilities"][valid], threshold)
        if fm != detail["outer_evaluation_metrics"] or fc != detail["outer_evaluation_confusion_counts"]:
            raise ValueError("Component fold metrics differ.")
        print(f"VERIFIED seed={reference_seed} fold={fold}: checkpoint probabilities and inner threshold identical", flush=True)
        del model, ct, cv
        gc.collect()
    return arrays


def evaluate_mean(root, first_record, second_record, suite, directory):
    first, second = read(root, first_record), read(root, second_record)
    a, b = _arrays(root, first), _arrays(root, second)
    for key in ("development_indices", "fold_ids", "labels"):
        np.testing.assert_array_equal(a[key], b[key])
    probabilities = (a["probabilities"] + b["probabilities"]) / 2.0
    thresholds = np.empty(probabilities.size, dtype=np.float64)
    details = []
    for fold, (af, bf) in enumerate(zip(first["outcome"]["nested_threshold_evaluation"]["folds"],
                                      second["outcome"]["nested_threshold_evaluation"]["folds"]), 1):
        ai = load_inner(root, af["inner_oof_predictions_artifact"])
        bi = load_inner(root, bf["inner_oof_predictions_artifact"])
        ip = aligned_mean(ai, bi)
        threshold = best_f1_threshold(ai["labels"], ip).threshold
        valid = a["fold_ids"] == fold
        thresholds[valid] = threshold
        fm, fc = runner._metrics(a["labels"][valid], probabilities[valid], threshold)
        info = runner._save_inner_oof_predictions(directory / "inner_oof_predictions", "phase26_uniform_mean", fold,
            int(ai["inner_seed"][0]), ai["training_row_indices"], ai["outer_validation_row_indices"],
            ai["fold_ids"], ai["labels"], ip)
        info["path"] = str(Path(info["path"]).relative_to(root))
        details.append({"outer_fold": fold, "inner_threshold": threshold,
            "inner_selection_metrics": runner._metrics(ai["labels"], ip, threshold)[0],
            "outer_evaluation_metrics": fm, "outer_evaluation_confusion_counts": fc,
            "inner_oof_predictions_artifact": info})
    predictions = probabilities >= thresholds
    counts, metrics = classification_metrics(a["labels"], predictions)
    metrics.update(log_loss=binary_log_loss(a["labels"], probabilities),
                   average_precision=average_precision(a["labels"], probabilities))
    selected_thresholds = [detail["inner_threshold"] for detail in details]
    nested = {"protocol": "Fixed equal weights; thresholds selected on the mean of aligned component INNER OOF probabilities only.",
        "folds": details, "threshold_summary": {"minimum": min(selected_thresholds), "maximum": max(selected_thresholds),
            "mean": float(np.mean(selected_thresholds))}, "pooled_metrics": metrics, "pooled_confusion_counts": vars(counts)}
    exploratory = best_f1_threshold(a["labels"], probabilities).threshold
    em, ec = runner._metrics(a["labels"], probabilities, exploratory)
    config = copy.deepcopy(first["config"])
    config.update(experiment_name="phase26_phase14_uniform_mean_two_seeds", hypothesis=suite["hypothesis"])
    config["model"] = {"name": "numpy_histogram_boosting_uniform_mean", "parameters": {
        "weights": [0.5, 0.5], "component_model_parameters": [first["config"]["model"]["parameters"], second["config"]["model"]["parameters"]]}}
    config["threshold"] = {"value": exploratory, "selection_method": "Pooled outer OOF exploratory only; decisions use inner-only nested thresholds.", "fixed_reference": 0.5}
    path, fingerprint = runner._save_oof_predictions(directory / "oof_predictions", config["experiment_name"],
        a["development_indices"], a["fold_ids"], a["labels"], probabilities, exploratory, thresholds)
    outcome = {"status": "completed", "evaluation_partition": "development_only_3_fold_oof", "metrics": em,
        "confusion_counts": ec, "nested_threshold_evaluation": nested,
        "source_sha256": first["outcome"]["source_sha256"], "input_sha256": first["outcome"]["input_sha256"],
        "environment": first["outcome"]["environment"],
        "runtime_seconds": first["outcome"]["runtime_seconds"] + second["outcome"]["runtime_seconds"],
        "runtime_basis": "Total new component training for this study; historical baseline outer training excluded.",
        "component_records": [{"path": str(p.relative_to(root)), "sha256": _sha256(p)} for p in (first_record, second_record)],
        "component_checkpoints": [first["outcome"]["model_checkpoints"], second["outcome"]["model_checkpoints"]],
        "oof_predictions_artifact": {"path": str(path.relative_to(root)), "sha256": fingerprint,
            "arrays": ["development_indices", "fold_ids", "labels", "probabilities", "exploratory_threshold", "nested_thresholds", "nested_predictions"]}}
    return save_experiment(config, outcome, directory / "records")


def compare(root, recovery_path, second_path):
    suite, _, _ = configuration(root)
    check = read(root, PREFLIGHT)
    verify_frozen(root, suite, check, allow_verification_update=True)
    (root / ARTIFACTS / "verification_runner.py").write_bytes((root / HELPER).read_bytes())
    recovery, second = read(root, recovery_path), read(root, second_path)
    verify_component(root, recovery, suite, check, 20260920)
    verify_component(root, second, suite, check, 20260921)
    directory = root / ARTIFACTS / "mean"
    existing = list((directory / "records").glob("*.json"))
    mean_path = existing[0] if len(existing) == 1 else evaluate_mean(root, recovery_path, second_path, suite, directory)
    mean = read(root, mean_path)
    baseline = read(root, BASELINE)
    ba, ra, sa, ma = (_arrays(root, item) for item in (baseline, recovery, second, mean))
    np.testing.assert_array_equal(ra["nested_predictions"], ba["nested_predictions"])
    np.testing.assert_array_equal(ma["probabilities"], (ba["probabilities"] + sa["probabilities"]) / 2.0)
    for fold, (rf, sf, mf) in enumerate(zip(recovery["outcome"]["nested_threshold_evaluation"]["folds"],
                                         second["outcome"]["nested_threshold_evaluation"]["folds"],
                                         mean["outcome"]["nested_threshold_evaluation"]["folds"]), 1):
        averaged = aligned_mean(load_inner(root, rf["inner_oof_predictions_artifact"]),
                                load_inner(root, sf["inner_oof_predictions_artifact"]))
        stored = load_inner(root, mf["inner_oof_predictions_artifact"])
        np.testing.assert_array_equal(stored["probabilities"], averaged)
        threshold = best_f1_threshold(stored["labels"], averaged).threshold
        if threshold != mf["inner_threshold"]:
            raise ValueError("Mean threshold does not come from mean inner probabilities.")
        np.testing.assert_array_equal(ma["nested_thresholds"][ma["fold_ids"] == fold], np.full(np.sum(ma["fold_ids"] == fold), threshold))
    candidates = []
    bn = baseline["outcome"]["nested_threshold_evaluation"]
    for title, path, record, arrays in (("Secondo seed", second_path, second, sa), ("Media uniforme", mean_path, mean, ma)):
        np.testing.assert_array_equal(arrays["nested_predictions"], arrays["probabilities"] >= arrays["nested_thresholds"])
        counts, metrics = classification_metrics(arrays["labels"], arrays["nested_predictions"])
        metrics.update(log_loss=binary_log_loss(arrays["labels"], arrays["probabilities"]),
                       average_precision=average_precision(arrays["labels"], arrays["probabilities"]))
        nested = record["outcome"]["nested_threshold_evaluation"]
        if metrics != nested["pooled_metrics"] or vars(counts) != nested["pooled_confusion_counts"]:
            raise ValueError("Candidate pooled metrics cannot be reproduced.")
        deltas = {key: value - bn["pooled_metrics"][key] for key, value in metrics.items()}
        fold_deltas = [cf["outer_evaluation_metrics"]["f1"] - bf["outer_evaluation_metrics"]["f1"]
                       for cf, bf in zip(nested["folds"], bn["folds"])]
        published = _publish(path, root)
        candidates.append({"name": title, "record": str(published.relative_to(root)), "nested": nested,
            "metric_deltas": deltas, "fold_f1_deltas": fold_deltas, "positive_fold_count": sum(d > 0 for d in fold_deltas),
            "meets_plan_promotion_rule": deltas["f1"] > 0 and sum(d > 0 for d in fold_deltas) >= 2,
            "paired_conditional_f1_delta_interval_95": _paired_interval(arrays["labels"], ba["nested_predictions"], arrays["nested_predictions"]),
            "decision_changes": {"recovered_false_negatives": int(np.sum((arrays["labels"] == 1) & (ba["nested_predictions"] == 0) & (arrays["nested_predictions"] == 1))),
                "lost_true_positives": int(np.sum((arrays["labels"] == 1) & (ba["nested_predictions"] == 1) & (arrays["nested_predictions"] == 0))),
                "added_false_positives": int(np.sum((arrays["labels"] == 0) & (ba["nested_predictions"] == 0) & (arrays["nested_predictions"] == 1))),
                "removed_false_positives": int(np.sum((arrays["labels"] == 0) & (ba["nested_predictions"] == 1) & (arrays["nested_predictions"] == 0)))}})
    disagreement = ba["nested_predictions"] != sa["nested_predictions"]
    summary = {"phase": 26, "baseline_record": BASELINE, "baseline_nested": bn, "candidates": candidates,
        "training_helper_artifact": {"path": f"{ARTIFACTS}/executed_runner.py", "sha256": check["helper_sha256"]},
        "verification_helper_artifact": {"path": f"{ARTIFACTS}/verification_runner.py", "sha256": _sha256(root / HELPER)},
        "verification_metadata_fix": "Legacy Phase 14 checkpoints lack output_feature_names. Their order is verified from frozen preprocessing, bin edges and exact probabilities; new checkpoints must retain the field. No additional training.",
        "baseline_inner_recovery_record": str(recovery_path.relative_to(root)), "baseline_inner_recovery_sha256": _sha256(recovery_path),
        "baseline_thresholds_reproduced_exactly": True, "component_prediction_checks_exact": True,
        "weights": [0.5, 0.5], "seeds": [20260920, 20260921], "new_fit_count": 15,
        "runtime_seconds": recovery["outcome"]["runtime_seconds"] + second["outcome"]["runtime_seconds"],
        "runtime_basis": "Six recovered baseline inner fits and nine second-seed fits; excludes preflight and final checks.",
        "environment": second["outcome"]["environment"], "final_model_changed": False,
        "bootstrap_seed": 20260929, "bootstrap_repetitions": 20000,
        "bootstrap_scope": "Fixed fitted models and thresholds; excludes training and repeated model-selection uncertainty.",
        "component_complementarity": {"probability_correlation": float(np.corrcoef(ba["probabilities"], sa["probabilities"])[0, 1]),
            "mean_absolute_probability_difference": float(np.mean(np.abs(ba["probabilities"] - sa["probabilities"]))),
            "prediction_disagreements": int(np.sum(disagreement)),
            "positive_prediction_disagreements": int(np.sum(disagreement & (ba["labels"] == 1))),
            "negative_prediction_disagreements": int(np.sum(disagreement & (ba["labels"] == 0)))}}
    (root / COMPARISON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(root, summary)
    print(json.dumps({"phase": 26, "results": [{"name": c["name"], "f1": c["nested"]["pooled_metrics"]["f1"],
          "delta_f1": c["metric_deltas"]["f1"], "fold_deltas": c["fold_f1_deltas"],
          "meets_plan_rule": c["meets_plan_promotion_rule"]} for c in candidates]}), flush=True)
    return summary


def write_report(root, summary):
    base = summary["baseline_nested"]
    candidates = summary["candidates"]
    lines = ["# Phase 26 — Secondo seed e media uniforme dei due booster", "", "## Protocollo", "",
        "Phase 14 invariata: 295 feature inclusa HAREHAB1, preprocessing codebook, binning a quantili, NaN gestiti dagli alberi; 200 alberi, profondità 5, learning rate 0,05, foglia minima 200, 64 bin, 128 feature candidate, L2=1.", "",
        "Seed 20260920 e 20260921; pesi fissati a 0,5 prima della valutazione. Stesse 262.508 righe development e stessi tre fold esterni. Due fold interni per selezionare ogni soglia. La media usa la soglia ottimizzata sulle probabilità medie INTERNE, mai la media delle soglie individuali.", "",
        "Tre fit esterni Phase 14 riutilizzati dopo verifica numerica. Sei fit interni recuperati: soglie e metriche storiche riprodotte esattamente. Nove nuovi fit per il secondo seed. Nessun nuovo training richiesto per calcolare la media.", "",
        "## Risultati annidati", "", "| Metrica | Phase 14 | Secondo seed | Media uniforme |", "| --- | ---: | ---: | ---: |"]
    for key in ("f1", "precision", "recall", "average_precision", "log_loss"):
        lines.append(f"| {key} | {base['pooled_metrics'][key]:.9f} | {candidates[0]['nested']['pooled_metrics'][key]:.9f} | {candidates[1]['nested']['pooled_metrics'][key]:.9f} |")
    for c in candidates:
        lines += ["", f"## {c['name']}", "", "| Fold | F1 | Delta Phase 14 | Soglia interna |", "| --- | ---: | ---: | ---: |"]
        for fold, item in enumerate(c["nested"]["folds"], 1):
            lines.append(f"| {fold} | {item['outer_evaluation_metrics']['f1']:.9f} | {c['fold_f1_deltas'][fold-1]:+.9f} | {item['inner_threshold']:.9f} |")
        ci = c["paired_conditional_f1_delta_interval_95"]
        lines += ["", f"Delta F1 {c['metric_deltas']['f1']:+.9f}; fold favorevoli {c['positive_fold_count']}/3. Criterio del piano {'soddisfatto' if c['meets_plan_promotion_rule'] else 'non soddisfatto'}.", "",
                  f"Intervallo bootstrap appaiato descrittivo 95%: [{ci[0]:+.9f}; {ci[1]:+.9f}].", "",
                  "| Esito | Phase 14 | Candidato | Differenza |", "| --- | ---: | ---: | ---: |"]
        for key in ("true_positives", "false_negatives", "false_positives", "true_negatives"):
            b, v = base["pooled_confusion_counts"][key], c["nested"]["pooled_confusion_counts"][key]
            lines.append(f"| {key} | {b} | {v} | {v-b:+d} |")
        lines += ["", f"Record: [{Path(c['record']).name}](../experiments/{Path(c['record']).name})."]
    comp = summary["component_complementarity"]
    lines += ["", "## Complementarità e limiti", "",
        f"Correlazione delle probabilità dei due seed: {comp['probability_correlation']:.9f}; differenza assoluta media: {comp['mean_absolute_probability_difference']:.9f}. Decisioni discordanti: {comp['prediction_disagreements']}, di cui {comp['positive_prediction_disagreements']} nei positivi e {comp['negative_prediction_disagreements']} nei negativi. Le decisioni usano le rispettive soglie interne: il disaccordo riflette sia score sia soglia.", "",
        "Bootstrap: 20.000 ripetizioni, seed 20260929, modelli e soglie fissi; non misura la variabilità da riaddestramento o dalla selezione ripetuta dei candidati. La validazione annidata protegge la scelta della soglia; i fold development sono stati usati per più confronti e non costituiscono un test finale intatto.", "",
        "## Verifiche e costo", "",
        f"15 nuovi fit completati in {summary['runtime_seconds']/60:.2f} minuti, esclusi controlli preliminari e finali. Probabilità dei checkpoint verificate esattamente; righe, etichette, fold interni, hash e soglie ricontrollati. Componenti e media conservano OOF esterni e interni.", "",
        f"Ambiente: Python {summary['environment']['python']}, NumPy {summary['environment']['numpy']}. Compatibilità con Python 3.9 / NumPy 1.23.1 ancora da verificare.", "",
        "La configurazione finale resta Phase 14 durante il confronto. Un eventuale candidato favorevole richiede una decisione esplicita prima del training finale.", "",
        "Confronto completo: [JSON](../experiments/phase26_two_seeds_comparison.json). Controlli: [preflight](phase26_two_seeds_preflight.json)."]
    (root / REPORT).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(root):
    suite, reference, variant = configuration(root)
    check = read(root, PREFLIGHT)
    verify_frozen(root, suite, check)
    directory = root / ARTIFACTS
    if list(directory.rglob("*.npz")) or list(directory.rglob("*.pkl")) or list(directory.rglob("records/*.json")):
        raise ValueError("Phase 26 training artifacts already exist; inspect them before rerunning.")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "executed_runner.py").write_bytes((root / HELPER).read_bytes())
    (directory / "training_preflight.json").write_bytes((root / PREFLIGHT).read_bytes())
    with (directory / "run_stdout.log").open("w", encoding="utf-8") as log:
        tee = _Tee(sys.stdout, log)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            recovery = recover_baseline_inner(root, suite, reference, variant, check, directory / "baseline_recovery")
            print("START SECOND SEED: nine fits with seed 20260921", flush=True)
            records = run_logged_suite(root / SUITE, root, directory / "second_seed" / "records")
            if len(records) != 1:
                raise ValueError("Expected one second-seed record.")
            verify_frozen(root, suite, check)
            compare(root, recovery, records[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run", "compare"))
    parser.add_argument("records", nargs="*", type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    if args.action == "preflight":
        preflight(root)
    elif args.action == "run":
        run(root)
    elif len(args.records) == 2:
        compare(root, *(p.resolve() for p in args.records))
    else:
        parser.error("compare requires recovery and second-seed records")


if __name__ == "__main__":
    main()

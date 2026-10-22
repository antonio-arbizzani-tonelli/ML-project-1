"""Execute report experiments from official CSVs, without original local artifacts.

Examples:
  python -m tools.reproduce_report list
  python -m tools.reproduce_report run --experiment all
  python -m tools.reproduce_report run --experiment two_seeds top100 random3
  python -m tools.reproduce_report run --smoke --experiment all
  python -m tools.reproduce_report check --require-complete
"""
from __future__ import annotations

import os
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
import json
from pathlib import Path

from tools.report_reproduction import ReportRun, paired_comparisons, read, write

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs/report_experiments.json"


def selected_experiments(catalog, names):
    entries = catalog["experiments"]
    if names == ["all"]:
        return entries
    if names == ["report"]:
        return [e for e in entries if e.get("in_pdf", True)]
    if names == ["historical"]:
        return [e for e in entries if e["kind"] != "calibration"]
    if names == ["calibration"]:
        return [e for e in entries if e["kind"] == "calibration"]
    known = {e["id"] for e in entries}
    if set(names) - known:
        raise ValueError("Unknown experiments: " + ", ".join(sorted(set(names)-known)))
    return [e for e in entries if e["id"] in names]


def checked_metrics(output, result):
    """Recompute metrics from freshly generated OOF arrays, never a status flag."""
    import numpy as np
    from src.evaluation import average_precision, binary_log_loss, classification_metrics, best_f1_threshold
    from src.run_preprocessing_cv import _sha256
    path = output / result["oof"]
    if _sha256(path) != result["oof_sha256"]:
        raise ValueError("Generated OOF fingerprint differs.")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    y, p, rows = arrays["labels"], arrays["probabilities"], arrays["rows"]
    if y.shape != p.shape or rows.shape != y.shape or np.unique(rows).size != len(rows) or not np.all(np.isfinite(p)):
        raise ValueError("Generated OOF arrays are incomplete or misaligned.")
    if result["experiment"].startswith("cal_"):
        threshold = best_f1_threshold(y, p).threshold
        if threshold != result["threshold"]:
            raise ValueError("Calibration threshold does not match fresh OOF predictions.")
    else:
        threshold = arrays["thresholds"]
        if threshold.shape != p.shape or not np.all(np.isfinite(threshold)) or set(np.unique(arrays["folds"])) != {1, 2, 3}:
            raise ValueError("Nested thresholds or fold assignments are incomplete.")
    counts, metrics = classification_metrics(y, p >= threshold)
    metrics.update(average_precision=average_precision(y, p), log_loss=binary_log_loss(y, p))
    for key, value in result["metrics"].items():
        if key not in metrics or abs(metrics[key]-value) > 1e-12:
            raise ValueError("Stored metric differs from generated OOF arrays: " + key)
    if result.get("counts") is not None and result["counts"] != vars(counts):
        raise ValueError("Stored confusion counts differ from generated OOF arrays.")
    if result.get("record"):
        source = output / result["record"]
        if _sha256(source) != result["record_sha256"]:
            raise ValueError("Generated configured-runner record fingerprint differs.")
    return metrics


def check(output, catalog, require_complete=False, smoke=False, expected_ids=None):
    checked, missing, different = [], [], []
    manifest = read(output / "run.json")
    from tools.report_reproduction import fingerprint
    run_id = fingerprint(manifest)
    for entry in catalog["experiments"]:
        path = output / "records" / (entry["id"] + ".json")
        if not path.is_file():
            missing.append(entry["id"])
            continue
        result = read(path)
        if result["run_fingerprint"] != run_id or result["mode"] != ("smoke" if smoke else "full"):
            raise ValueError("Record belongs to another run or validation scope.")
        metrics = checked_metrics(output, result)
        if not smoke:
            expected = entry.get("expected", {})
            mismatch = {key: metrics[key]-value if key in metrics else "missing" for key, value in expected.items()
                        if key not in metrics or abs(metrics[key]-value) > catalog["full_metric_tolerance"]}
            if not expected:
                mismatch["missing_historical_target"] = True
            if entry.get("expected_threshold") is not None and abs(result["threshold"]-entry["expected_threshold"]) > 1e-6:
                mismatch["threshold"] = result["threshold"]-entry["expected_threshold"]
            if entry.get("expected_features") is not None and result.get("features") != entry["expected_features"]:
                mismatch["features"] = "different"
        else:
            mismatch = {}
        if result.get("submission"):
            import csv
            import numpy as np
            from run import _read_sample_submission_ids
            from src.run_preprocessing_cv import _sha256
            path = output / result["submission"]
            if _sha256(path) != result["submission_sha256"]:
                raise ValueError("Generated CSV fingerprint differs.")
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.reader(stream)
                if next(reader) != ["Id", "Prediction"]:
                    raise ValueError("Generated CSV has an incorrect header.")
                pairs = np.asarray([[int(a), int(b)] for a, b in reader], dtype=np.int64)
            if pairs.shape != (result["test_rows"], 2) or not np.all(np.isin(pairs[:, 1], [-1, 1])):
                raise ValueError("Generated CSV has incorrect rows or labels.")
            ids = np.load(output / "data/id_test_int64.npy", allow_pickle=False)
            np.testing.assert_array_equal(pairs[:, 0], ids[:len(pairs)] if smoke else ids)
            if int(np.sum(pairs[:, 1] == 1)) != result["positive_predictions"]:
                raise ValueError("Generated CSV positive count differs.")
            if not smoke:
                expected = entry.get("expected_submission", {})
                if expected.get("sha256") and _sha256(path) != expected["sha256"]:
                    mismatch["submission_sha256"] = "different"
                if expected.get("positive_predictions") is not None and result["positive_predictions"] != expected["positive_predictions"]:
                    mismatch["positive_predictions"] = "different"
        if mismatch:
            different.append({"experiment": entry["id"], "differences": mismatch})
        checked.append(entry["id"])
    required = set(expected_ids if expected_ids is not None else [e["id"] for e in catalog["experiments"] if e.get("in_pdf", True)])
    missing_required = sorted(required-set(checked))
    data = read(output / "data_claims.json")
    data_passed = data["comparison"]["status"] == ("not_applicable" if smoke else "matched")
    pairs = paired_comparisons(output)
    paired_passed = all(p["comparison"]["status"] == ("not_applicable" if smoke else "matched") for p in pairs.values())
    if require_complete and {"raw321", "phase14", "phase34"} <= required and set(pairs) != {"cleaning", "split_search"}:
        paired_passed = False
    report = {"scope": "reduced workflow" if smoke else "full scientific metrics", "completed": checked,
              "missing": missing, "different": different,
              "missing_required": missing_required, "data_claims_passed": data_passed, "paired_comparisons_passed": paired_passed,
              "passed": bool(checked) and not different and data_passed and paired_passed and (not require_complete or not missing_required),
              "external_feedback": catalog["external_evidence"]}
    write(output / "validation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "run", "check", "data"))
    parser.add_argument("--experiment", nargs="+", default=["all"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw/dataset")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/reproduction/report")
    parser.add_argument("--smoke", action="store_true", help="Reduced rows/trees/updates; never reproduces report metrics.")
    parser.add_argument("--smoke-rows", type=int, default=6000)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve() / ("smoke" if args.smoke else "full")
    catalog = read(CATALOG)
    if args.command == "list":
        print(json.dumps(catalog, indent=2))
        return
    if args.command == "check":
        # Validate current source/data/cache identity and recompute descriptive
        # claims before inspecting results; a previously saved PASS is no input.
        manifest = read(output / "run.json")
        run = ReportRun(ROOT, args.data_dir, output, args.smoke, manifest["smoke_rows"] or args.smoke_rows)
        run.data_claims()
        paired_comparisons(output)
        result = check(output, catalog, args.require_complete, args.smoke)
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit(2)
        return
    run = ReportRun(ROOT, args.data_dir, output, args.smoke, args.smoke_rows)
    data = run.data_claims()
    if args.command == "data":
        print(json.dumps(data, indent=2))
        return
    for entry in selected_experiments(catalog, args.experiment):
        print("START " + entry["id"], flush=True)
        result = run.execute(entry)
        print(json.dumps({"completed": entry["id"], "metrics": result["metrics"], "comparison": result["comparison"]}), flush=True)
        paired_comparisons(output)
    result = check(output, catalog, args.require_complete, args.smoke,
                   [e["id"] for e in selected_experiments(catalog, args.experiment)])
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

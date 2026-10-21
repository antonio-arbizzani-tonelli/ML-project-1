"""Build Phase 18 + restricted Phase 24 submissions with leaf L2 of 1 and 5.

Each independent pipeline calibrates its own threshold on three full-training
OOF fits, fits all labeled rows, and delivers its CSV immediately. At most two
single-thread fitting subprocesses run concurrently. The historical candidate
engine is loaded privately and frozen alongside this wrapper; existing runs and
sources are never reconfigured or overwritten.
"""

from __future__ import annotations

import os
for _thread_name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_name] = "1"

import argparse
import concurrent.futures
import copy
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/boosting_artifacts/combined18_24_20261013"
DEFAULT_DELIVERY = "results/aicrowd_combined18_24_20261013"
VARIANTS = ("combined18_24_l2_1", "combined18_24_l2_5")
REFERENCES = {variant: f"configs/experiments/{variant}.json" for variant in VARIANTS}
PREPROCESSING_PATH = "configs/experiments/combined18_24_preprocessing.json"
PREPROCESSING_VARIANT = "combined18_24"
EXPECTED_SHAPE = (328135, 109379, 321)
EXPECTED_FEATURES = {variant: 297 for variant in VARIANTS}
EXPECTED_FOLD_SEED = 20260919
ENGINE_PATH = "tools/final_promising_candidates.py"
ENGINE_SHA256 = "e79e9a416e40845aabed15bf0bbbab624e302d60c1f6876c34f847f3794260b3"
HISTORICAL = "src/numpy_boosting_phase14.py"
HISTORICAL_SHA256 = "6fede57915b07a6ea7675292833f8ad79269e665f09977bffd2a9562ce1aaf98"
HISTORICAL_SUITES = (
    "configs/experiments/phase18_boosting_diet_frequency_features.json",
    "configs/experiments/phase24_original_restricted_imputation.json",
)
DIET_EXCLUSIONS = ("FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1")
MODEL_PARAMETERS = {
    "learning_rate": 0.05, "max_depth": 5, "min_samples_leaf": 200,
    "n_bins": 64, "l2_regularization": 1.0, "max_features": 128,
    "random_seed": 20260920, "n_estimators": 200,
}
_DELIVERY_LOCK = threading.Lock()


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_manifest(path):
    path = Path(path).resolve()
    if checksum(path) != path.with_suffix(".sha256").read_text(encoding="ascii").strip():
        raise ValueError("The prepared manifest was changed.")
    manifest = read_json(path)
    if "combined_engine" not in manifest or "combined_wrapper" not in manifest:
        raise ValueError("The manifest is not a prepared combined Phase 18 + 24 run.")
    return manifest


def _engine(manifest=None):
    """Load a private engine instance, preserving every imported live module."""
    if manifest is None:
        path, expected = ROOT / ENGINE_PATH, ENGINE_SHA256
    else:
        path = Path(manifest["combined_engine"]["path"])
        expected = manifest["combined_engine"]["sha256"]
    if checksum(path) != expected:
        raise ValueError(f"Frozen base engine changed: {path}.")
    name = "_combined18_24_engine_" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        module._combined_engine_sha256 = expected
    elif module._combined_engine_sha256 != expected:
        raise ValueError("Another base engine version is already loaded privately.")
    module.ROOT = Path(manifest["root"]) if manifest else ROOT
    module.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    module.DEFAULT_DELIVERY = DEFAULT_DELIVERY
    module.REFERENCES = REFERENCES
    module.VARIANTS = VARIANTS
    module.EXPECTED_SHAPE = EXPECTED_SHAPE
    module.EXPECTED_FEATURES = EXPECTED_FEATURES
    module.EXPECTED_FOLD_SEED = EXPECTED_FOLD_SEED
    return module


def _suite_plan(root, suite):
    entries = read_json(root / suite["preprocessing_config_path"])["variants"]
    matches = [entry["preprocessing"] for entry in entries
               if entry["experiment_name"] == suite["preprocessing_variant"]]
    if len(matches) != 1:
        raise ValueError("Each suite must select one recorded preprocessing plan.")
    return matches[0]


def validate_configuration(root=ROOT):
    """Require exact historical composition and L2 as the sole model difference."""
    root = Path(root)
    historical_suites = [read_json(root / path) for path in HISTORICAL_SUITES]
    diet, imputation = [_suite_plan(root, suite) for suite in historical_suites]
    common_diet = copy.deepcopy(diet)
    common_diet["exclude_features"] = [name for name in diet["exclude_features"] if name not in DIET_EXCLUSIONS]
    common_imputation = copy.deepcopy(imputation)
    common_imputation.pop("imputation")
    if common_diet != common_imputation or not all(name in diet["exclude_features"] for name in DIET_EXCLUSIONS):
        raise ValueError("Historical Phase 18 and restricted Phase 24 foundations differ.")
    expected_plan = copy.deepcopy(imputation)
    expected_plan["exclude_features"] = list(diet["exclude_features"])
    if set(DIET_EXCLUSIONS) & set(imputation["imputation"]["anchor_features"]):
        raise ValueError("Diet exclusions overlap the historical imputation anchors.")
    controls = [root / path for path in HISTORICAL_SUITES]
    controls.extend(root / suite["preprocessing_config_path"] for suite in historical_suites)
    suites = []
    for variant, l2 in zip(VARIANTS, (1.0, 5.0)):
        suite = read_json(root / REFERENCES[variant])
        if (suite["preprocessing_config_path"] != PREPROCESSING_PATH
                or suite["preprocessing_variant"] != PREPROCESSING_VARIANT
                or suite["fold_count"] != 3 or suite["seed"] != EXPECTED_FOLD_SEED
                or len(suite["models"]) != 1 or suite["models"][0]["checkpoints"] != [200]
                or suite["models"][0]["parameters"] != {
                    key: value for key, value in dict(MODEL_PARAMETERS, l2_regularization=l2).items()
                    if key != "n_estimators"}
                or _suite_plan(root, suite) != expected_plan
                or suite["data"] != historical_suites[0]["data"]
                or suite["feature_metadata_path"] != historical_suites[0]["feature_metadata_path"]):
            raise ValueError(f"Combined configuration differs from the authorized historical plan: {variant}.")
        suites.append(suite)
    if suites[0]["data"] != suites[1]["data"]:
        raise ValueError("Combined variants must share the official dataset.")
    return controls


def prepare(root=ROOT, output=DEFAULT_OUTPUT):
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = read_manifest(manifest_path)
        _engine(manifest).frozen(manifest, inputs=True)
        print(f"PREPARED (reused): {manifest_path}", flush=True)
        return manifest
    controls = validate_configuration(root)
    engine = _engine()
    # The base engine freezes all numerical modules, official inputs and folds.
    # Its copied driver becomes our frozen base engine rather than the launcher.
    with redirect_stdout(io.StringIO()):
        manifest = engine.prepare(root, output)
    snapshots = output / "source_snapshot"
    frozen_engine = Path(manifest["driver"])
    live_wrapper = Path(__file__).resolve()
    frozen_wrapper = snapshots / "final_combined_18_24.py"
    shutil.copyfile(live_wrapper, frozen_wrapper)
    controls.extend((live_wrapper, frozen_wrapper, ROOT / ENGINE_PATH, frozen_engine))
    manifest["control_sha256"].update({str(path): checksum(path) for path in controls})
    manifest["combined_engine"] = {"path": str(frozen_engine), "sha256": checksum(frozen_engine),
                                   "original_path": str(ROOT / ENGINE_PATH), "original_sha256": ENGINE_SHA256}
    manifest["combined_wrapper"] = {"path": str(frozen_wrapper), "sha256": checksum(frozen_wrapper),
                                    "original_path": str(live_wrapper), "original_sha256": checksum(live_wrapper)}
    manifest["driver"] = str(frozen_wrapper)
    manifest["scope"] = "Exact Phase 18 diet exclusions plus restricted Phase 24 imputation; leaf L2 of 1 versus 5."
    manifest["execution_protocol"] = "At most two fitting subprocesses; each variant runs its three OOF fits, calibration, full fit and immediate CSV delivery."
    write_json(manifest_path, manifest)
    manifest_path.with_suffix(".sha256").write_text(checksum(manifest_path) + "\n", encoding="ascii")
    _engine(manifest).frozen(manifest, inputs=True)
    print(f"PREPARED six OOF fits and two full fits; no training: {manifest_path}", flush=True)
    return manifest


def worker(manifest_path, variant, partition):
    manifest = read_manifest(manifest_path)
    return _engine(manifest).worker(manifest_path, variant, partition)


def _launch(manifest_path, variant, partition):
    manifest = read_manifest(manifest_path)
    return _engine(manifest)._launch(manifest_path, variant, partition)


def _deliver_variant(manifest, variant, record):
    """Atomically add one completed CSV without waiting for the other pipeline."""
    engine = _engine(manifest)
    directory = Path(manifest["delivery"])
    directory.mkdir(parents=True, exist_ok=True)
    delivery_path = directory / "manifest.json"
    run_manifest_path = Path(manifest["output"]) / "manifest.json"
    with _DELIVERY_LOCK, engine.file_lock(directory / "delivery.lock"):
        existing = read_json(delivery_path) if delivery_path.exists() else None
        if existing is not None and existing["run_manifest_sha256"] != checksum(run_manifest_path):
            raise ValueError("Delivery directory belongs to a different frozen run.")
        source = Path(record["submission"]["path"])
        if checksum(source) != record["submission"]["sha256"]:
            raise ValueError("Source submission changed before delivery.")
        ids = np.load(manifest["data"]["test_ids"])
        scores = np.load(engine.task_directory(manifest, variant, "full") / "probabilities.npy")
        predictions = np.where(scores >= record["decision_threshold"], 1, -1).astype(np.int8)
        validation = engine.validate_submission(source, ids, predictions)
        destination = directory / source.name
        if destination.exists() and checksum(destination) != checksum(source):
            raise ValueError(f"Refusing to overwrite a different existing delivery CSV: {destination}.")
        temporary = destination.with_suffix(".tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
        records = {} if existing is None else existing["variants"]
        records[variant] = {
            "csv": destination.name, "sha256": checksum(destination), **validation,
            "features": manifest["variants"][variant]["expected_features"],
            "model_parameters": manifest["variants"][variant]["model_parameters"],
            "preprocessing_variant": manifest["variants"][variant]["preprocessing_variant"],
            "decision_threshold": record["decision_threshold"],
            "threshold_source": manifest["threshold_protocol"],
            "threshold_training_rows": manifest["training_rows"],
            "calibration_metrics": record["calibration_metrics"],
            "positive_predictions": record["positive_predictions"],
            "negative_predictions": record["negative_predictions"],
            "oof_artifact": record["oof_artifact"],
            "configuration_sha256": record["configuration_sha256"],
            "full_model": record["full_model"], "test_probabilities": record["test_probabilities"],
            "official_test_f1": None,
        }
        now = datetime.now(timezone.utc).isoformat()
        delivery = {
            "created_at_utc": now if existing is None else existing["created_at_utc"],
            "updated_at_utc": now, "run_manifest": str(run_manifest_path),
            "run_manifest_sha256": checksum(run_manifest_path),
            "historical_booster_sha256": manifest["sources"]["booster"]["original_sha256"],
            "fold_seed": manifest["fold_seed"], "folds_sha256": manifest["folds_sha256"],
            "input_sha256": manifest["input_sha256"], "calibration_warning": manifest["calibration_warning"],
            "variants": records, "complete": set(records) == set(VARIANTS), "uploaded_to_aicrowd": False,
        }
        write_json(delivery_path, delivery)
    print(f"CSV READY {variant}: {destination}", flush=True)
    return {"directory": str(directory), "manifest": str(delivery_path), "csv": str(destination)}


def _pipeline(manifest_path, manifest, variant):
    engine = _engine(manifest)
    for fold in (1, 2, 3):
        partition = f"fold_{fold}"
        if engine.completed(manifest, variant, partition) is None:
            _launch(manifest_path, variant, partition)
    engine.calibrate(manifest, variant)
    if engine.completed(manifest, variant, "full") is None:
        _launch(manifest_path, variant, "full")
    engine.frozen(manifest, inputs=True)
    record = engine.publish_submission(manifest, variant)
    _deliver_variant(manifest, variant, record)
    return record


def run(manifest_path, workers=2):
    if workers not in (1, 2):
        raise ValueError("Use one or at most two isolated single-thread workers.")
    manifest_path = Path(manifest_path).resolve()
    manifest = read_manifest(manifest_path)
    engine = _engine(manifest)
    engine.frozen(manifest, inputs=True)
    # Initialize the isolated numerical package once before concurrent calibration.
    engine.modules(manifest)
    with engine.file_lock(Path(manifest["output"]) / "run.lock"):
        records = {}
        if workers == 1:
            for variant in VARIANTS:
                records[variant] = _pipeline(manifest_path, manifest, variant)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_pipeline, manifest_path, manifest, variant): variant for variant in VARIANTS}
                try:
                    for future in concurrent.futures.as_completed(futures):
                        records[futures[future]] = future.result()
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
        engine.frozen(manifest, inputs=True)
        delivery_path = Path(manifest["delivery"]) / "manifest.json"
        comparison = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(), "scope": manifest["scope"],
            "calibration_warning": manifest["calibration_warning"],
            "variants": {variant: records[variant] for variant in VARIANTS},
            "official_test_f1_comparison": None,
            "delivery": {"directory": manifest["delivery"], "manifest": str(delivery_path), "sha256": checksum(delivery_path)},
            "next_action": "User uploads both combined candidate CSVs to AIcrowd and records their official test F1.",
        }
        write_json(Path(manifest["output"]) / "comparison.json", comparison)
        print(json.dumps(comparison, indent=2), flush=True)
        return comparison


def status(manifest_path):
    manifest = read_manifest(manifest_path)
    result = _engine(manifest).status(manifest_path)
    delivery_path = Path(manifest["delivery"]) / "manifest.json"
    delivery = read_json(delivery_path) if delivery_path.exists() else {"variants": {}}
    result["csv_ready"] = {variant: str(Path(manifest["delivery"]) / record["csv"])
                           for variant, record in delivery["variants"].items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status", "worker"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--partition", choices=("fold_1", "fold_2", "fold_3", "full"))
    args = parser.parse_args()
    manifest_path = args.manifest or args.output_dir / "manifest.json"
    if args.command == "prepare":
        prepare(ROOT, args.output_dir)
    elif args.command == "run":
        run(manifest_path, args.workers)
    elif args.command == "status":
        print(json.dumps(status(manifest_path), indent=2))
    else:
        if args.variant is None or args.partition is None:
            parser.error("worker requires --variant and --partition")
        worker(manifest_path, args.variant, args.partition)


if __name__ == "__main__":
    main()

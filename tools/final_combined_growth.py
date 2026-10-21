"""Extend the verified features256 base or separately search all 297 features.

The first extends copied 200-tree checkpoints to 300 trees; the second uses
297 candidate features per node with the original 200 trees. At most two
single-thread fits run concurrently across all OOF and full-data tasks.
Original fitted bins, random generator state and training scores are preserved
for extension. Each variant calibrates its own OOF threshold and delivers its
CSV as soon as its three OOF fits and full fit are complete.
"""

from __future__ import annotations

import os
for _thread_name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_name] = "1"

import argparse
import copy
import concurrent.futures
from contextlib import redirect_stdout
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import pickle
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/boosting_artifacts/combined_growth_20261014"
DEFAULT_DELIVERY = "results/aicrowd_combined_growth_20261014"
VARIANTS = ("combined18_24_trees300", "combined18_24_features297")
REFERENCES = {variant: f"configs/experiments/{variant}.json" for variant in VARIANTS}
PREPROCESSING_PATH = "configs/experiments/combined18_24_preprocessing.json"
PREPROCESSING_VARIANT = "combined18_24"
PREPROCESSING_SHA256 = "26a90a950aa29a1fa179d19978d0aaa4138c808b01c85108d1283cb2a5bebcc0"
EXPECTED_SHAPE = (328135, 109379, 321)
EXPECTED_FEATURES = {variant: 297 for variant in VARIANTS}
EXPECTED_FOLD_SEED = 20260919
BASE_SUITE = "configs/experiments/combined18_24_features256.json"
BASE_SUITE_SHA256 = "8b7b874dca513e6e28d2d1fba93a6a41f9f2346526930f51a59a944a67049218"
BASE_FEATURES = 297
ORIGIN_MANIFEST_PATH = "results/boosting_artifacts/combined_tuning_20261014/manifest.json"
ORIGIN_MANIFEST_SHA256 = "55dd6a95fea2fae58993eaa5c6fc1debf5ea62f838e15a0915a6871db56b7633"
ORIGIN_VARIANT = "combined18_24_features256"
EXTENSION_VARIANT = "combined18_24_trees300"
ENGINE_PATH = "tools/final_promising_candidates.py"
ENGINE_SHA256 = "e79e9a416e40845aabed15bf0bbbab624e302d60c1f6876c34f847f3794260b3"
SCHEDULER_PATH = "tools/final_combined_18_24.py"
SCHEDULER_SHA256 = "6ffd2b93585fbe82475158a5d66a5239ebd48ae8cf4e80a47de71032e2c1571b"
MODEL_PARAMETERS = {
    "learning_rate": 0.05, "max_depth": 5, "min_samples_leaf": 200,
    "n_bins": 64, "l2_regularization": 1.0, "max_features": 256,
    "random_seed": 20260920, "n_estimators": 200,
}
VARIANT_OVERRIDES = {
    "combined18_24_trees300": {"n_estimators": 300},
    "combined18_24_features297": {"max_features": 297},
}


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_private(path, expected, prefix):
    if checksum(path) != expected:
        raise ValueError(f"Frozen source changed: {path}.")
    name = prefix + hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:16]
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
        module._growth_source_sha256 = expected
    elif module._growth_source_sha256 != expected:
        raise ValueError("Another scheduler source version is already loaded privately.")
    return module


def _scheduler(manifest=None):
    source = manifest["growth_scheduler"] if manifest else {
        "path": ROOT / SCHEDULER_PATH, "sha256": SCHEDULER_SHA256}
    module = _load_private(source["path"], source["sha256"], "_combined_growth_scheduler_")
    if not hasattr(module, "_growth_original_engine"):
        module._growth_original_engine = module._engine
        module._growth_original_pipeline = module._pipeline
        module._growth_original_deliver = module._deliver_variant
    module.ROOT = Path(manifest["root"]) if manifest else ROOT
    module.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    module.DEFAULT_DELIVERY = DEFAULT_DELIVERY
    module.VARIANTS = VARIANTS
    module.REFERENCES = REFERENCES
    module.EXPECTED_FEATURES = EXPECTED_FEATURES
    module.EXPECTED_SHAPE = EXPECTED_SHAPE
    module.EXPECTED_FOLD_SEED = EXPECTED_FOLD_SEED
    module._engine = _engine
    module._launch = _launch
    module._pipeline = _pipeline
    module._deliver_variant = _deliver_variant
    module.read_manifest = read_manifest
    module.write_json = _write_scheduler_json
    return module


def _engine(manifest=None):
    engine = _scheduler(manifest)._growth_original_engine(manifest)
    if not hasattr(engine, "_growth_original_modules"):
        engine._growth_original_modules = engine.modules

        def growth_modules(current):
            if "extension_origin" not in current:
                return engine._growth_original_modules(current)
            origin = _read_origin(current)
            # Preserve the verified original package and its directory. Loading
            # origin pickles into another class namespace would fail isinstance.
            if current["sources"] != origin["sources"] or current["source_package"] != origin["source_package"]:
                raise ValueError("Growth numerical dependencies differ from the trusted origin.")
            return _origin_engine(origin).modules(origin)

        engine.modules = growth_modules
    return engine


def _origin_engine(origin):
    source = origin["combined_engine"]
    if source["sha256"] != ENGINE_SHA256:
        raise ValueError("The origin engine is not the trusted historical engine.")
    engine = _load_private(source["path"], source["sha256"], "_growth_origin_engine_")
    engine.ROOT = Path(origin["root"])
    engine.VARIANTS = tuple(origin["variants"])
    return engine


def _read_origin(manifest):
    source = manifest["extension_origin"]
    path = Path(source["manifest_path"])
    if checksum(path) != source["manifest_sha256"]:
        raise ValueError("The trusted origin manifest changed.")
    return read_json(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_scheduler_json(path, value):
    if Path(path).name == "comparison.json":
        value["next_action"] = "User uploads the two separate Phase 18+24, L2=1 tuning CSVs to AIcrowd and records their official scores."
    write_json(path, value)


def read_manifest(path):
    path = Path(path).resolve()
    if checksum(path) != path.with_suffix(".sha256").read_text(encoding="ascii").strip():
        raise ValueError("The prepared manifest was changed.")
    manifest = read_json(path)
    if not all(key in manifest for key in ("growth_wrapper", "growth_scheduler", "combined_engine", "extension_origin")):
        raise ValueError("The manifest is not a prepared two-variant combined growth run.")
    return manifest


def validate_configuration(root=ROOT):
    """Reject any numerical or preprocessing change beyond the two fixed axes."""
    root = Path(root)
    if (checksum(root / BASE_SUITE) != BASE_SUITE_SHA256
            or checksum(root / PREPROCESSING_PATH) != PREPROCESSING_SHA256):
        raise ValueError("The recorded Phase 18+24, L2=1 foundation was changed.")
    base = read_json(root / BASE_SUITE)
    if (base["preprocessing_config_path"] != PREPROCESSING_PATH
            or base["preprocessing_variant"] != PREPROCESSING_VARIANT
            or base["fold_count"] != 3 or base["seed"] != EXPECTED_FOLD_SEED
            or len(base["models"]) != 1
            or base["models"][0]["parameters"] != {
                key: value for key, value in MODEL_PARAMETERS.items() if key != "n_estimators"}
            or base["models"][0]["checkpoints"] != [200]):
        raise ValueError("The recorded Phase 18+24, L2=1 model differs from the fixed base.")
    plans = read_json(root / PREPROCESSING_PATH)["variants"]
    if len([item for item in plans if item["experiment_name"] == PREPROCESSING_VARIANT]) != 1:
        raise ValueError("The unchanged combined preprocessing must identify exactly one plan.")
    for variant in VARIANTS:
        suite = read_json(root / REFERENCES[variant])
        expected = copy.deepcopy(base)
        expected["experiment_prefix"] = variant
        expected["artifact_dir"] = f"results/boosting_artifacts/combined_growth_20261014/{variant}"
        expected["hypothesis"] = suite.get("hypothesis")
        overrides = VARIANT_OVERRIDES[variant]
        if "n_estimators" in overrides:
            expected["models"][0]["checkpoints"] = [overrides["n_estimators"]]
            expected["nested_threshold"]["model_checkpoints"][0]["checkpoints"] = [overrides["n_estimators"]]
        else:
            expected["models"][0]["parameters"].update(overrides)
        if suite != expected or not isinstance(suite.get("hypothesis"), str):
            raise ValueError(f"Tuning configuration differs from its authorized one-parameter change: {variant}.")
    return [root / BASE_SUITE, root / PREPROCESSING_PATH]


def _origin_provenance(path):
    path = Path(path).resolve()
    if checksum(path) != path.with_suffix(".sha256").read_text(encoding="ascii").strip():
        raise ValueError("Origin manifest integrity failed.")
    origin = read_json(path)
    engine = _origin_engine(origin)
    engine.frozen(origin, inputs=True)
    if ORIGIN_VARIANT not in origin["variants"]:
        raise ValueError("The trusted features256 origin is missing.")
    directory = Path(origin["output"]) / ORIGIN_VARIANT
    config_path = directory / "calibrated_final_config.json"
    calibration_path = directory / "calibration.json"
    calibration = read_json(calibration_path)
    config = read_json(config_path)
    recipe = origin["variants"][ORIGIN_VARIANT]
    if (checksum(config_path) != calibration["configuration_sha256"]
            or config["model"] != recipe["model_parameters"]
            or config["preprocessing"]["plan"] != recipe["preprocessing"]
            or config["source_dependencies"] != origin["sources"]
            or config["input_sha256"] != origin["input_sha256"]
            or config["folds_sha256"] != origin["folds_sha256"]):
        raise ValueError("The calibrated origin configuration changed.")
    controls = dict(origin["control_sha256"])
    controls.update({str(path): checksum(path), str(path.with_suffix(".sha256")): checksum(path.with_suffix(".sha256")),
                     str(config_path): checksum(config_path), str(calibration_path): checksum(calibration_path)})
    oof = directory / "full_train_oof.npz"
    if checksum(oof) != calibration["oof_artifact"]["sha256"] or checksum(oof) != config["oof_sha256"]:
        raise ValueError("The trusted origin OOF artifact changed.")
    controls[str(oof)] = checksum(oof)
    checkpoints = {}
    for partition in ("fold_1", "fold_2", "fold_3", "full"):
        completed = engine.completed(origin, ORIGIN_VARIANT, partition)
        if completed is None:
            raise ValueError(f"The trusted origin fit is incomplete: {partition}.")
        task = directory / partition
        model_path = task / completed["model"]["file"]
        completed_path = task / "completed.json"
        probability_path = task / completed["probabilities"]["file"]
        controls.update({str(model_path): completed["model"]["sha256"],
                         str(completed_path): checksum(completed_path),
                         str(probability_path): completed["probabilities"]["sha256"]})
        checkpoints[partition] = {
            "original_path": str(model_path), "sha256": completed["model"]["sha256"],
            "completed_path": str(completed_path), "completed_sha256": checksum(completed_path),
            "signature": completed["signature"], "trees": completed["trees"],
        }
    return origin, controls, {
        "manifest_path": str(path), "manifest_sha256": checksum(path), "variant": ORIGIN_VARIANT,
        "parameters": recipe["model_parameters"], "source_package": origin["source_package"],
        "configuration": str(config_path), "configuration_sha256": checksum(config_path),
        "checkpoints": checkpoints,
        "operation": "Copy trusted checkpoints unchanged, then append trees with native continue_fit; preserve fitted bins, RNG and training scores.",
    }


def prepare(root=ROOT, output=DEFAULT_OUTPUT, origin_manifest_path=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = read_manifest(manifest_path)
        _engine(manifest).frozen(manifest, inputs=True)
        print(f"PREPARED (reused): {manifest_path}", flush=True)
        return manifest
    controls = validate_configuration(root)
    if origin_manifest_path is None and checksum(root / ORIGIN_MANIFEST_PATH) != ORIGIN_MANIFEST_SHA256:
        raise ValueError("The production origin differs from the recorded features256 run.")
    origin, origin_controls, derivation = _origin_provenance(origin_manifest_path or root / ORIGIN_MANIFEST_PATH)
    engine = _engine()
    with redirect_stdout(io.StringIO()):
        manifest = engine.prepare(root, output)
    recipe = origin["variants"][ORIGIN_VARIANT]
    if (manifest["input_sha256"] != origin["input_sha256"]
            or manifest["environment"] != origin["environment"]
            or manifest["folds_sha256"] != origin["folds_sha256"]
            or manifest["feature_names"] != origin["feature_names"]):
        raise ValueError("Growth inputs, environment, rows or folds differ from the trusted base.")
    for variant in VARIANTS:
        current = manifest["variants"][variant]
        parameters = dict(current["model_parameters"])
        base_parameters = dict(recipe["model_parameters"])
        if variant == EXTENSION_VARIANT:
            target = parameters.pop("n_estimators")
            if target <= base_parameters.pop("n_estimators"):
                raise ValueError("Extension must append trees beyond the trusted original target.")
        else:
            if parameters.pop("max_features") != current["expected_features"]:
                raise ValueError("The separate feature trial must consider all retained features.")
            base_parameters.pop("max_features")
        if (parameters != base_parameters or current["preprocessing"] != recipe["preprocessing"]
                or current["metadata"] != recipe["metadata"]
                or current["expected_features"] != recipe["expected_features"]):
            raise ValueError(f"Growth recipe differs beyond its one authorized axis: {variant}.")
    snapshots = output / "source_snapshot"
    for key, entry in origin["sources"].items():
        if manifest["sources"][key]["original_sha256"] != entry["original_sha256"]:
            raise ValueError("Growth numerical source differs from its trusted origin.")
        destination = snapshots / "frozen" / Path(entry["path"]).name
        shutil.copyfile(entry["path"], destination)
        origin_controls[str(destination)] = entry["sha256"]
    manifest["sources"] = copy.deepcopy(origin["sources"])
    manifest["source_package"] = origin["source_package"]
    frozen_origin_manifest = snapshots / "origin_manifest.json"
    shutil.copyfile(derivation["manifest_path"], frozen_origin_manifest)
    origin_controls[str(frozen_origin_manifest)] = derivation["manifest_sha256"]
    for partition, entry in derivation["checkpoints"].items():
        copy_path = output / EXTENSION_VARIANT / partition / "origin_model.pkl"
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(entry["original_path"], copy_path)
        if checksum(copy_path) != entry["sha256"]:
            raise ValueError("Copied origin checkpoint differs from its verified source.")
        entry["copy_path"] = str(copy_path)
        origin_controls[str(copy_path)] = entry["sha256"]
    manifest["extension_origin"] = derivation
    manifest["control_sha256"].update(origin_controls)
    frozen_engine = Path(manifest["driver"])
    live_scheduler = ROOT / SCHEDULER_PATH
    frozen_scheduler = snapshots / "final_combined_18_24.py"
    frozen_wrapper = snapshots / "final_combined_growth.py"
    live_wrapper = Path(__file__).resolve()
    for source, destination in ((live_scheduler, frozen_scheduler), (live_wrapper, frozen_wrapper)):
        shutil.copyfile(source, destination)
        controls.extend((source, destination))
    manifest["control_sha256"].update({str(path): checksum(path) for path in controls})
    manifest["combined_engine"] = {"path": str(frozen_engine), "sha256": checksum(frozen_engine),
                                   "original_path": str(ROOT / ENGINE_PATH), "original_sha256": ENGINE_SHA256}
    manifest["growth_scheduler"] = {"path": str(frozen_scheduler), "sha256": checksum(frozen_scheduler),
                                   "original_path": str(live_scheduler), "original_sha256": SCHEDULER_SHA256}
    manifest["growth_wrapper"] = {"path": str(frozen_wrapper), "sha256": checksum(frozen_wrapper),
                                 "original_path": str(live_wrapper), "original_sha256": checksum(live_wrapper)}
    manifest["driver"] = str(frozen_wrapper)
    manifest["scope"] = "Two separate changes to the verified Phase 18+24 features256 base: append trees to 300, or retain 200 trees and consider all 297 features."
    manifest["execution_protocol"] = "Globally fill at most two single-thread fitting slots across independent OOF/full tasks; calibrate only after three OOF fits and deliver only after calibration plus full fit."
    manifest["baseline_suite"] = {"path": str(root / BASE_SUITE), "sha256": checksum(root / BASE_SUITE)}
    manifest["authorized_parameter_changes"] = copy.deepcopy(VARIANT_OVERRIDES)
    manifest["tasks"] = [
        {"variant": variant, "partition": partition} for variant, partition in (
            (EXTENSION_VARIANT, "fold_1"), (VARIANTS[1], "fold_1"),
            (EXTENSION_VARIANT, "fold_2"), (EXTENSION_VARIANT, "fold_3"),
            (EXTENSION_VARIANT, "full"), (VARIANTS[1], "fold_2"),
            (VARIANTS[1], "full"), (VARIANTS[1], "fold_3"))]
    configuration_path = output / "frozen_configuration.json"
    configuration = read_json(configuration_path)
    configuration.update(sources=manifest["sources"], extension_origin=derivation)
    write_json(configuration_path, configuration)
    manifest["control_sha256"][str(configuration_path)] = checksum(configuration_path)
    write_json(manifest_path, manifest)
    manifest_path.with_suffix(".sha256").write_text(checksum(manifest_path) + "\n", encoding="ascii")
    _engine(manifest).frozen(manifest, inputs=True)
    print(f"PREPARED six OOF fits and two full fits; no training: {manifest_path}", flush=True)
    return manifest


def _checkpoint_metadata(manifest, variant, partition, signature):
    origin = manifest["extension_origin"]
    entry = origin["checkpoints"][partition]
    return {
        "variant": variant, "partition": partition, "signature": signature,
        "parameters": manifest["variants"][variant]["model_parameters"],
        "source_sha256": {key: value["sha256"] for key, value in manifest["sources"].items()},
        "environment": manifest["environment"], "input_sha256": manifest["input_sha256"],
        "folds_sha256": manifest["folds_sha256"],
        "manifest_sha256": checksum(Path(manifest["output"]) / "manifest.json"),
        "derivation": {
            "origin_manifest_sha256": origin["manifest_sha256"], "origin_variant": origin["variant"],
            "origin_model_sha256": entry["sha256"], "origin_trees": entry["trees"],
            "operation": "Native continue_fit from an unchanged copied checkpoint; append trees only.",
        },
    }


def _load_origin_copy(manifest, partition, signature, model_class):
    origin = _read_origin(manifest)
    provenance = manifest["extension_origin"]
    entry = provenance["checkpoints"][partition]
    if signature != entry["signature"]:
        raise ValueError("Extension matrices or ordered rows differ from the origin signature.")
    for path in (entry["copy_path"], entry["original_path"]):
        if checksum(path) != entry["sha256"]:
            raise ValueError("Copied or original origin checkpoint changed.")
    model, metadata = model_class.load_checkpoint(entry["copy_path"])
    expected = {
        "variant": provenance["variant"], "partition": partition, "signature": signature,
        "parameters": provenance["parameters"],
        "source_sha256": {key: value["sha256"] for key, value in origin["sources"].items()},
        "environment": origin["environment"], "input_sha256": origin["input_sha256"],
        "folds_sha256": origin["folds_sha256"],
    }
    if (metadata != expected or len(model.trees_) != entry["trees"]
            or len(model.trees_) != provenance["parameters"]["n_estimators"]
            or model.n_features_in_ != signature["training_shape"][1]
            or model.training_scores_.shape != (signature["training_shape"][0],)
            or not np.all(np.isfinite(model.training_scores_))
            or any(getattr(model, key) != value for key, value in provenance["parameters"].items())):
        raise ValueError("Origin checkpoint metadata, parameters or continuation state changed.")
    return model


def worker(manifest_path, variant, partition):
    manifest = read_manifest(manifest_path)
    engine = _engine(manifest)
    if variant != EXTENSION_VARIANT:
        return engine.worker(manifest_path, variant, partition)
    engine.frozen(manifest, inputs=True)
    directory = engine.task_directory(manifest, variant, partition)
    with engine.file_lock(directory / "worker.lock"):
        if engine.completed(manifest, variant, partition) is not None:
            print(f"CACHED {variant}/{partition}", flush=True)
            return
        loaded = engine.modules(manifest)
        train, labels, valid, signature = engine._matrices(manifest, variant, partition, loaded["preprocessing"])
        model_class = loaded["booster"].HistogramGradientBoostingClassifier
        original = _load_origin_copy(manifest, partition, signature, model_class)
        metadata = _checkpoint_metadata(manifest, variant, partition, signature)
        parameters = metadata["parameters"]
        target = parameters["n_estimators"]
        origin_trees = len(original.trees_)
        checkpoint = directory / "model.pkl"
        started = time.perf_counter()
        resumed_from = 0
        if checkpoint.exists():
            model, saved = model_class.load_checkpoint(checkpoint)
            resumed_from = len(model.trees_)
            if (saved != metadata or not origin_trees < resumed_from <= target
                    or model.n_features_in_ != train.shape[1]
                    or model.training_scores_.shape != labels.shape
                    or any(getattr(model, key) != value for key, value in parameters.items())
                    or pickle.dumps(model.binner_, protocol=pickle.HIGHEST_PROTOCOL)
                    != pickle.dumps(original.binner_, protocol=pickle.HIGHEST_PROTOCOL)):
                raise ValueError("Growth resume checkpoint differs from its frozen provenance or fitted bins.")
            prefix = next(model.staged_predict_proba(valid, [origin_trees]))[1]
            if not np.array_equal(prefix, original.predict_proba(valid)[:, 1]):
                raise ValueError("Growth resume changed the original tree prefix.")
        else:
            model = original
        initial_trees = len(model.trees_)
        steps = sorted(set(range(manifest["checkpoint_interval"], target + 1, manifest["checkpoint_interval"])) | {target})

        def progress(step, current):
            current.save_checkpoint(checkpoint, metadata)
            engine.write_json(directory / "progress.json", {
                "variant": variant, "partition": partition, "trees": step, "target": target,
                "elapsed_seconds": time.perf_counter() - started, "pid": os.getpid(),
                "origin_trees": origin_trees, "resumed_from_trees": resumed_from,
            })
            print(f"PROGRESS {variant}/{partition}: {step}/{target} trees, elapsed={time.perf_counter()-started:.1f}s", flush=True)

        if initial_trees < target:
            # No fitted state or metadata from the original checkpoint is
            # rewritten. Native continuation sets its truthful new target.
            model.continue_fit(train, labels, target - initial_trees,
                               [step for step in steps if step > initial_trees], progress)
        probabilities = model.predict_proba(valid)[:, 1]
        if probabilities.shape != (valid.shape[0],) or not np.all(np.isfinite(probabilities)):
            raise ValueError("Extended probabilities are not finite aligned vectors.")
        engine.save_array(directory / "probabilities.npy", probabilities)
        engine.frozen(manifest)
        result = {
            "variant": variant, "partition": partition, "signature": signature, "trees": len(model.trees_),
            "manifest_sha256": checksum(manifest_path), "resumed_from_trees": resumed_from,
            "continued_from_origin_trees": origin_trees, "new_trees_this_invocation": target - initial_trees,
            "derivation": metadata["derivation"], "fit_and_prediction_seconds": time.perf_counter() - started,
            "total_tree_seconds": float(np.sum(model.tree_fit_seconds_)),
            "model": {"file": "model.pkl", "sha256": checksum(checkpoint)},
            "probabilities": {"file": "probabilities.npy", "sha256": checksum(directory / "probabilities.npy")},
        }
        engine.write_json(directory / "completed.json", result)
        print(f"DONE {variant}/{partition}", flush=True)


def _launch(manifest_path, variant, partition):
    return _engine(read_manifest(manifest_path))._launch(manifest_path, variant, partition)


def _pipeline(manifest_path, manifest, variant):
    return _scheduler(manifest)._growth_original_pipeline(manifest_path, manifest, variant)


def _deliver_variant(manifest, variant, record):
    return _scheduler(manifest)._growth_original_deliver(manifest, variant, record)


def run(manifest_path, workers=2):
    if workers not in (1, 2):
        raise ValueError("Use one or at most two isolated single-thread workers.")
    manifest_path = Path(manifest_path).resolve()
    manifest = read_manifest(manifest_path)
    engine = _engine(manifest)
    engine.frozen(manifest, inputs=True)
    engine.modules(manifest)
    with engine.file_lock(Path(manifest["output"]) / "run.lock"):
        # A stopped parent can leave fitting subprocesses alive. Audit every
        # task lock before creating a globally bounded pool of new workers.
        with ExitStack() as locks:
            for task in manifest["tasks"]:
                locks.enter_context(engine.file_lock(
                    engine.task_directory(manifest, task["variant"], task["partition"]) / "worker.lock"))
        records = {}

        def finalize_ready():
            for variant in VARIANTS:
                if variant in records:
                    continue
                if any(engine.completed(manifest, variant, f"fold_{fold}") is None for fold in (1, 2, 3)):
                    continue
                calibration_path = Path(manifest["output"]) / variant / "calibration.json"
                if not calibration_path.exists():
                    engine.calibrate(manifest, variant)
                if engine.completed(manifest, variant, "full") is None:
                    continue
                engine.frozen(manifest, inputs=True)
                record = engine.publish_submission(manifest, variant)
                if variant == EXTENSION_VARIANT:
                    origin = manifest["extension_origin"]
                    record["derivation"] = {
                        "origin_manifest_sha256": origin["manifest_sha256"],
                        "origin_variant": origin["variant"], "origin_parameters": origin["parameters"],
                        "origin_model_sha256": {key: entry["sha256"] for key, entry in origin["checkpoints"].items()},
                        "operation": origin["operation"],
                    }
                _deliver_variant(manifest, variant, record)
                records[variant] = record

        finalize_ready()
        pending = [task for task in manifest["tasks"]
                   if engine.completed(manifest, task["variant"], task["partition"]) is None]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_launch, manifest_path, task["variant"], task["partition"]) for task in pending]
            try:
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    finalize_ready()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        finalize_ready()
        if set(records) != set(VARIANTS):
            raise ValueError("Some growth fits or calibrated submissions remain incomplete.")
        engine.frozen(manifest, inputs=True)
        delivery_path = Path(manifest["delivery"]) / "manifest.json"
        comparison = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(), "scope": manifest["scope"],
            "calibration_warning": manifest["calibration_warning"], "variants": records,
            "official_test_f1_comparison": None,
            "delivery": {"directory": manifest["delivery"], "manifest": str(delivery_path), "sha256": checksum(delivery_path)},
            "next_action": "User uploads the two separate growth CSVs to AIcrowd and records their official scores.",
        }
        write_json(Path(manifest["output"]) / "comparison.json", comparison)
        print(json.dumps(comparison, indent=2), flush=True)
        return comparison


def status(manifest_path):
    return _scheduler(read_manifest(manifest_path)).status(manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status", "worker"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--origin-manifest", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--partition", choices=("fold_1", "fold_2", "fold_3", "full"))
    args = parser.parse_args()
    manifest_path = args.manifest or args.output_dir / "manifest.json"
    if args.command == "prepare":
        prepare(ROOT, args.output_dir, args.origin_manifest)
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

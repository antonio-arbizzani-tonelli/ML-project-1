"""Fit native categorical activity splits on the verified features256 base.

Only EXRACT11 and EXRACT21 use native categorical split groups; all other
retained columns preserve historical numeric binning. The candidate fits new
200-tree models with the exact verified base preprocessing. At most two
single-thread fits run concurrently, with a full fit independent of the OOF
threshold. CSV delivery requires all three OOF fits and the full fit.
"""

from __future__ import annotations

import os
for _thread_name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_name] = "1"

import argparse
import concurrent.futures
import copy
from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/boosting_artifacts/combined_categorical_20261015"
DEFAULT_DELIVERY = "results/aicrowd_combined_categorical_20261015"
VARIANTS = ("combined18_24_native_categories",)
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
BASE_MANIFEST_PATH = "results/boosting_artifacts/combined_tuning_20261014/manifest.json"
BASE_MANIFEST_SHA256 = "55dd6a95fea2fae58993eaa5c6fc1debf5ea62f838e15a0915a6871db56b7633"
BASE_VARIANT = "combined18_24_features256"
CATEGORICAL_FEATURES = ("EXRACT11", "EXRACT21")
CATEGORICAL_SOURCE = "src/numpy_boosting_categorical.py"
HISTORICAL = "src/numpy_boosting_phase14.py"
HISTORICAL_SHA256 = "6fede57915b07a6ea7675292833f8ad79269e665f09977bffd2a9562ce1aaf98"
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
    "combined18_24_native_categories": {"histogram_strategy": "recursive"},
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
        module._categorical_source_sha256 = expected
    elif module._categorical_source_sha256 != expected:
        raise ValueError("Another scheduler source version is already loaded privately.")
    return module


def _scheduler(manifest=None):
    source = manifest["categorical_scheduler"] if manifest else {
        "path": ROOT / SCHEDULER_PATH, "sha256": SCHEDULER_SHA256}
    module = _load_private(source["path"], source["sha256"], "_combined_categorical_scheduler_")
    if not hasattr(module, "_categorical_original_engine"):
        module._categorical_original_engine = module._engine
        module._categorical_original_pipeline = module._pipeline
        module._categorical_original_deliver = module._deliver_variant
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
    engine = _scheduler(manifest)._categorical_original_engine(manifest)
    if not hasattr(engine, "_categorical_original_matrices"):
        engine._categorical_original_matrices = engine._matrices

        def checked_matrices(current, variant, partition, preprocessing):
            result = engine._categorical_original_matrices(current, variant, partition, preprocessing)
            reference = current.get("base_reference")
            if reference is not None and result[3] != reference["partitions"][partition]["signature"]:
                raise ValueError("Categorical matrices or ordered rows differ from the verified base.")
            return result

        engine._matrices = checked_matrices
    return engine


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
        value["next_action"] = "User uploads the native categorical candidate CSV to AIcrowd and records its official scores."
    if "run_manifest" in value and "historical_booster_sha256" in value:
        # Rename the legacy field before the scheduler's first atomic write,
        # while its delivery lock is held, including every cached rerun.
        value["booster_sha256"] = value.pop("historical_booster_sha256")
        value["baseline_classic_booster_sha256"] = HISTORICAL_SHA256
        value["booster_backend"] = "native_categorical"
        indices = value["variants"][VARIANTS[0]]["model_parameters"]["categorical_feature_indices"]
        value["categorical_columns"] = {
            "feature_names": list(CATEGORICAL_FEATURES), "indices": indices,
            "mapping": dict(zip(CATEGORICAL_FEATURES, indices)),
        }
    write_json(path, value)


def read_manifest(path):
    path = Path(path).resolve()
    if checksum(path) != path.with_suffix(".sha256").read_text(encoding="ascii").strip():
        raise ValueError("The prepared manifest was changed.")
    manifest = read_json(path)
    if not all(key in manifest for key in ("categorical_wrapper", "categorical_scheduler", "combined_engine", "categorical_columns")):
        raise ValueError("The manifest is not a prepared native categorical run.")
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
        expected["artifact_dir"] = f"results/boosting_artifacts/combined_categorical_20261015/{variant}"
        expected["hypothesis"] = suite.get("hypothesis")
        expected["models"][0]["parameters"].update(VARIANT_OVERRIDES[variant])
        expected["categorical_features"] = list(CATEGORICAL_FEATURES)
        if suite != expected or not isinstance(suite.get("hypothesis"), str):
            raise ValueError(f"Tuning configuration differs from its authorized one-parameter change: {variant}.")
    return [root / BASE_SUITE, root / PREPROCESSING_PATH]


def resolve_categorical_indices(output_names):
    """Resolve documented names in the actual retained feature order."""
    names = list(output_names)
    if len(set(names)) != len(names) or any(names.count(name) != 1 for name in CATEGORICAL_FEATURES):
        raise ValueError("The retained feature order must contain each categorical target exactly once.")
    return [names.index(name) for name in CATEGORICAL_FEATURES]


def _base_reference(root, base_manifest_path=None):
    path = Path(base_manifest_path or Path(root) / BASE_MANIFEST_PATH).resolve()
    if (checksum(path) != path.with_suffix(".sha256").read_text(encoding="ascii").strip()
            or (base_manifest_path is None and checksum(path) != BASE_MANIFEST_SHA256)):
        raise ValueError("The verified features256 base manifest changed.")
    base = read_json(path)
    recipe = base["variants"][BASE_VARIANT]
    if recipe["model_parameters"] != MODEL_PARAMETERS:
        raise ValueError("The verified base differs from the authorized 200-tree features256 model.")
    controls = {str(path): checksum(path), str(path.with_suffix(".sha256")): checksum(path.with_suffix(".sha256"))}
    partitions = {}
    for partition in ("fold_1", "fold_2", "fold_3", "full"):
        completed_path = Path(base["output"]) / BASE_VARIANT / partition / "completed.json"
        completed = read_json(completed_path)
        if (completed["variant"] != BASE_VARIANT or completed["partition"] != partition
                or completed["manifest_sha256"] != checksum(path)
                or completed["trees"] != MODEL_PARAMETERS["n_estimators"]):
            raise ValueError(f"The verified base fit differs from its frozen manifest: {partition}.")
        controls[str(completed_path)] = checksum(completed_path)
        partitions[partition] = {"completed_path": str(completed_path),
                                 "completed_sha256": checksum(completed_path),
                                 "signature": completed["signature"]}
    return {"manifest_path": str(path), "manifest_sha256": checksum(path),
            "variant": BASE_VARIANT, "partitions": partitions}, controls


def prepare(root=ROOT, output=DEFAULT_OUTPUT, base_manifest_path=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = read_manifest(manifest_path)
        _engine(manifest).frozen(manifest, inputs=True)
        print(f"PREPARED (reused): {manifest_path}", flush=True)
        return manifest
    controls = validate_configuration(root)
    reference, reference_controls = _base_reference(root, base_manifest_path)
    engine = _engine()
    with redirect_stdout(io.StringIO()):
        manifest = engine.prepare(root, output)
    if reference is not None:
        base = read_json(reference["manifest_path"])
        recipe = base["variants"][BASE_VARIANT]
        if (manifest["input_sha256"] != base["input_sha256"]
                or manifest["folds_sha256"] != base["folds_sha256"]
                or manifest["environment"] != base["environment"]
                or manifest["feature_names"] != base["feature_names"]
                or any(manifest["sources"][key]["original_sha256"] != value["original_sha256"]
                       for key, value in base["sources"].items())
                or any(item["preprocessing"] != recipe["preprocessing"]
                       or item["metadata"] != recipe["metadata"]
                       or item["expected_features"] != recipe["expected_features"]
                       for item in manifest["variants"].values())):
            raise ValueError("Categorical sources, inputs, folds or preprocessing differ from the verified base.")
    manifest["base_reference"] = reference
    manifest["control_sha256"].update(reference_controls)
    snapshots = output / "source_snapshot"
    if reference is None:
        # Synthetic or externally prepared datasets can resolve their actual
        # output order directly. Production uses the already verified base
        # signatures, avoiding another full matrix allocation during prepare.
        clean_engine = _load_private(manifest["driver"], ENGINE_SHA256, "_categorical_prepare_engine_")
        loaded = clean_engine.modules(manifest)
        train, labels, valid, signature = clean_engine._matrices(
            manifest, VARIANTS[0], "full", loaded["preprocessing"])
        output_names = signature["output_feature_names"]
        del train, labels, valid
    else:
        output_names = reference["partitions"]["full"]["signature"]["output_feature_names"]
        if any(item["signature"]["output_feature_names"] != output_names
               for item in reference["partitions"].values()):
            raise ValueError("The verified base has inconsistent retained feature orders.")
    indices = resolve_categorical_indices(output_names)
    if len(output_names) != manifest["variants"][VARIANTS[0]]["expected_features"]:
        raise ValueError("The categorical feature order has the wrong retained width.")
    manifest["variants"][VARIANTS[0]]["model_parameters"]["categorical_feature_indices"] = indices
    manifest["categorical_columns"] = {
        "feature_names": list(CATEGORICAL_FEATURES), "indices": indices,
        "mapping": dict(zip(CATEGORICAL_FEATURES, indices)), "output_feature_names": output_names,
        "index_space": "Retained preprocessing output columns, not raw dataset columns.",
        "fit_partition": "Category levels are learned only from each model's training rows.",
    }
    source = root / CATEGORICAL_SOURCE
    original = snapshots / "original" / source.name
    frozen_booster = snapshots / "frozen" / source.name
    shutil.copyfile(source, original)
    source_text = source.read_text(encoding="utf-8")
    frozen_text = source_text.replace("from src.", f"from {manifest['source_package']}.")
    if source_text == frozen_text:
        shutil.copyfile(source, frozen_booster)
    else:
        with frozen_booster.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(frozen_text)
    manifest["sources"]["booster"] = {
        "path": str(frozen_booster), "module": manifest["source_package"] + "." + source.stem,
        "sha256": checksum(frozen_booster), "original_path": str(source),
        "original_snapshot": str(original), "original_sha256": checksum(source),
        "import_rewrite": "src namespace to isolated frozen package" if source_text != frozen_text else None,
    }
    controls.extend((source, original))
    manifest["baseline_classic_booster_sha256"] = HISTORICAL_SHA256
    frozen_engine = Path(manifest["driver"])
    live_scheduler = ROOT / SCHEDULER_PATH
    frozen_scheduler = snapshots / "final_combined_18_24.py"
    frozen_wrapper = snapshots / "final_combined_categorical.py"
    live_wrapper = Path(__file__).resolve()
    for source, destination in ((live_scheduler, frozen_scheduler), (live_wrapper, frozen_wrapper)):
        shutil.copyfile(source, destination)
        controls.extend((source, destination))
    manifest["control_sha256"].update({str(path): checksum(path) for path in controls})
    manifest["combined_engine"] = {"path": str(frozen_engine), "sha256": checksum(frozen_engine),
                                   "original_path": str(ROOT / ENGINE_PATH), "original_sha256": ENGINE_SHA256}
    manifest["categorical_scheduler"] = {"path": str(frozen_scheduler), "sha256": checksum(frozen_scheduler),
                                   "original_path": str(live_scheduler), "original_sha256": SCHEDULER_SHA256}
    manifest["categorical_wrapper"] = {"path": str(frozen_wrapper), "sha256": checksum(frozen_wrapper),
                                 "original_path": str(live_wrapper), "original_sha256": checksum(live_wrapper)}
    manifest["driver"] = str(frozen_wrapper)
    manifest["scope"] = "One fresh 200-tree Phase 18+24 features256 trial with native categorical splits only for EXRACT11 and EXRACT21; numeric binning, other model parameters and 297-feature preprocessing are unchanged."
    manifest["execution_protocol"] = "Globally fill at most two single-thread fitting slots across independent OOF/full fits; calibrate after three OOF fits and deliver only after calibration plus full fit."
    manifest["baseline_suite"] = {"path": str(root / BASE_SUITE), "sha256": checksum(root / BASE_SUITE)}
    manifest["authorized_parameter_changes"] = {
        VARIANTS[0]: {"histogram_strategy": "recursive", "categorical_feature_indices": indices}}
    manifest["tasks"] = [{"variant": VARIANTS[0], "partition": partition}
                         for partition in ("fold_1", "full", "fold_2", "fold_3")]
    configuration_path = output / "frozen_configuration.json"
    configuration = read_json(configuration_path)
    configuration["base_reference"] = reference
    configuration["sources"] = manifest["sources"]
    configuration["variants"] = manifest["variants"]
    configuration["categorical_columns"] = manifest["categorical_columns"]
    configuration["baseline_classic_booster_sha256"] = HISTORICAL_SHA256
    write_json(configuration_path, configuration)
    manifest["control_sha256"][str(configuration_path)] = checksum(configuration_path)
    write_json(manifest_path, manifest)
    manifest_path.with_suffix(".sha256").write_text(checksum(manifest_path) + "\n", encoding="ascii")
    _engine(manifest).frozen(manifest, inputs=True)
    print(f"PREPARED three OOF fits and one full fit; no training: {manifest_path}", flush=True)
    return manifest


def worker(manifest_path, variant, partition):
    return _engine(read_manifest(manifest_path)).worker(manifest_path, variant, partition)


def _launch(manifest_path, variant, partition):
    return _engine(read_manifest(manifest_path))._launch(manifest_path, variant, partition)


def _pipeline(manifest_path, manifest, variant):
    return _scheduler(manifest)._categorical_original_pipeline(manifest_path, manifest, variant)


def _deliver_variant(manifest, variant, record):
    return _scheduler(manifest)._categorical_original_deliver(manifest, variant, record)


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
            raise ValueError("Some categorical fits or calibrated submissions remain incomplete.")
        engine.frozen(manifest, inputs=True)
        delivery_path = Path(manifest["delivery"]) / "manifest.json"
        comparison = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(), "scope": manifest["scope"],
            "calibration_warning": manifest["calibration_warning"], "variants": records,
            "official_test_f1_comparison": None,
            "delivery": {"directory": manifest["delivery"], "manifest": str(delivery_path), "sha256": checksum(delivery_path)},
            "next_action": "User uploads the native categorical candidate CSV to AIcrowd and records its official scores.",
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
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--partition", choices=("fold_1", "fold_2", "fold_3", "full"))
    args = parser.parse_args()
    manifest_path = args.manifest or args.output_dir / "manifest.json"
    if args.command == "prepare":
        prepare(ROOT, args.output_dir, args.base_manifest)
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

"""Calibrate A, B and A+B counts on the frozen Phase 18+24, L2=1 base.

Two independent pipelines start A and B; A+B starts when either finishes.
Each pipeline runs three full-training OOF folds, calibrates its threshold,
fits all labeled rows and delivers its CSV immediately. Historical numerical
sources and the existing two-pipeline scheduler are reused privately and frozen.
"""

from __future__ import annotations

import os
for _thread_name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_name] = "1"

import argparse
import copy
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/boosting_artifacts/count_candidates_20261013"
DEFAULT_DELIVERY = "results/aicrowd_count_candidates_20261013"
VARIANTS = ("count_a", "count_b", "count_ab")
REFERENCES = {variant: f"configs/experiments/{variant}.json" for variant in VARIANTS}
PREPROCESSING_PATH = "configs/experiments/count_candidates_preprocessing.json"
EXPECTED_SHAPE = (328135, 109379, 321)
EXPECTED_FEATURES = {"count_a": 299, "count_b": 299, "count_ab": 301}
EXPECTED_FOLD_SEED = 20260919
BASE_SUITE = "configs/experiments/combined18_24_l2_1.json"
BASE_FEATURES = 297
ENGINE_PATH = "tools/final_promising_candidates.py"
ENGINE_SHA256 = "e79e9a416e40845aabed15bf0bbbab624e302d60c1f6876c34f847f3794260b3"
SCHEDULER_PATH = "tools/final_combined_18_24.py"
SCHEDULER_SHA256 = "6ffd2b93585fbe82475158a5d66a5239ebd48ae8cf4e80a47de71032e2c1571b"
COUNT_SOURCE = "src/count_features.py"
COUNT_GROUPS = {"count_a": ["A"], "count_b": ["B"], "count_ab": ["A", "B"]}
GROUP_FEATURES = {"A": ["BLIND", "DECIDE", "USEEQUIP"],
                  "B": ["ADDEPEV2", "CHCCOPD1", "CHCKIDNY"]}
MODEL_PARAMETERS = {
    "learning_rate": 0.05, "max_depth": 5, "min_samples_leaf": 200,
    "n_bins": 64, "l2_regularization": 1.0, "max_features": 128,
    "random_seed": 20260920, "n_estimators": 200,
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
        module._counts_source_sha256 = expected
    elif module._counts_source_sha256 != expected:
        raise ValueError("Another source version is already loaded privately.")
    return module


def _scheduler(manifest=None):
    source = manifest["count_scheduler"] if manifest else {
        "path": ROOT / SCHEDULER_PATH, "sha256": SCHEDULER_SHA256}
    module = _load_private(source["path"], source["sha256"], "_counts_scheduler_")
    if not hasattr(module, "_counts_original_engine"):
        module._counts_original_engine = module._engine
        module._counts_original_pipeline = module._pipeline
        module._counts_original_deliver = module._deliver_variant
    module.ROOT = ROOT
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
        value["next_action"] = "User uploads the three count-candidate CSVs to AIcrowd and records their official test F1."
    write_json(path, value)


def read_manifest(path):
    path = Path(path).resolve()
    if checksum(path) != path.with_suffix(".sha256").read_text(encoding="ascii").strip():
        raise ValueError("The prepared manifest was changed.")
    manifest = read_json(path)
    if not all(key in manifest for key in ("count_wrapper", "count_scheduler", "combined_engine")):
        raise ValueError("The manifest is not a prepared A/B/A+B count run.")
    return manifest


def _engine(manifest=None):
    scheduler = _scheduler(manifest)
    engine = scheduler._counts_original_engine(manifest)
    if not hasattr(engine, "_counts_original_matrices"):
        engine._counts_original_matrices = engine._matrices
        engine._counts_original_modules = engine.modules

        def count_modules(current):
            loaded = engine._counts_original_modules(current)
            if "count_features" in current["sources"]:
                spec = current["sources"]["count_features"]
                loaded["count_features"] = engine.load_source(spec["module"], spec["path"], spec["sha256"])
            return loaded

        def count_matrices(current, variant, partition, preprocessing):
            plan = current["variants"][variant]["preprocessing"]
            recipe = plan["derived_counts"]
            base = dict(current)
            base["variants"] = dict(current["variants"])
            base["variants"][variant] = dict(current["variants"][variant])
            base["variants"][variant]["preprocessing"] = {
                key: value for key, value in plan.items() if key != "derived_counts"}
            base["variants"][variant]["expected_features"] = BASE_FEATURES
            train, labels, valid, signature = engine._counts_original_matrices(
                base, variant, partition, preprocessing)
            raw = np.load(current["data"]["train_features"], mmap_mode="r")
            raw_names = [name for group in recipe["groups"] for name in recipe["group_features"][group]]
            columns = [current["feature_names"].index(name) for name in raw_names]
            raw_counts = raw[:, columns]
            if partition == "full":
                train_source = raw_counts
                valid_source = np.load(current["data"]["test_features"], mmap_mode="r")[:, columns]
            else:
                folds = np.load(current["folds_path"])
                train_source, valid_source = raw_counts[folds != int(partition[-1])], raw_counts[folds == int(partition[-1])]
            helper = engine.modules(current)["count_features"]
            names = signature["output_feature_names"]
            train, output_names = helper.append_count_features(
                train, names, train_source, raw_names, recipe["groups"])
            valid, valid_names = helper.append_count_features(
                valid, names, valid_source, raw_names, recipe["groups"])
            if (output_names != valid_names
                    or len(output_names) != current["variants"][variant]["expected_features"]):
                raise ValueError(f"Count feature names or dimensions changed: {variant}.")
            signature.update(training_matrix_sha256=engine.array_checksum(train),
                             validation_matrix_sha256=engine.array_checksum(valid),
                             training_shape=list(train.shape), validation_shape=list(valid.shape),
                             output_feature_names=output_names)
            return train, labels, valid, signature

        engine.modules = count_modules
        engine._matrices = count_matrices
    return engine


def count_recipe(groups):
    return {"groups": list(groups), "group_features": {name: GROUP_FEATURES[name] for name in groups},
            "yes_code": 1, "no_code": 2, "valid_response_codes": [1, 2],
            "all_missing_yes_count": "NaN", "all_missing_answered_count": 0,
            "source_values": "original raw feature values before imputation",
            "fit_required": False, "retain_individual_features": True}


def validate_configuration(root=ROOT):
    root = Path(root)
    base = read_json(root / BASE_SUITE)
    base_plan_path = root / base["preprocessing_config_path"]
    base_plan = next(item["preprocessing"] for item in read_json(base_plan_path)["variants"]
                     if item["experiment_name"] == base["preprocessing_variant"])
    controls = [root / BASE_SUITE, base_plan_path]
    if (base["models"][0]["parameters"] != {key: value for key, value in MODEL_PARAMETERS.items()
                                            if key != "n_estimators"}
            or base["models"][0]["checkpoints"] != [200]):
        raise ValueError("The recorded Phase 18+24, L2=1 model differs from the fixed base.")
    for variant in VARIANTS:
        suite = read_json(root / REFERENCES[variant])
        matching = [item["preprocessing"] for item in read_json(root / PREPROCESSING_PATH)["variants"]
                    if item["experiment_name"] == variant]
        expected = copy.deepcopy(base_plan)
        expected["derived_counts"] = count_recipe(COUNT_GROUPS[variant])
        if (matching != [expected] or suite["preprocessing_config_path"] != PREPROCESSING_PATH
                or suite["preprocessing_variant"] != variant or suite["fold_count"] != 3
                or suite["seed"] != EXPECTED_FOLD_SEED or suite["models"] != base["models"]
                or suite["data"] != base["data"] or suite["feature_metadata_path"] != base["feature_metadata_path"]):
            raise ValueError(f"Count configuration differs from the authorized fixed base: {variant}.")
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
    with redirect_stdout(io.StringIO()):
        manifest = engine.prepare(root, output)
    snapshots = output / "source_snapshot"
    frozen_engine = Path(manifest["driver"])
    live_scheduler = ROOT / SCHEDULER_PATH
    frozen_scheduler = snapshots / "final_combined_18_24.py"
    frozen_wrapper = snapshots / "final_count_candidates.py"
    live_wrapper = Path(__file__).resolve()
    for source, destination in ((live_scheduler, frozen_scheduler), (live_wrapper, frozen_wrapper)):
        shutil.copyfile(source, destination)
        controls.extend((source, destination))
    source = root / COUNT_SOURCE
    original = snapshots / "original" / "count_features.py"
    frozen_helper = snapshots / "frozen" / "count_features.py"
    shutil.copyfile(source, original)
    shutil.copyfile(source, frozen_helper)
    controls.extend((source, original))
    manifest["sources"]["count_features"] = {
        "path": str(frozen_helper), "module": manifest["source_package"] + ".count_features",
        "sha256": checksum(frozen_helper), "original_path": str(source),
        "original_snapshot": str(original), "original_sha256": checksum(source), "import_rewrite": None}
    manifest["control_sha256"].update({str(path): checksum(path) for path in controls})
    manifest["combined_engine"] = {"path": str(frozen_engine), "sha256": checksum(frozen_engine),
                                   "original_path": str(ROOT / ENGINE_PATH), "original_sha256": ENGINE_SHA256}
    manifest["count_scheduler"] = {"path": str(frozen_scheduler), "sha256": checksum(frozen_scheduler),
                                   "original_path": str(live_scheduler), "original_sha256": SCHEDULER_SHA256}
    manifest["count_wrapper"] = {"path": str(frozen_wrapper), "sha256": checksum(frozen_wrapper),
                                 "original_path": str(live_wrapper), "original_sha256": checksum(live_wrapper)}
    manifest["driver"] = str(frozen_wrapper)
    manifest["scope"] = "Three fixed A, B and A+B raw yes/valid-response counts on exact Phase 18+24, leaf L2=1."
    manifest["execution_protocol"] = "At most two fitting subprocesses: A and B first, then A+B when a pipeline frees a slot; each CSV delivered immediately."
    configuration_path = output / "frozen_configuration.json"
    configuration = read_json(configuration_path)
    configuration["sources"] = manifest["sources"]
    write_json(configuration_path, configuration)
    manifest["control_sha256"][str(configuration_path)] = checksum(configuration_path)
    write_json(manifest_path, manifest)
    manifest_path.with_suffix(".sha256").write_text(checksum(manifest_path) + "\n", encoding="ascii")
    _engine(manifest).frozen(manifest, inputs=True)
    print(f"PREPARED nine OOF fits and three full fits; no training: {manifest_path}", flush=True)
    return manifest


def worker(manifest_path, variant, partition):
    return _engine(read_manifest(manifest_path)).worker(manifest_path, variant, partition)


def _launch(manifest_path, variant, partition):
    return _engine(read_manifest(manifest_path))._launch(manifest_path, variant, partition)


def _pipeline(manifest_path, manifest, variant):
    return _scheduler(manifest)._counts_original_pipeline(manifest_path, manifest, variant)


def _deliver_variant(manifest, variant, record):
    return _scheduler(manifest)._counts_original_deliver(manifest, variant, record)


def run(manifest_path, workers=2):
    if workers not in (1, 2):
        raise ValueError("Use one or at most two isolated single-thread workers.")
    manifest = read_manifest(manifest_path)
    return _scheduler(manifest).run(manifest_path, workers)


def status(manifest_path):
    return _scheduler(read_manifest(manifest_path)).status(manifest_path)


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

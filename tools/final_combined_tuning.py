"""Build two separate tuning submissions on exact Phase 18+24, leaf L2=1.

The first changes only quantile bins from 64 to 128; the second changes only
candidate features per node from 128 to 256. Two independent single-thread
fitting processes run concurrently, each calibrating its own three-fold OOF
threshold, fitting all labeled rows and delivering its CSV immediately.
Historical numerical sources and the existing scheduler are reused privately.
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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/boosting_artifacts/combined_tuning_20261014"
DEFAULT_DELIVERY = "results/aicrowd_combined_tuning_20261014"
VARIANTS = ("combined18_24_bins128", "combined18_24_features256")
REFERENCES = {variant: f"configs/experiments/{variant}.json" for variant in VARIANTS}
PREPROCESSING_PATH = "configs/experiments/combined18_24_preprocessing.json"
PREPROCESSING_VARIANT = "combined18_24"
PREPROCESSING_SHA256 = "26a90a950aa29a1fa179d19978d0aaa4138c808b01c85108d1283cb2a5bebcc0"
EXPECTED_SHAPE = (328135, 109379, 321)
EXPECTED_FEATURES = {variant: 297 for variant in VARIANTS}
EXPECTED_FOLD_SEED = 20260919
BASE_SUITE = "configs/experiments/combined18_24_l2_1.json"
BASE_SUITE_SHA256 = "0fa08a3bb87e9c966cfeece0502c93b7ccd5427a0ec4e9a1e7d5b40f82114fca"
BASE_FEATURES = 297
ENGINE_PATH = "tools/final_promising_candidates.py"
ENGINE_SHA256 = "e79e9a416e40845aabed15bf0bbbab624e302d60c1f6876c34f847f3794260b3"
SCHEDULER_PATH = "tools/final_combined_18_24.py"
SCHEDULER_SHA256 = "6ffd2b93585fbe82475158a5d66a5239ebd48ae8cf4e80a47de71032e2c1571b"
MODEL_PARAMETERS = {
    "learning_rate": 0.05, "max_depth": 5, "min_samples_leaf": 200,
    "n_bins": 64, "l2_regularization": 1.0, "max_features": 128,
    "random_seed": 20260920, "n_estimators": 200,
}
VARIANT_OVERRIDES = {
    "combined18_24_bins128": {"n_bins": 128},
    "combined18_24_features256": {"max_features": 256},
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
        module._tuning_source_sha256 = expected
    elif module._tuning_source_sha256 != expected:
        raise ValueError("Another scheduler source version is already loaded privately.")
    return module


def _scheduler(manifest=None):
    source = manifest["tuning_scheduler"] if manifest else {
        "path": ROOT / SCHEDULER_PATH, "sha256": SCHEDULER_SHA256}
    module = _load_private(source["path"], source["sha256"], "_combined_tuning_scheduler_")
    if not hasattr(module, "_tuning_original_engine"):
        module._tuning_original_engine = module._engine
        module._tuning_original_pipeline = module._pipeline
        module._tuning_original_deliver = module._deliver_variant
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
    return _scheduler(manifest)._tuning_original_engine(manifest)


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
    if not all(key in manifest for key in ("tuning_wrapper", "tuning_scheduler", "combined_engine")):
        raise ValueError("The manifest is not a prepared two-variant combined tuning run.")
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
        expected["artifact_dir"] = f"results/boosting_artifacts/combined_tuning_20261014/{variant}"
        expected["hypothesis"] = suite.get("hypothesis")
        expected["models"][0]["parameters"].update(VARIANT_OVERRIDES[variant])
        if suite != expected or not isinstance(suite.get("hypothesis"), str):
            raise ValueError(f"Tuning configuration differs from its authorized one-parameter change: {variant}.")
    return [root / BASE_SUITE, root / PREPROCESSING_PATH]


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
    frozen_wrapper = snapshots / "final_combined_tuning.py"
    live_wrapper = Path(__file__).resolve()
    for source, destination in ((live_scheduler, frozen_scheduler), (live_wrapper, frozen_wrapper)):
        shutil.copyfile(source, destination)
        controls.extend((source, destination))
    manifest["control_sha256"].update({str(path): checksum(path) for path in controls})
    manifest["combined_engine"] = {"path": str(frozen_engine), "sha256": checksum(frozen_engine),
                                   "original_path": str(ROOT / ENGINE_PATH), "original_sha256": ENGINE_SHA256}
    manifest["tuning_scheduler"] = {"path": str(frozen_scheduler), "sha256": checksum(frozen_scheduler),
                                   "original_path": str(live_scheduler), "original_sha256": SCHEDULER_SHA256}
    manifest["tuning_wrapper"] = {"path": str(frozen_wrapper), "sha256": checksum(frozen_wrapper),
                                 "original_path": str(live_wrapper), "original_sha256": checksum(live_wrapper)}
    manifest["driver"] = str(frozen_wrapper)
    manifest["scope"] = "Two separate changes to exact Phase 18+24, leaf L2=1: n_bins=128 versus max_features=256; all other parameters and 297-feature preprocessing unchanged."
    manifest["execution_protocol"] = "At most two isolated single-thread fitting subprocesses; each variant runs three OOF fits, calibration, full fit and immediate CSV delivery."
    manifest["baseline_suite"] = {"path": str(root / BASE_SUITE), "sha256": checksum(root / BASE_SUITE)}
    manifest["authorized_parameter_changes"] = copy.deepcopy(VARIANT_OVERRIDES)
    write_json(manifest_path, manifest)
    manifest_path.with_suffix(".sha256").write_text(checksum(manifest_path) + "\n", encoding="ascii")
    _engine(manifest).frozen(manifest, inputs=True)
    print(f"PREPARED six OOF fits and two full fits; no training: {manifest_path}", flush=True)
    return manifest


def worker(manifest_path, variant, partition):
    return _engine(read_manifest(manifest_path)).worker(manifest_path, variant, partition)


def _launch(manifest_path, variant, partition):
    return _engine(read_manifest(manifest_path))._launch(manifest_path, variant, partition)


def _pipeline(manifest_path, manifest, variant):
    return _scheduler(manifest)._tuning_original_pipeline(manifest_path, manifest, variant)


def _deliver_variant(manifest, variant, record):
    return _scheduler(manifest)._tuning_original_deliver(manifest, variant, record)


def run(manifest_path, workers=2):
    if workers not in (1, 2):
        raise ValueError("Use one or at most two isolated single-thread workers.")
    return _scheduler(read_manifest(manifest_path)).run(manifest_path, workers)


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

"""Check isolated structural trials and their frozen two-slot submission flow."""

import contextlib
import copy
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

from tests import test_combined_tuning as fixtures
from tools import final_combined_structural as final


CLASSIC_SHA = "6fede57915b07a6ea7675292833f8ad79269e665f09977bffd2a9562ce1aaf98"
OVERRIDES = {
    "combined18_24_subsample08": {"subsample": 0.8},
    "combined18_24_leafwise32_depth8": {
        "growth_policy": "leafwise", "max_leaves": 32, "max_depth": 8,
    },
}


class TestStructuralRecipes(unittest.TestCase):
    def test_only_authorized_structural_changes_are_accepted(self):
        final.validate_configuration(final.ROOT)
        base = final.read_json(final.ROOT / final.BASE_SUITE)
        self.assertEqual(set(final.VARIANTS), set(OVERRIDES))
        for variant in final.VARIANTS:
            with self.subTest(variant=variant):
                suite = final.read_json(final.ROOT / final.REFERENCES[variant])
                self.assertEqual(suite["models"][0]["parameters"],
                                 dict(base["models"][0]["parameters"], histogram_strategy="recursive",
                                      **OVERRIDES[variant]))
                self.assertEqual(suite["models"][0]["checkpoints"], [200])
                for key in ("data", "seed", "fold_count", "preprocessing_config_path",
                            "preprocessing_variant", "feature_metadata_path", "nested_threshold"):
                    self.assertEqual(suite[key], base[key])
                self.assertEqual(final.EXPECTED_FEATURES[variant], 297)
        with tempfile.TemporaryDirectory(dir=final.ROOT / ".local") as temporary:
            root = Path(temporary)
            for relative in (final.BASE_SUITE, final.PREPROCESSING_PATH,
                             *final.REFERENCES.values()):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(final.ROOT / relative, destination)
            final.validate_configuration(root)
            mutations = {
                "subsample": lambda value: value["models"][0]["parameters"].update(subsample=0.7),
                "mix_both_methods": lambda value: value["models"][0]["parameters"].update(
                    growth_policy="leafwise", max_leaves=32, max_depth=8),
                "feature_count": lambda value: value["models"][0]["parameters"].update(max_features=297),
                "native_categories": lambda value: value["models"][0]["parameters"].update(
                    categorical_feature_indices=[74, 77]),
                "bins": lambda value: value["models"][0]["parameters"].update(n_bins=128),
                "l2": lambda value: value["models"][0]["parameters"].update(l2_regularization=5),
                "trees": lambda value: value["models"][0].update(checkpoints=[300]),
                "folds": lambda value: value.update(seed=20261016),
                "preprocessing": lambda value: value.update(preprocessing_variant="count_ab"),
            }
            path = root / final.REFERENCES[final.VARIANTS[0]]
            original = final.read_json(path)
            for name, mutate in mutations.items():
                with self.subTest(change=name):
                    changed = copy.deepcopy(original)
                    mutate(changed)
                    final.write_json(path, changed)
                    with self.assertRaises(ValueError):
                        final.validate_configuration(root)
            final.write_json(path, original)
            path = root / final.REFERENCES[final.VARIANTS[1]]
            original = final.read_json(path)
            for key, value in (("max_leaves", 64), ("max_depth", 7), ("subsample", 0.8)):
                with self.subTest(leafwise_change=key):
                    changed = copy.deepcopy(original)
                    changed["models"][0]["parameters"][key] = value
                    final.write_json(path, changed)
                    with self.assertRaises(ValueError):
                        final.validate_configuration(root)
            final.write_json(path, original)
            final.validate_configuration(root)

    def test_invalid_worker_count_is_rejected_before_loading_data(self):
        for workers in (0, 3):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, "two|2"):
                final.run(Path("not-a-manifest.json"), workers=workers)


class TestStructuralFrozenFlow(unittest.TestCase):
    write_csv = staticmethod(fixtures.TestCombinedTuningSyntheticSubmissions.write_csv)

    def setUp(self):
        fixtures.TestCombinedTuningSyntheticSubmissions.setUp(self)
        shutil.copyfile(final.ROOT / "src/numpy_boosting_structural.py",
                        self.root / "src/numpy_boosting_structural.py")
        shutil.copyfile(final.ROOT / final.BASE_SUITE, self.root / final.BASE_SUITE)
        # Remove a perfect synthetic separator so the micro trees can branch.
        self.train[:, self.names.index("_AGE80")] = 30 + np.arange(36) % 15
        selected = np.array([15, 2, 25, 4, 18, 11, 24, 7, 29, 16, 35, 0])
        self.test = self.train[selected].copy()
        for split, values, ids in (("train", self.train, self.train_ids),
                                   ("test", self.test, self.test_ids)):
            np.save(self.root / f"data/processed/x_{split}_float32.npy", values)
            self.write_csv(self.root / f"data/raw/dataset/x_{split}.csv", ["Id", *self.names],
                           [(int(row_id), *row) for row_id, row in zip(ids, values)])
        for variant in final.VARIANTS:
            suite = final.read_json(final.ROOT / final.REFERENCES[variant])
            parameters = suite["models"][0]["parameters"]
            parameters.update(max_depth=2, min_samples_leaf=1)
            if variant.endswith("leafwise32_depth8"):
                parameters.update(max_depth=4, max_leaves=5)
            suite["models"][0]["checkpoints"] = [50]
            final.write_json(self.root / final.REFERENCES[variant], suite)
        for patch in (mock.patch.object(final, "EXPECTED_SHAPE", (36, 12, 321)),
                      mock.patch.object(final, "validate_configuration", return_value=[]),
                      mock.patch.object(final, "_base_reference", return_value=(None, {}))):
            patch.start()
            self.addCleanup(patch.stop)
        self.manifest = final.prepare(self.root, self.root / "structural_output")
        self.manifest_path = self.root / "structural_output/manifest.json"
        self.engine = final._engine(self.manifest)

    def test_two_slots_deliver_each_own_threshold_only_after_oof_and_full_and_cache(self):
        first, second = final.VARIANTS
        initial_pair = threading.Barrier(2)
        first_full_done = threading.Event()
        guard = threading.Lock()
        active = maximum_active = 0
        launches = []
        original_launch, original_deliver = final._launch, final._deliver_variant
        for value in self.manifest["environment"]["threads"].values():
            self.assertEqual(value, "1")

        def launch(manifest_path, variant, partition):
            nonlocal active, maximum_active
            with guard:
                active += 1
                maximum_active = max(maximum_active, active)
                launches.append((variant, partition))
            try:
                if partition == "fold_1":
                    initial_pair.wait(timeout=10)
                if variant == first and partition == "fold_3":
                    self.assertTrue(first_full_done.wait(timeout=20))
                    self.assertFalse((Path(self.manifest["delivery"]) /
                                      f"submission_{first}_full_train_oof.csv").exists())
                    self.assertFalse((Path(self.manifest["output"]) / first / "calibration.json").exists())
                original_launch(manifest_path, variant, partition)
                if variant == first and partition == "full":
                    first_full_done.set()
            finally:
                with guard:
                    active -= 1

        def deliver(manifest, variant, record):
            for partition in ("fold_1", "fold_2", "fold_3", "full"):
                self.assertIsNotNone(self.engine.completed(manifest, variant, partition))
            self.assertTrue((Path(manifest["output"]) / variant / "calibration.json").is_file())
            return original_deliver(manifest, variant, record)

        with mock.patch.object(final, "_launch", side_effect=launch), \
                mock.patch.object(final, "_deliver_variant", side_effect=deliver), \
                contextlib.redirect_stdout(io.StringIO()):
            comparison = final.run(self.manifest_path, workers=2)
        self.assertEqual(maximum_active, 2)
        self.assertEqual(set(launches), {(variant, partition) for variant in final.VARIANTS
                                       for partition in ("fold_1", "fold_2", "fold_3", "full")})
        self.assertEqual(len(launches), 8)
        self.assertTrue(first_full_done.is_set())
        loaded = self.engine.modules(self.manifest)

        def assert_delivery_backend():
            delivery = final.read_json(Path(self.manifest["delivery"]) / "manifest.json")
            self.assertNotIn("historical_booster_sha256", delivery)
            self.assertEqual(delivery["booster_sha256"], self.manifest["sources"]["booster"]["original_sha256"])
            self.assertEqual(delivery["baseline_classic_booster_sha256"], CLASSIC_SHA)
            self.assertEqual(delivery["booster_backend"], "structural")
            self.assertEqual(delivery["run_manifest_sha256"], final.checksum(self.manifest_path))
            self.assertTrue(delivery["complete"])
            self.assertFalse(delivery["uploaded_to_aicrowd"])
            return delivery

        delivery = assert_delivery_backend()
        saved_bytes = {}
        for variant in final.VARIANTS:
            with self.subTest(variant=variant):
                directory = Path(self.manifest["output"]) / variant
                summary = comparison["variants"][variant]
                with np.load(directory / "full_train_oof.npz") as oof:
                    np.testing.assert_array_equal(oof["labels"], self.labels)
                    np.testing.assert_array_equal(oof["train_ids"], self.train_ids)
                    np.testing.assert_array_equal(oof["row_indices"], np.arange(36))
                    np.testing.assert_array_equal(oof["fold_ids"], np.load(self.manifest["folds_path"]))
                    threshold = loaded["evaluation"].best_f1_threshold(self.labels, oof["probabilities"]).threshold
                    self.assertEqual(threshold, summary["decision_threshold"])
                    for fold in (1, 2, 3):
                        rows = np.flatnonzero(oof["fold_ids"] == fold)
                        np.testing.assert_array_equal(oof["probabilities"][rows],
                                                      np.load(directory / f"fold_{fold}/probabilities.npy"))
                full = self.engine.completed(self.manifest, variant, "full")
                self.assertEqual(full["signature"]["training_shape"], [36, 297])
                model, metadata = loaded["booster"].HistogramGradientBoostingClassifier.load_checkpoint(
                    directory / "full/model.pkl")
                self.assertEqual(metadata["parameters"], self.manifest["variants"][variant]["model_parameters"])
                for key, default in (("subsample", 1.0), ("growth_policy", "depthwise"), ("max_leaves", None)):
                    self.assertEqual(getattr(model, key), metadata["parameters"].get(key, default))
                self.assertEqual(model.training_scores_.shape, (36,))
                for tree in model.trees_:
                    self.assertEqual(tree.training_row_count_, 36)
                    self.assertEqual(tree.sample_row_count_, 28 if model.subsample < 1 else 36)
                    self.assertEqual(tree.sample_class_counts_, (19, 9) if model.subsample < 1 else (24, 12))
                    if model.growth_policy == "leafwise":
                        self.assertLessEqual(tree.leaf_count_, 5)
                sample_hashes = {tree.sample_row_indices_sha256_ for tree in model.trees_}
                if model.subsample < 1:
                    self.assertGreater(len(sample_hashes), 1, "All trees reused a fixed row subset")
                else:
                    self.assertEqual(len(sample_hashes), 1)
                record = delivery["variants"][variant]
                csv_path = Path(self.manifest["delivery"]) / record["csv"]
                ids, predictions = self.engine._read_pair_csv(csv_path, ["Id", "Prediction"])
                np.testing.assert_array_equal(ids, self.test_ids)
                np.testing.assert_array_equal(predictions,
                                              np.where(np.load(directory / "full/probabilities.npy") >= threshold, 1, -1))
                self.assertTrue(np.all(np.isin(predictions, [-1, 1])))
                self.assertEqual(record["rows"], 12)
                self.assertEqual(record["sha256"], final.checksum(csv_path))
                self.assertIsNone(summary["official_test_f1"])
                saved_bytes[csv_path] = csv_path.read_bytes()
        with mock.patch.object(final, "_launch", side_effect=AssertionError("Unexpected training")), \
                contextlib.redirect_stdout(io.StringIO()):
            final.run(self.manifest_path, workers=2)
        assert_delivery_backend()
        frozen_run = subprocess.run(
            [sys.executable, "-B", self.manifest["driver"], "run", "--manifest", str(self.manifest_path), "--workers", "2"],
            cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(frozen_run.returncode, 0, frozen_run.stderr)
        self.assertNotIn("START ", frozen_run.stdout)
        assert_delivery_backend()
        for path, before in saved_bytes.items():
            self.assertEqual(path.read_bytes(), before)

    def test_both_actual_frozen_workers_resume25_to50_exactly_like_fresh_models(self):
        loaded = self.engine.modules(self.manifest)
        model_class = loaded["booster"].HistogramGradientBoostingClassifier
        original_save = model_class.save_checkpoint

        def interrupted_save(model, path, metadata=None):
            original_save(model, path, metadata)
            raise RuntimeError("Synthetic structural interruption")

        for variant in final.VARIANTS:
            with self.subTest(variant=variant):
                with mock.patch.object(model_class, "save_checkpoint", new=interrupted_save), \
                        contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "Synthetic structural"):
                        final.worker(self.manifest_path, variant, "fold_1")
                final._launch(self.manifest_path, variant, "fold_1")
                completed = self.engine.completed(self.manifest, variant, "fold_1")
                self.assertEqual(completed["resumed_from_trees"], 25)
                self.assertEqual(completed["trees"], 50)
                train, labels, valid, _ = self.engine._matrices(
                    self.manifest, variant, "fold_1", loaded["preprocessing"])
                reference = model_class(**self.manifest["variants"][variant]["model_parameters"]).fit(train, labels)
                directory = Path(self.manifest["output"]) / variant / "fold_1"
                saved = np.load(directory / "probabilities.npy")
                np.testing.assert_array_equal(saved, reference.predict_proba(valid)[:, 1])
                resumed, _ = model_class.load_checkpoint(directory / "model.pkl")
                np.testing.assert_array_equal(resumed.training_scores_, reference.training_scores_)
                self.assertEqual(resumed.random_generator_state_, reference.random_generator_state_)
                self.assertEqual(resumed.sampling_generator_state_, reference.sampling_generator_state_)
                self.assertEqual([tree.sample_row_indices_sha256_ for tree in resumed.trees_],
                                 [tree.sample_row_indices_sha256_ for tree in reference.trees_])

    def test_sources_controls_matrices_and_parent_or_orphan_locks_are_enforced(self):
        self.engine.frozen(self.manifest, inputs=True)
        loaded = self.engine.modules(self.manifest)
        first, second = final.VARIANTS
        original = self.engine._matrices(self.manifest, first, "fold_1", loaded["preprocessing"])
        other = self.engine._matrices(self.manifest, second, "fold_1", loaded["preprocessing"])
        self.assertEqual(original[3], other[3])
        for index in (0, 1, 2):
            np.testing.assert_array_equal(original[index], other[index])
        referenced = dict(self.manifest, base_reference={
            "partitions": {"fold_1": {"signature": copy.deepcopy(original[3])}}})
        matched = self.engine._matrices(referenced, first, "fold_1", loaded["preprocessing"])
        np.testing.assert_array_equal(original[0], matched[0])
        referenced["base_reference"]["partitions"]["fold_1"]["signature"]["training_matrix_sha256"] = "different"
        original_manifest_bytes = self.manifest_path.read_bytes()
        original_digest_bytes = self.manifest_path.with_suffix(".sha256").read_bytes()
        try:
            final.write_json(self.manifest_path, referenced)
            self.manifest_path.with_suffix(".sha256").write_text(final.checksum(self.manifest_path) + "\n", encoding="ascii")
            with mock.patch.object(loaded["booster"].HistogramGradientBoostingClassifier, "fit",
                                   side_effect=AssertionError("Fit preceded signature validation")):
                with self.assertRaisesRegex(ValueError, "matrices|rows|base|differ"):
                    final.worker(self.manifest_path, first, "fold_1")
        finally:
            self.manifest_path.write_bytes(original_manifest_bytes)
            self.manifest_path.with_suffix(".sha256").write_bytes(original_digest_bytes)
        sources = self.manifest["sources"]
        self.assertEqual(set(sources), {"booster", "preprocessing", "baseline_preprocessing",
                                       "conditional_imputation", "evaluation"})
        self.assertEqual(Path(sources["booster"]["original_path"]).name, "numpy_boosting_structural.py")
        self.assertEqual(sources["booster"]["original_sha256"], final.checksum(final.ROOT / "src/numpy_boosting_structural.py"))
        self.assertNotEqual(sources["booster"]["original_sha256"], CLASSIC_SHA)
        self.assertEqual(final.checksum(final.ROOT / "src/numpy_boosting_phase14.py"), CLASSIC_SHA)
        for key, source in sources.items():
            if key != "booster":
                self.assertEqual(source["original_sha256"],
                                 final.checksum(final.ROOT / "src" / Path(source["original_path"]).name))
        for path in (Path(sources["booster"]["path"]), Path(self.manifest["structural_wrapper"]["path"]),
                     Path(self.manifest["structural_scheduler"]["path"]),
                     self.root / final.REFERENCES[first], Path(self.manifest["data"]["train_labels"]),
                     Path(self.manifest["folds_path"])):
            with self.subTest(path=path):
                before = path.read_bytes()
                try:
                    path.write_bytes(before + b"changed")
                    with self.assertRaisesRegex(ValueError, "[Ff]rozen|changed"):
                        self.engine.frozen(self.manifest, inputs=True)
                finally:
                    path.write_bytes(before)
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "2"}):
            with self.assertRaisesRegex(ValueError, "environment|Python|NumPy|thread|changed"):
                self.engine.frozen(self.manifest, inputs=True)
        with mock.patch.object(final, "_launch", side_effect=AssertionError("Unexpected launch")):
            with self.engine.file_lock(Path(self.manifest["output"]) / "run.lock"):
                with self.assertRaises(OSError):
                    final.run(self.manifest_path, workers=2)
            worker_lock = self.engine.task_directory(self.manifest, second, "full") / "worker.lock"
            with self.engine.file_lock(worker_lock):
                with self.assertRaises(OSError):
                    final.run(self.manifest_path, workers=2)
        changed = copy.deepcopy(self.manifest)
        changed["variants"][first]["model_parameters"]["subsample"] = 1.0
        final.write_json(self.manifest_path, changed)
        with self.assertRaisesRegex(ValueError, "manifest.*changed"):
            final.read_manifest(self.manifest_path)


if __name__ == "__main__":
    unittest.main()

"""Verify isolated tree continuation and all-feature growth submissions."""

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
from tools import final_combined_growth as final
from tools import final_combined_tuning as origin_final


class TestCombinedGrowthRecipes(unittest.TestCase):
    def test_each_authorized_recipe_changes_only_trees_or_feature_candidates(self):
        final.validate_configuration(final.ROOT)
        base = final.read_json(final.ROOT / final.BASE_SUITE)
        for variant in final.VARIANTS:
            with self.subTest(variant=variant):
                suite = final.read_json(final.ROOT / final.REFERENCES[variant])
                expected_parameters = copy.deepcopy(base["models"][0]["parameters"])
                expected_trees = 200
                if variant == final.EXTENSION_VARIANT:
                    expected_trees = 300
                else:
                    expected_parameters["max_features"] = 297
                self.assertEqual(suite["models"][0]["parameters"], expected_parameters)
                self.assertEqual(suite["models"][0]["checkpoints"], [expected_trees])
                self.assertEqual(suite["data"], base["data"])
                self.assertEqual(suite["fold_count"], base["fold_count"])
                self.assertEqual(suite["seed"], base["seed"])
                self.assertEqual(suite["preprocessing_config_path"], base["preprocessing_config_path"])
                self.assertEqual(suite["preprocessing_variant"], base["preprocessing_variant"])
                self.assertEqual(final.EXPECTED_FEATURES[variant], 297)

    def test_extra_parameter_or_protocol_changes_are_rejected(self):
        (final.ROOT / ".local").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=final.ROOT / ".local") as directory:
            root = Path(directory)
            for relative in (final.BASE_SUITE, final.PREPROCESSING_PATH,
                             *final.REFERENCES.values()):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(final.ROOT / relative, destination)
            final.validate_configuration(root)
            path = root / final.REFERENCES[final.EXTENSION_VARIANT]
            original = final.read_json(path)
            mutations = {
                "both_axes": lambda value: value["models"][0]["parameters"].update(max_features=297),
                "depth": lambda value: value["models"][0]["parameters"].update(max_depth=6),
                "learning_rate": lambda value: value["models"][0]["parameters"].update(learning_rate=0.025),
                "regularization": lambda value: value["models"][0]["parameters"].update(l2_regularization=5),
                "fold_seed": lambda value: value.update(seed=20261014),
                "fold_count": lambda value: value.update(fold_count=4),
                "data": lambda value: value["data"].update(features_path="different.npy"),
                "preprocessing": lambda value: value.update(preprocessing_variant="count_ab"),
            }
            for name, mutation in mutations.items():
                with self.subTest(change=name):
                    changed = copy.deepcopy(original)
                    mutation(changed)
                    final.write_json(path, changed)
                    with self.assertRaises(ValueError):
                        final.validate_configuration(root)
            final.write_json(path, original)
            final.validate_configuration(root)

    def test_three_workers_are_rejected_without_loading_a_manifest(self):
        with self.assertRaisesRegex(ValueError, "two|2"):
            final.run(Path("not-a-manifest.json"), workers=3)


class TestCombinedGrowthSyntheticSubmissions(unittest.TestCase):
    write_csv = staticmethod(fixtures.TestCombinedTuningSyntheticSubmissions.write_csv)

    def setUp(self):
        fixtures.TestCombinedTuningSyntheticSubmissions.setUp(self)
        self.origin_manifest = self.manifest
        self.origin_manifest_path = self.manifest_path
        with mock.patch.object(origin_final, "_launch", side_effect=origin_final.worker), \
                contextlib.redirect_stdout(io.StringIO()):
            origin_final._pipeline(self.origin_manifest_path, self.origin_manifest,
                                   final.ORIGIN_VARIANT)
        for variant in final.VARIANTS:
            suite = final.read_json(final.ROOT / final.REFERENCES[variant])
            suite["models"][0]["parameters"].update(max_depth=2, min_samples_leaf=1)
            suite["models"][0]["checkpoints"] = [100 if variant == final.EXTENSION_VARIANT else 50]
            final.write_json(self.root / final.REFERENCES[variant], suite)
        for patch in (mock.patch.object(final, "EXPECTED_SHAPE", (36, 12, 321)),
                      mock.patch.object(final, "validate_configuration", return_value=[])):
            patch.start()
            self.addCleanup(patch.stop)
        self.manifest = final.prepare(self.root, self.root / "growth_output",
                                      origin_manifest_path=self.origin_manifest_path)
        self.manifest_path = self.root / "growth_output/manifest.json"
        self.engine = final._engine(self.manifest)
        self.origin_engine = origin_final._engine(self.origin_manifest)

    def origin_paths(self):
        directory = Path(self.origin_manifest["output"]) / final.ORIGIN_VARIANT
        return [directory / f"{partition}/model.pkl"
                for partition in ("fold_1", "fold_2", "fold_3", "full")]

    def test_original_prefix_partial_resume_and_fresh_target_are_exactly_equal(self):
        variant = final.EXTENSION_VARIANT
        origin_bytes = {path: path.read_bytes() for path in self.origin_paths()}
        loaded = self.engine.modules(self.manifest)
        model_class = loaded["booster"].HistogramGradientBoostingClassifier
        original_save = model_class.save_checkpoint

        def interrupted_save(model, path, metadata=None):
            original_save(model, path, metadata)
            if len(model.trees_) == 75:
                raise RuntimeError("Synthetic interruption after continued checkpoint")

        with mock.patch.object(model_class, "save_checkpoint", new=interrupted_save), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Synthetic interruption"):
                final.worker(self.manifest_path, variant, "fold_1")
        checkpoint = self.engine.task_directory(self.manifest, variant, "fold_1") / "model.pkl"
        partial, partial_metadata = model_class.load_checkpoint(checkpoint)
        self.assertEqual(len(partial.trees_), 75)
        self.assertIn("derivation", partial_metadata)
        self.assertEqual(partial.n_estimators, 100)
        final._launch(self.manifest_path, variant, "fold_1")
        completed = self.engine.completed(self.manifest, variant, "fold_1")
        self.assertEqual(completed["trees"], 100)
        self.assertEqual(completed["resumed_from_trees"], 75)
        saved, saved_metadata = model_class.load_checkpoint(checkpoint)
        train, labels, valid, signature = self.engine._matrices(
            self.manifest, variant, "fold_1", loaded["preprocessing"])
        self.assertEqual(signature, completed["signature"])
        reference = model_class(**self.manifest["variants"][variant]["model_parameters"]).fit(train, labels)
        np.testing.assert_array_equal(saved.predict_proba(valid), reference.predict_proba(valid))
        np.testing.assert_array_equal(saved.training_scores_, reference.training_scores_)
        self.assertEqual(saved.random_generator_state_, reference.random_generator_state_)
        np.testing.assert_array_equal(saved.binner_.bin_counts_, reference.binner_.bin_counts_)
        np.testing.assert_array_equal(saved.binner_.exact_columns_, reference.binner_.exact_columns_)
        for continued, fresh in zip(saved.binner_.edges_, reference.binner_.edges_):
            np.testing.assert_array_equal(continued, fresh)
        original_model, original_metadata = model_class.load_checkpoint(self.origin_paths()[0])
        prefix = dict(saved.staged_predict_proba(valid, [50, 100]))
        np.testing.assert_array_equal(prefix[50], original_model.predict_proba(valid)[:, 1])
        self.assertEqual(len(original_model.trees_), 50)
        self.assertEqual(original_model.n_estimators, 50)
        self.assertEqual(original_metadata["signature"], signature)
        for path, before in origin_bytes.items():
            self.assertEqual(path.read_bytes(), before)

    def test_provenance_changes_and_wrong_runtime_signature_block_before_continuation(self):
        variant = final.EXTENSION_VARIANT
        loaded = self.engine.modules(self.manifest)
        model_class = loaded["booster"].HistogramGradientBoostingClassifier
        copied_origin = Path(self.manifest["extension_origin"]["checkpoints"]["fold_1"]["copy_path"])
        paths = {
            "copied_origin_sha": copied_origin,
            "original_origin_sha": self.origin_paths()[0],
            "source_sha": self.root / "src/numpy_boosting_phase14.py",
            "labels_sha": Path(self.manifest["data"]["train_labels"]),
            "input_matrix_sha": Path(self.manifest["data"]["train_features"]),
            "fold_sha": Path(self.manifest["folds_path"]),
        }
        with mock.patch.object(model_class, "continue_fit", side_effect=AssertionError("Invalid continuation")):
            for name, path in paths.items():
                with self.subTest(corruption=name):
                    original = path.read_bytes()
                    try:
                        path.write_bytes(original + b"changed")
                        with self.assertRaises(ValueError):
                            final.worker(self.manifest_path, variant, "fold_1")
                    finally:
                        path.write_bytes(original)
            with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "2"}):
                with self.assertRaisesRegex(ValueError, "environment|Python|NumPy|thread|changed"):
                    final.worker(self.manifest_path, variant, "fold_1")
            original_matrices = self.engine._matrices

            def wrong_signature(*args, **kwargs):
                train, labels, valid, signature = original_matrices(*args, **kwargs)
                signature = dict(signature, training_matrix_sha256="different ordered training matrix")
                return train, labels, valid, signature

            with mock.patch.object(self.engine, "_matrices", side_effect=wrong_signature):
                with self.assertRaisesRegex(ValueError, "signature|matrix|matrices|provenance|changed|differs"):
                    final.worker(self.manifest_path, variant, "fold_1")

    def test_two_pipelines_deliver_only_after_oof_and_full_with_own_thresholds(self):
        first, second = final.VARIANTS
        barrier = threading.Barrier(2)
        guard = threading.Lock()
        active = maximum_active = 0
        launches = []
        origin_bytes = {path: path.read_bytes() for path in self.origin_paths()}
        original_deliver = final._deliver_variant

        def launch(manifest_path, variant, partition):
            nonlocal active, maximum_active
            with guard:
                active += 1
                maximum_active = max(maximum_active, active)
                launches.append((variant, partition))
            try:
                if partition == "fold_1":
                    barrier.wait(timeout=10)
                final.worker(manifest_path, variant, partition)
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
            result = final.run(self.manifest_path, workers=2)
        self.assertEqual(maximum_active, 2)
        self.assertEqual(len(launches), 8)
        self.assertEqual(set(result["variants"]), {first, second})
        delivery = final.read_json(Path(self.manifest["delivery"]) / "manifest.json")
        self.assertTrue(delivery["complete"])
        self.assertFalse(delivery["uploaded_to_aicrowd"])
        evaluation = self.engine.modules(self.manifest)["evaluation"]
        saved_csv_bytes = {}
        for variant in final.VARIANTS:
            with self.subTest(variant=variant):
                directory = Path(self.manifest["output"]) / variant
                summary = result["variants"][variant]
                with np.load(directory / "full_train_oof.npz") as oof:
                    np.testing.assert_array_equal(oof["labels"], self.labels)
                    np.testing.assert_array_equal(oof["train_ids"], self.train_ids)
                    np.testing.assert_array_equal(oof["row_indices"], np.arange(36))
                    np.testing.assert_array_equal(oof["fold_ids"], np.load(self.manifest["folds_path"]))
                    threshold = evaluation.best_f1_threshold(self.labels, oof["probabilities"]).threshold
                    self.assertEqual(threshold, summary["decision_threshold"])
                    for fold in (1, 2, 3):
                        rows = np.flatnonzero(oof["fold_ids"] == fold)
                        np.testing.assert_array_equal(oof["probabilities"][rows],
                                                      np.load(directory / f"fold_{fold}/probabilities.npy"))
                full = self.engine.completed(self.manifest, variant, "full")
                self.assertEqual(full["signature"]["training_shape"], [36, 297])
                record = delivery["variants"][variant]
                csv_path = Path(self.manifest["delivery"]) / record["csv"]
                ids, predictions = self.engine._read_pair_csv(csv_path, ["Id", "Prediction"])
                probabilities = np.load(directory / "full/probabilities.npy")
                np.testing.assert_array_equal(ids, self.test_ids)
                np.testing.assert_array_equal(predictions, np.where(probabilities >= threshold, 1, -1))
                self.assertTrue(np.all(np.isin(predictions, [-1, 1])))
                self.assertEqual(final.checksum(csv_path), record["sha256"])
                self.assertEqual(record["rows"], 12)
                self.assertIsNone(summary["official_test_f1"])
                saved_csv_bytes[csv_path] = csv_path.read_bytes()
        with mock.patch.object(final, "_launch", side_effect=AssertionError("Unexpected retraining")), \
                contextlib.redirect_stdout(io.StringIO()):
            cached = final.run(self.manifest_path, workers=2)
        for path, before in saved_csv_bytes.items():
            self.assertEqual(path.read_bytes(), before)
        for variant in final.VARIANTS:
            self.assertEqual(result["variants"][variant]["submission"]["sha256"],
                             cached["variants"][variant]["submission"]["sha256"])
        for path, before in origin_bytes.items():
            self.assertEqual(path.read_bytes(), before)
        frozen_run = subprocess.run(
            [sys.executable, "-B", self.manifest["driver"], "run",
             "--manifest", str(self.manifest_path), "--workers", "2"],
            cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(frozen_run.returncode, 0, frozen_run.stderr)
        self.assertNotIn("START ", frozen_run.stdout)

    def test_live_parent_or_worker_lock_prevents_duplicate_fitting(self):
        variant = final.EXTENSION_VARIANT
        with mock.patch.object(final, "_launch", side_effect=AssertionError("Unexpected launch")):
            with self.engine.file_lock(Path(self.manifest["output"]) / "run.lock"):
                with self.assertRaises(OSError):
                    final.run(self.manifest_path, workers=2)
        worker_lock = self.engine.task_directory(self.manifest, variant, "fold_1") / "worker.lock"
        with self.engine.file_lock(worker_lock):
            with self.assertRaises(OSError):
                final.worker(self.manifest_path, variant, "fold_1")


if __name__ == "__main__":
    unittest.main()

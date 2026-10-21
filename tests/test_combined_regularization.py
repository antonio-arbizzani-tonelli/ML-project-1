"""Check isolated regularization recipes and their frozen submission flow."""

import contextlib
import copy
import io
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
from tools import final_combined_regularization as final


OVERRIDES = {
    "combined18_24_leaf400": {"min_samples_leaf": 400},
    "combined18_24_depth4": {"max_depth": 4},
}


class TestCombinedRegularizationRecipes(unittest.TestCase):
    def test_recipes_change_one_axis_and_extra_changes_are_rejected(self):
        final.validate_configuration(final.ROOT)
        base = final.read_json(final.ROOT / final.BASE_SUITE)
        self.assertEqual(set(final.VARIANTS), set(OVERRIDES))
        for variant in final.VARIANTS:
            with self.subTest(variant=variant):
                suite = final.read_json(final.ROOT / final.REFERENCES[variant])
                expected = dict(base["models"][0]["parameters"], **OVERRIDES[variant])
                self.assertEqual(suite["models"][0]["parameters"], expected)
                self.assertEqual(suite["models"][0]["checkpoints"], [200])
                for key in ("data", "seed", "fold_count", "preprocessing_config_path",
                            "preprocessing_variant", "feature_metadata_path", "nested_threshold"):
                    self.assertEqual(suite[key], base[key])
                self.assertEqual(final.EXPECTED_FEATURES[variant], 297)
        (final.ROOT / ".local").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=final.ROOT / ".local") as temporary:
            root = Path(temporary)
            for relative in (final.BASE_SUITE, final.PREPROCESSING_PATH,
                             *final.REFERENCES.values()):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(final.ROOT / relative, destination)
            final.validate_configuration(root)
            path = root / final.REFERENCES["combined18_24_leaf400"]
            original = final.read_json(path)
            changes = {
                "both_axes": lambda value: value["models"][0]["parameters"].update(max_depth=4),
                "features": lambda value: value["models"][0]["parameters"].update(max_features=297),
                "trees": lambda value: value["models"][0].update(checkpoints=[300]),
                "bins": lambda value: value["models"][0]["parameters"].update(n_bins=128),
                "l2": lambda value: value["models"][0]["parameters"].update(l2_regularization=5),
                "folds": lambda value: value.update(seed=20261015),
                "data": lambda value: value["data"].update(features_path="another.npy"),
                "preprocessing": lambda value: value.update(preprocessing_variant="count_ab"),
            }
            for name, change in changes.items():
                with self.subTest(change=name):
                    changed = copy.deepcopy(original)
                    change(changed)
                    final.write_json(path, changed)
                    with self.assertRaises(ValueError):
                        final.validate_configuration(root)
            final.write_json(path, original)
            final.validate_configuration(root)

    def test_worker_limit_is_checked_before_loading_data(self):
        for workers in (0, 3):
            with self.subTest(workers=workers):
                with self.assertRaisesRegex(ValueError, "two|2"):
                    final.run(Path("not-a-manifest.json"), workers=workers)


class TestCombinedRegularizationSyntheticFlow(unittest.TestCase):
    write_csv = staticmethod(fixtures.TestCombinedTuningSyntheticSubmissions.write_csv)

    def setUp(self):
        # Reuse only temporary data/source creation; the origin is not fitted.
        fixtures.TestCombinedTuningSyntheticSubmissions.setUp(self)
        for variant in final.VARIANTS:
            suite = final.read_json(final.ROOT / final.REFERENCES[variant])
            parameters = suite["models"][0]["parameters"]
            parameters.update(max_depth=2, min_samples_leaf=1)
            parameters.update({"min_samples_leaf": 2} if variant.endswith("leaf400")
                              else {"max_depth": 1})
            suite["models"][0]["checkpoints"] = [50]
            final.write_json(self.root / final.REFERENCES[variant], suite)
        for patch in (mock.patch.object(final, "EXPECTED_SHAPE", (36, 12, 321)),
                      mock.patch.object(final, "validate_configuration", return_value=[]),
                      mock.patch.object(final, "_base_reference", return_value=(None, {}))):
            patch.start()
            self.addCleanup(patch.stop)
        self.manifest = final.prepare(self.root, self.root / "regularization_output")
        self.manifest_path = self.root / "regularization_output/manifest.json"
        self.engine = final._engine(self.manifest)

    def test_two_frozen_workers_withhold_csv_until_oof_and_full_then_reuse_results(self):
        first, second = final.VARIANTS
        initial_pair = threading.Barrier(2)
        first_full_done = threading.Event()
        guard = threading.Lock()
        active = maximum_active = 0
        launches = []
        original_launch = final._launch
        original_deliver = final._deliver_variant
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
                    self.assertTrue(first_full_done.wait(timeout=20), "Full fit was not scheduled first")
                    csv_path = Path(self.manifest["delivery"]) / f"submission_{first}_full_train_oof.csv"
                    self.assertFalse(csv_path.exists(), "CSV appeared before the last OOF fold")
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
        self.assertEqual(len(launches), 8)
        self.assertTrue(first_full_done.is_set())
        delivery = final.read_json(Path(self.manifest["delivery"]) / "manifest.json")
        self.assertTrue(delivery["complete"])
        self.assertFalse(delivery["uploaded_to_aicrowd"])
        self.assertEqual(delivery["run_manifest_sha256"], final.checksum(self.manifest_path))
        evaluation = self.engine.modules(self.manifest)["evaluation"]
        saved_csv_bytes = {}
        for variant in final.VARIANTS:
            with self.subTest(variant=variant):
                directory = Path(self.manifest["output"]) / variant
                summary = comparison["variants"][variant]
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
        with mock.patch.object(final, "_launch", side_effect=AssertionError("Unexpected training")), \
                contextlib.redirect_stdout(io.StringIO()):
            repeated = final.run(self.manifest_path, workers=2)
        for path, before in saved_csv_bytes.items():
            self.assertEqual(path.read_bytes(), before)
        for variant in final.VARIANTS:
            self.assertEqual(comparison["variants"][variant]["submission"]["sha256"],
                             repeated["variants"][variant]["submission"]["sha256"])
        frozen_run = subprocess.run(
            [sys.executable, "-B", self.manifest["driver"], "run",
             "--manifest", str(self.manifest_path), "--workers", "2"],
            cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(frozen_run.returncode, 0, frozen_run.stderr)
        self.assertNotIn("START ", frozen_run.stdout)

    def test_frozen_controls_manifest_and_parent_or_orphan_locks_are_enforced(self):
        self.engine.frozen(self.manifest, inputs=True)
        loaded = self.engine.modules(self.manifest)
        variant = final.VARIANTS[0]
        matrices = self.engine._matrices(self.manifest, variant, "fold_1", loaded["preprocessing"])
        referenced = dict(self.manifest, base_reference={
            "partitions": {"fold_1": {"signature": copy.deepcopy(matrices[3])}}})
        matched = self.engine._matrices(referenced, variant, "fold_1", loaded["preprocessing"])
        np.testing.assert_array_equal(matrices[0], matched[0])
        referenced["base_reference"]["partitions"]["fold_1"]["signature"]["training_matrix_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "matrices|rows|base|differ"):
            self.engine._matrices(referenced, variant, "fold_1", loaded["preprocessing"])
        for path in (Path(self.manifest["regularization_wrapper"]["path"]),
                     Path(self.manifest["regularization_scheduler"]["path"]),
                     self.root / final.REFERENCES[final.VARIANTS[0]],
                     self.root / "src/feature_preprocessing.py"):
            with self.subTest(path=path):
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"changed")
                    with self.assertRaisesRegex(ValueError, "[Ff]rozen|changed"):
                        self.engine.frozen(self.manifest, inputs=True)
                finally:
                    path.write_bytes(original)
        with mock.patch.object(final, "_launch", side_effect=AssertionError("Unexpected launch")):
            with self.engine.file_lock(Path(self.manifest["output"]) / "run.lock"):
                with self.assertRaises(OSError):
                    final.run(self.manifest_path, workers=2)
            worker_lock = self.engine.task_directory(self.manifest, final.VARIANTS[0], "fold_3") / "worker.lock"
            with self.engine.file_lock(worker_lock):
                with self.assertRaises(OSError):
                    final.run(self.manifest_path, workers=2)
        changed = copy.deepcopy(self.manifest)
        changed["variants"][final.VARIANTS[0]]["model_parameters"]["max_depth"] = 6
        final.write_json(self.manifest_path, changed)
        with self.assertRaisesRegex(ValueError, "manifest.*changed"):
            final.read_manifest(self.manifest_path)


if __name__ == "__main__":
    unittest.main()

"""Verify frozen native activity-category submissions on transformed columns."""

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
from tools import final_combined_categorical as final


CLASSIC_SHA = "6fede57915b07a6ea7675292833f8ad79269e665f09977bffd2a9562ce1aaf98"
FEATURES = ("EXRACT11", "EXRACT21")


class TestNativeCategoricalRecipe(unittest.TestCase):
    def test_only_two_named_categories_are_authorized_and_indices_are_transformed(self):
        final.validate_configuration(final.ROOT)
        self.assertEqual(final.CATEGORICAL_FEATURES, FEATURES)
        self.assertEqual(len(final.VARIANTS), 1)
        base = final.read_json(final.ROOT / final.BASE_SUITE)
        suite = final.read_json(final.ROOT / final.REFERENCES[final.VARIANTS[0]])
        self.assertEqual(suite["categorical_features"], list(FEATURES))
        expected = dict(base["models"][0]["parameters"], histogram_strategy="recursive")
        self.assertEqual(suite["models"][0]["parameters"], expected)
        self.assertEqual(suite["models"][0]["checkpoints"], [200])
        self.assertNotIn("categorical_feature_indices", suite["models"][0]["parameters"])
        for key in ("data", "seed", "fold_count", "preprocessing_config_path",
                    "preprocessing_variant", "feature_metadata_path", "nested_threshold"):
            self.assertEqual(suite[key], base[key])
        arbitrary = [f"feature_{index}" for index in range(297)]
        arbitrary[9], arbitrary[93] = FEATURES
        self.assertEqual(final.resolve_categorical_indices(arbitrary), [9, 93])
        with self.assertRaises(ValueError):
            final.resolve_categorical_indices([name for name in arbitrary if name != "EXRACT21"])
        (final.ROOT / ".local").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=final.ROOT / ".local") as temporary:
            root = Path(temporary)
            for relative in (final.BASE_SUITE, final.PREPROCESSING_PATH,
                             *final.REFERENCES.values()):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(final.ROOT / relative, destination)
            final.validate_configuration(root)
            path = root / final.REFERENCES[final.VARIANTS[0]]
            original = final.read_json(path)
            changes = {
                "third_category": lambda value: value.update(categorical_features=[*FEATURES, "GENHLTH"]),
                "missing_category": lambda value: value.update(categorical_features=[FEATURES[0]]),
                "raw_indices": lambda value: value["models"][0]["parameters"].update(categorical_feature_indices=[90, 93]),
                "depth": lambda value: value["models"][0]["parameters"].update(max_depth=4),
                "bins": lambda value: value["models"][0]["parameters"].update(n_bins=128),
                "feature_count": lambda value: value["models"][0]["parameters"].update(max_features=297),
                "trees": lambda value: value["models"][0].update(checkpoints=[300]),
                "folds": lambda value: value.update(seed=20261016),
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

    def test_more_than_two_workers_is_rejected_before_loading_data(self):
        for workers in (0, 3):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, "two|2"):
                final.run(Path("not-a-manifest.json"), workers=workers)


class TestNativeCategoricalFrozenFlow(unittest.TestCase):
    write_csv = staticmethod(fixtures.TestCombinedTuningSyntheticSubmissions.write_csv)

    def setUp(self):
        fixtures.TestCombinedTuningSyntheticSubmissions.setUp(self)
        backend = self.root / "src/numpy_boosting_categorical.py"
        shutil.copyfile(final.ROOT / "src/numpy_boosting_categorical.py", backend)
        self.train[:, self.names.index("_AGE80")] = 50
        self.train[:, self.names.index("EXRACT11")] = np.tile([1, 1, 2, 2, 3, 3], 6)
        self.train[:, self.names.index("EXRACT21")] = np.tile([2, 3, 1, 2, 3, 4], 6)
        self.train[1, self.names.index("EXRACT21")] = np.nan
        self.train[7, self.names.index("EXRACT11")] = np.nan
        selected = np.array([15, 2, 25, 4, 18, 11, 24, 7, 29, 16, 35, 0])
        self.test = self.train[selected].copy()
        for split, values, ids in (("train", self.train, self.train_ids),
                                   ("test", self.test, self.test_ids)):
            np.save(self.root / f"data/processed/x_{split}_float32.npy", values)
            self.write_csv(self.root / f"data/raw/dataset/x_{split}.csv", ["Id", *self.names],
                           [(int(row_id), *row) for row_id, row in zip(ids, values)])
        variant = final.VARIANTS[0]
        suite = final.read_json(final.ROOT / final.REFERENCES[variant])
        suite["models"][0]["parameters"].update(max_depth=2, min_samples_leaf=1)
        suite["models"][0]["checkpoints"] = [50]
        final.write_json(self.root / final.REFERENCES[variant], suite)
        for patch in (mock.patch.object(final, "EXPECTED_SHAPE", (36, 12, 321)),
                      mock.patch.object(final, "validate_configuration", return_value=[]),
                      mock.patch.object(final, "_base_reference", return_value=(None, {}))):
            patch.start()
            self.addCleanup(patch.stop)
        self.manifest = final.prepare(self.root, self.root / "categorical_output")
        self.manifest_path = self.root / "categorical_output/manifest.json"
        self.engine = final._engine(self.manifest)

    def test_two_frozen_slots_withhold_csv_until_all_oof_and_full_then_cache_exactly(self):
        variant = final.VARIANTS[0]
        initial_pair = threading.Barrier(2)
        full_done = threading.Event()
        guard = threading.Lock()
        active = maximum_active = 0
        launches = []
        original_launch = final._launch
        original_deliver = final._deliver_variant
        for value in self.manifest["environment"]["threads"].values():
            self.assertEqual(value, "1")

        def launch(manifest_path, selected, partition):
            nonlocal active, maximum_active
            self.assertEqual(selected, variant)
            with guard:
                active += 1
                maximum_active = max(maximum_active, active)
                launches.append(partition)
            try:
                if partition in ("fold_1", "full"):
                    initial_pair.wait(timeout=10)
                if partition == "fold_3":
                    self.assertTrue(full_done.wait(timeout=20))
                    self.assertFalse((Path(self.manifest["delivery"]) /
                                      f"submission_{variant}_full_train_oof.csv").exists())
                    self.assertFalse((Path(self.manifest["output"]) / variant / "calibration.json").exists())
                original_launch(manifest_path, selected, partition)
                if partition == "full":
                    full_done.set()
            finally:
                with guard:
                    active -= 1

        def deliver(manifest, selected, record):
            for partition in ("fold_1", "fold_2", "fold_3", "full"):
                self.assertIsNotNone(self.engine.completed(manifest, selected, partition))
            return original_deliver(manifest, selected, record)

        with mock.patch.object(final, "_launch", side_effect=launch), \
                mock.patch.object(final, "_deliver_variant", side_effect=deliver), \
                contextlib.redirect_stdout(io.StringIO()):
            comparison = final.run(self.manifest_path, workers=2)
        self.assertEqual(maximum_active, 2)
        self.assertEqual(sorted(launches), ["fold_1", "fold_2", "fold_3", "full"])
        self.assertTrue(full_done.is_set())
        directory = Path(self.manifest["output"]) / variant
        loaded = self.engine.modules(self.manifest)
        full = self.engine.completed(self.manifest, variant, "full")
        names = full["signature"]["output_feature_names"]
        indices = [names.index(name) for name in FEATURES]
        self.assertEqual(indices, final.resolve_categorical_indices(names))
        self.assertNotEqual(indices, [self.names.index(name) for name in FEATURES])
        self.assertEqual(self.manifest["variants"][variant]["model_parameters"]["categorical_feature_indices"], indices)
        self.assertEqual(full["signature"]["training_shape"], [36, 297])
        model, metadata = loaded["booster"].HistogramGradientBoostingClassifier.load_checkpoint(directory / "full/model.pkl")
        self.assertEqual(model.categorical_feature_indices, indices)
        self.assertEqual(metadata["parameters"]["categorical_feature_indices"], indices)
        self.assertEqual(np.flatnonzero(model.binner_.categorical_columns_).tolist(), sorted(indices))

        def uses_categories(node):
            if node.feature is None:
                return False
            return (node.left_categories is not None
                    or uses_categories(node.left) or uses_categories(node.right))

        self.assertTrue(any(uses_categories(tree.root_) for tree in model.trees_))
        with np.load(directory / "full_train_oof.npz") as oof:
            np.testing.assert_array_equal(oof["labels"], self.labels)
            np.testing.assert_array_equal(oof["train_ids"], self.train_ids)
            np.testing.assert_array_equal(oof["row_indices"], np.arange(36))
            np.testing.assert_array_equal(oof["fold_ids"], np.load(self.manifest["folds_path"]))
            threshold = loaded["evaluation"].best_f1_threshold(self.labels, oof["probabilities"]).threshold
            self.assertEqual(threshold, comparison["variants"][variant]["decision_threshold"])
            for fold in (1, 2, 3):
                rows = np.flatnonzero(oof["fold_ids"] == fold)
                np.testing.assert_array_equal(oof["probabilities"][rows],
                                              np.load(directory / f"fold_{fold}/probabilities.npy"))
        def assert_delivery_backend():
            current = final.read_json(Path(self.manifest["delivery"]) / "manifest.json")
            self.assertNotIn("historical_booster_sha256", current)
            self.assertEqual(current["booster_sha256"], self.manifest["sources"]["booster"]["original_sha256"])
            self.assertEqual(current["baseline_classic_booster_sha256"], CLASSIC_SHA)
            self.assertEqual(current["booster_backend"], "native_categorical")
            self.assertEqual(current["categorical_columns"]["indices"], indices)
            self.assertEqual(current["categorical_columns"]["mapping"], dict(zip(FEATURES, indices)))
            return current

        delivery = assert_delivery_backend()
        self.assertTrue(delivery["complete"])
        self.assertFalse(delivery["uploaded_to_aicrowd"])
        self.assertNotIn("historical_booster_sha256", delivery)
        self.assertEqual(delivery["baseline_classic_booster_sha256"], CLASSIC_SHA)
        record = delivery["variants"][variant]
        csv_path = Path(self.manifest["delivery"]) / record["csv"]
        ids, predictions = self.engine._read_pair_csv(csv_path, ["Id", "Prediction"])
        np.testing.assert_array_equal(ids, self.test_ids)
        np.testing.assert_array_equal(predictions,
                                      np.where(np.load(directory / "full/probabilities.npy") >= threshold, 1, -1))
        self.assertTrue(np.all(np.isin(predictions, [-1, 1])))
        self.assertEqual(record["rows"], 12)
        self.assertEqual(record["sha256"], final.checksum(csv_path))
        before = csv_path.read_bytes()
        with mock.patch.object(final, "_launch", side_effect=AssertionError("Unexpected training")), \
                contextlib.redirect_stdout(io.StringIO()):
            final.run(self.manifest_path, workers=2)
        self.assertEqual(csv_path.read_bytes(), before)
        assert_delivery_backend()
        frozen_run = subprocess.run(
            [sys.executable, "-B", self.manifest["driver"], "run",
             "--manifest", str(self.manifest_path), "--workers", "2"],
            cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(frozen_run.returncode, 0, frozen_run.stderr)
        self.assertNotIn("START ", frozen_run.stdout)
        self.assertEqual(csv_path.read_bytes(), before)
        assert_delivery_backend()

    def test_categorical_checkpoint_resumes_in_frozen_worker_without_changing_parameters(self):
        variant = final.VARIANTS[0]
        loaded = self.engine.modules(self.manifest)
        model_class = loaded["booster"].HistogramGradientBoostingClassifier
        original_save = model_class.save_checkpoint

        def interrupted_save(model, path, metadata=None):
            original_save(model, path, metadata)
            raise RuntimeError("Synthetic categorical checkpoint interruption")

        with mock.patch.object(model_class, "save_checkpoint", new=interrupted_save), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Synthetic categorical"):
                final.worker(self.manifest_path, variant, "fold_1")
        final._launch(self.manifest_path, variant, "fold_1")
        completed = self.engine.completed(self.manifest, variant, "fold_1")
        self.assertEqual(completed["resumed_from_trees"], 25)
        self.assertEqual(completed["trees"], 50)
        train, labels, valid, _ = self.engine._matrices(
            self.manifest, variant, "fold_1", loaded["preprocessing"])
        parameters = self.manifest["variants"][variant]["model_parameters"]
        reference = model_class(**parameters).fit(train, labels)
        saved = np.load(Path(self.manifest["output"]) / variant / "fold_1/probabilities.npy")
        np.testing.assert_array_equal(saved, reference.predict_proba(valid)[:, 1])

    def test_only_booster_source_changes_and_frozen_controls_and_locks_reject_mutation(self):
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
        sources = self.manifest["sources"]
        self.assertEqual(set(sources), {"booster", "preprocessing", "baseline_preprocessing",
                                       "conditional_imputation", "evaluation"})
        self.assertEqual(Path(sources["booster"]["original_path"]).name, "numpy_boosting_categorical.py")
        self.assertEqual(sources["booster"]["original_sha256"],
                         final.checksum(final.ROOT / "src/numpy_boosting_categorical.py"))
        self.assertNotEqual(sources["booster"]["original_sha256"], CLASSIC_SHA)
        self.assertEqual(final.checksum(final.ROOT / "src/numpy_boosting_phase14.py"), CLASSIC_SHA)
        for key, source in sources.items():
            if key != "booster":
                self.assertEqual(source["original_sha256"],
                                 final.checksum(final.ROOT / "src" / Path(source["original_path"]).name))
        for path in (Path(sources["booster"]["path"]),
                     self.root / final.REFERENCES[final.VARIANTS[0]],
                     Path(self.manifest["categorical_wrapper"]["path"]),
                     Path(self.manifest["data"]["train_labels"]),
                     Path(self.manifest["folds_path"])):
            with self.subTest(path=path):
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"changed")
                    with self.assertRaisesRegex(ValueError, "[Ff]rozen|changed"):
                        self.engine.frozen(self.manifest, inputs=True)
                finally:
                    path.write_bytes(original)
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "2"}):
            with self.assertRaisesRegex(ValueError, "environment|Python|NumPy|thread|changed"):
                self.engine.frozen(self.manifest, inputs=True)
        with mock.patch.object(final, "_launch", side_effect=AssertionError("Unexpected launch")):
            with self.engine.file_lock(Path(self.manifest["output"]) / "run.lock"):
                with self.assertRaises(OSError):
                    final.run(self.manifest_path, workers=2)
            worker_lock = self.engine.task_directory(self.manifest, final.VARIANTS[0], "fold_3") / "worker.lock"
            with self.engine.file_lock(worker_lock):
                with self.assertRaises(OSError):
                    final.run(self.manifest_path, workers=2)
        changed = copy.deepcopy(self.manifest)
        changed["variants"][final.VARIANTS[0]]["model_parameters"]["categorical_feature_indices"] = [90, 93]
        final.write_json(self.manifest_path, changed)
        with self.assertRaisesRegex(ValueError, "manifest.*changed"):
            final.read_manifest(self.manifest_path)


if __name__ == "__main__":
    unittest.main()

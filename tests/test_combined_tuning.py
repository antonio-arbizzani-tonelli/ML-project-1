"""Verify isolated tuning of the recorded Phase 18+24, L2=1 submission."""

import copy
import csv
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

from tools import final_combined_tuning as final


DIET_RAW = {"FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1"}
DIET_DAILY = {"FTJUDA1_", "FRUTDA1_", "BEANDAY_", "GRENDAY_", "ORNGDAY_", "VEGEDA1_"}
IMPUTATION_FLAGS = {
    "INCOME2__was_unknown", "INCOME2__was_refused",
    "PNEUVAC3__was_unknown", "PNEUVAC3__was_refused", "EMPLOY1__was_refused",
    "HTM4__was_blank", "WTKG3__was_blank", "_BMI5__was_blank",
}
OVERRIDES = {
    "combined18_24_bins128": {"n_bins": 128},
    "combined18_24_features256": {"max_features": 256},
}


def preprocessing(root, suite):
    entries = final.read_json(root / suite["preprocessing_config_path"])["variants"]
    return next(item["preprocessing"] for item in entries
                if item["experiment_name"] == suite["preprocessing_variant"])


class TestCombinedTuningRecipes(unittest.TestCase):
    def test_each_production_recipe_changes_exactly_one_model_axis(self):
        final.validate_configuration(final.ROOT)
        base = final.read_json(final.ROOT / final.BASE_SUITE)
        expected_plan = preprocessing(final.ROOT, base)
        self.assertEqual(set(final.VARIANTS), set(OVERRIDES))
        for variant in final.VARIANTS:
            with self.subTest(variant=variant):
                suite = final.read_json(final.ROOT / final.REFERENCES[variant])
                parameters = suite["models"][0]["parameters"]
                expected = dict(base["models"][0]["parameters"], **OVERRIDES[variant])
                self.assertEqual(parameters, expected)
                changed = {key for key in parameters
                           if parameters[key] != base["models"][0]["parameters"][key]}
                self.assertEqual(changed, set(OVERRIDES[variant]))
                self.assertEqual(parameters["l2_regularization"], 1.0)
                self.assertEqual(suite["models"][0]["checkpoints"], [200])
                self.assertEqual(preprocessing(final.ROOT, suite), expected_plan)
                self.assertEqual(suite["data"], base["data"])
                self.assertEqual(suite["seed"], base["seed"])
                self.assertEqual(suite["fold_count"], 3)
                self.assertEqual(suite["preprocessing_variant"], base["preprocessing_variant"])
                self.assertEqual(final.EXPECTED_FEATURES[variant], 297)

    def test_unintended_extra_changes_are_rejected_before_prepare(self):
        with tempfile.TemporaryDirectory(dir=final.ROOT / ".local") as directory:
            root = Path(directory)
            for relative in (final.BASE_SUITE, final.PREPROCESSING_PATH,
                             *final.REFERENCES.values()):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(final.ROOT / relative, destination)
            final.validate_configuration(root)
            path = root / final.REFERENCES["combined18_24_bins128"]
            original = final.read_json(path)
            mutations = {
                "both_tuning_axes": lambda value: value["models"][0]["parameters"].update(max_features=256),
                "depth": lambda value: value["models"][0]["parameters"].update(max_depth=6),
                "regularization": lambda value: value["models"][0]["parameters"].update(l2_regularization=5.0),
                "trees": lambda value: value["models"][0].update(checkpoints=[250]),
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

    def test_a_third_parallel_worker_is_rejected_before_loading_data(self):
        with self.assertRaisesRegex(ValueError, "two|2"):
            final.run(Path("does-not-exist.json"), workers=3)


class TestCombinedTuningSyntheticSubmissions(unittest.TestCase):
    def setUp(self):
        (final.ROOT / ".local").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=final.ROOT / ".local")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for relative in (
            "src/numpy_boosting_phase14.py", "src/numpy_boosting.py",
            "src/feature_preprocessing.py", "src/conditional_imputation.py",
            "src/baseline_preprocessing.py", "src/evaluation.py",
            "src/run_preprocessing_cv.py", "run.py", "helpers.py",
        ):
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(final.ROOT / relative, destination)
        self.names = [item["name"] for item in final.read_json(
            final.ROOT / "configs/feature_metadata.json")
            if item["name"] not in {"Id", "_MICHD"}]
        self.assertEqual(len(self.names), 321)
        metadata_path = self.root / "configs/feature_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(final.ROOT / "configs/feature_metadata.json", metadata_path)
        baseline_path = self.root / final.BASE_SUITE
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(final.ROOT / final.BASE_SUITE, baseline_path)
        self.labels = np.tile(np.array([0, 0, 1], dtype=np.int8), 12)
        rng = np.random.default_rng(122)
        self.train = rng.integers(1, 3, size=(36, len(self.names))).astype(np.float32)
        synthetic = {
            "_AGE80": 30 + np.arange(36) % 10 + 30 * self.labels,
            "_BMI5": np.tile([2400, 2500, 2600, np.nan, 2800, 2300], 6),
            "HTM4": np.tile([160, 165, np.nan, 170, 175, 180], 6),
            "WTKG3": np.tile([6500, np.nan, 7000, 8000, 7500, 6500], 6),
            "INCOME2": np.tile([1, 2, 77, 99, np.nan, 1], 6),
            "PNEUVAC3": np.tile([1, 2, 7, 9, np.nan, 1], 6),
            "EMPLOY1": np.tile([1, 2, 9, 1, 2, 1], 6),
        }
        for name, values in synthetic.items():
            self.train[:, self.names.index(name)] = values
        selected = np.array([15, 2, 25, 4, 18, 11, 24, 7, 29, 16, 35, 0])
        self.test = self.train[selected].copy()
        self.train_ids = np.arange(1, 37, dtype=np.int64)
        self.test_ids = 100 + selected.astype(np.int64)
        raw, cache = self.root / "data/raw/dataset", self.root / "data/processed"
        raw.mkdir(parents=True)
        cache.mkdir(parents=True)
        for split, values, ids in (("train", self.train, self.train_ids),
                                   ("test", self.test, self.test_ids)):
            np.save(cache / f"x_{split}_float32.npy", values)
            np.save(cache / f"id_{split}_int64.npy", ids)
            self.write_csv(raw / f"x_{split}.csv", ["Id", *self.names],
                           [(int(row_id), *row) for row_id, row in zip(ids, values)])
        signed = (2 * self.labels - 1).astype(np.int8)
        np.save(cache / "y_train_int8.npy", signed)
        self.write_csv(raw / "y_train.csv", ["Id", "_MICHD"], zip(self.train_ids, signed))
        self.write_csv(raw / "sample_submission.csv", ["Id", "Prediction"],
                       zip(self.test_ids, np.ones(12, dtype=np.int8)))
        plans = final.read_json(final.ROOT / final.PREPROCESSING_PATH)
        for item in plans["variants"]:
            item["preprocessing"]["imputation"].update(
                minimum_observed=2, anchor_bins=4,
                maximum_continuous_training_rows=36, prediction_batch_size=12)
        final.write_json(self.root / final.PREPROCESSING_PATH, plans)
        for variant in final.VARIANTS:
            suite = final.read_json(final.ROOT / final.REFERENCES[variant])
            # Keep the actual distinct bin/feature settings in these small fits.
            suite["models"][0]["parameters"].update(max_depth=2, min_samples_leaf=1)
            suite["models"][0]["checkpoints"] = [50]
            final.write_json(self.root / final.REFERENCES[variant], suite)
        for patch in (mock.patch.object(final, "EXPECTED_SHAPE", (36, 12, 321)),
                      mock.patch.object(final, "validate_configuration", return_value=[])):
            patch.start()
            self.addCleanup(patch.stop)
        self.manifest = final.prepare(self.root, self.root / "output")
        self.manifest_path = self.root / "output/manifest.json"

    @staticmethod
    def write_csv(path, header, rows):
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    def test_two_slots_deliver_each_csv_with_its_own_oof_threshold_and_id_order(self):
        engine = final._engine(self.manifest)
        first, second = final.VARIANTS
        first_delivered = threading.Event()
        initial_pair = threading.Barrier(2)
        guard = threading.Lock()
        active = maximum_active = 0
        original_deliver = final._deliver_variant

        def launch(manifest_path, variant, partition):
            nonlocal active, maximum_active
            with guard:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                if partition == "fold_1":
                    initial_pair.wait(timeout=10)
                if variant == second and partition == "full":
                    self.assertTrue(first_delivered.wait(timeout=20), "First CSV was withheld")
                final.worker(manifest_path, variant, partition)
            finally:
                with guard:
                    active -= 1

        def deliver(manifest, variant, record):
            result = original_deliver(manifest, variant, record)
            if variant == first:
                self.assertIsNone(engine.completed(manifest, second, "full"))
                destination = Path(manifest["delivery"]) / f"submission_{first}_full_train_oof.csv"
                self.assertTrue(destination.is_file())
                first_delivered.set()
            return result

        with mock.patch.object(final, "_launch", side_effect=launch), \
                mock.patch.object(final, "_deliver_variant", side_effect=deliver):
            comparison = final.run(self.manifest_path, workers=2)
        self.assertEqual(maximum_active, 2)
        self.assertTrue(first_delivered.is_set())
        loaded = engine.modules(self.manifest)
        delivery = final.read_json(Path(self.manifest["delivery"]) / "manifest.json")
        self.assertEqual(delivery["run_manifest_sha256"], final.checksum(self.manifest_path))
        self.assertTrue(delivery["complete"])
        self.assertFalse(delivery["uploaded_to_aicrowd"])
        self.assertEqual(set(comparison["variants"]), set(final.VARIANTS))
        for variant in final.VARIANTS:
            with self.subTest(variant=variant):
                directory = Path(self.manifest["output"]) / variant
                summary = comparison["variants"][variant]
                with np.load(directory / "full_train_oof.npz") as oof:
                    np.testing.assert_array_equal(oof["row_indices"], np.arange(36))
                    np.testing.assert_array_equal(oof["train_ids"], self.train_ids)
                    np.testing.assert_array_equal(oof["labels"], self.labels)
                    np.testing.assert_array_equal(oof["fold_ids"], np.load(self.manifest["folds_path"]))
                    threshold = loaded["evaluation"].best_f1_threshold(
                        self.labels, oof["probabilities"]).threshold
                    self.assertEqual(threshold, summary["decision_threshold"])
                    for fold in (1, 2, 3):
                        rows = np.flatnonzero(oof["fold_ids"] == fold)
                        fit = engine.completed(self.manifest, variant, f"fold_{fold}")
                        self.assertEqual(fit["signature"]["training_shape"], [24, 297])
                        np.testing.assert_array_equal(oof["probabilities"][rows],
                                                      np.load(directory / f"fold_{fold}/probabilities.npy"))
                full = engine.completed(self.manifest, variant, "full")
                self.assertEqual(full["signature"]["training_shape"], [36, 297])
                names = set(full["signature"]["output_feature_names"])
                self.assertFalse(DIET_RAW & names)
                self.assertTrue(DIET_DAILY <= names)
                self.assertTrue(IMPUTATION_FLAGS <= names)
                probabilities = np.load(directory / "full/probabilities.npy")
                delivered = delivery["variants"][variant]
                csv_path = Path(self.manifest["delivery"]) / delivered["csv"]
                ids, predictions = engine._read_pair_csv(csv_path, ["Id", "Prediction"])
                np.testing.assert_array_equal(ids, self.test_ids)
                np.testing.assert_array_equal(predictions, np.where(probabilities >= threshold, 1, -1))
                self.assertTrue(np.all(np.isin(predictions, [-1, 1])))
                self.assertEqual(delivered["sha256"], final.checksum(csv_path))
                self.assertEqual(delivered["ids_sha256"], engine.array_checksum(self.test_ids))
                self.assertEqual(delivered["rows"], 12)
                self.assertIsNone(summary["official_test_f1"])
        with mock.patch.object(final, "_launch", side_effect=AssertionError("Unexpected retraining")):
            repeated = final.run(self.manifest_path, workers=2)
        for variant in final.VARIANTS:
            self.assertEqual(comparison["variants"][variant]["submission"]["sha256"],
                             repeated["variants"][variant]["submission"]["sha256"])
        frozen_run = subprocess.run(
            [sys.executable, "-B", self.manifest["driver"], "run",
             "--manifest", str(self.manifest_path), "--workers", "2"],
            cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(frozen_run.returncode, 0, frozen_run.stderr)
        self.assertNotIn("START ", frozen_run.stdout)
        reproduced = final.read_json(Path(self.manifest["output"]) / "comparison.json")
        for variant in final.VARIANTS:
            self.assertEqual(comparison["variants"][variant]["submission"]["sha256"],
                             reproduced["variants"][variant]["submission"]["sha256"])

    def test_both_recipes_share_exact_matrices_and_training_only_imputation(self):
        engine = final._engine(self.manifest)
        loaded = engine.modules(self.manifest)
        imputer = loaded["conditional_imputation"]
        fold_ids = np.load(self.manifest["folds_path"])
        train_rows, valid_rows = np.flatnonzero(fold_ids != 1), np.flatnonzero(fold_ids == 1)
        first, second = final.VARIANTS
        anchors = self.manifest["variants"][first]["preprocessing"]["imputation"]["anchor_features"]
        observed_anchors = []
        original_fit = imputer._AnchorBinner.fit

        def record_fit(binner, values):
            observed_anchors.append(values.copy())
            return original_fit(binner, values)

        with mock.patch.object(imputer._AnchorBinner, "fit", new=record_fit):
            before = engine._matrices(self.manifest, first, "fold_1", loaded["preprocessing"])
        self.assertEqual(len(observed_anchors), 1)
        self.assertEqual(observed_anchors[0].shape, (24, len(anchors)))
        np.testing.assert_array_equal(observed_anchors[0][:, anchors.index("_AGE80")],
                                      self.train[train_rows, self.names.index("_AGE80")])
        other = engine._matrices(self.manifest, second, "fold_1", loaded["preprocessing"])
        for index in (0, 1, 2):
            np.testing.assert_array_equal(before[index], other[index])
        self.assertEqual(before[3]["output_feature_names"], other[3]["output_feature_names"])
        self.assertEqual(before[3]["training_matrix_sha256"], other[3]["training_matrix_sha256"])
        self.assertEqual(before[3]["validation_matrix_sha256"], other[3]["validation_matrix_sha256"])
        changed = self.train.copy()
        changed[valid_rows] = 999
        np.save(self.manifest["data"]["train_features"], changed)
        after = engine._matrices(self.manifest, first, "fold_1", loaded["preprocessing"])
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[3]["output_feature_names"], after[3]["output_feature_names"])

    def test_frozen_worker_resumes_checkpoint_with_the_actual_tuning_parameters(self):
        engine = final._engine(self.manifest)
        loaded = engine.modules(self.manifest)
        model_class = loaded["booster"].HistogramGradientBoostingClassifier
        original_save = model_class.save_checkpoint
        variant = "combined18_24_features256"

        def interrupted_save(model, path, metadata=None):
            original_save(model, path, metadata)
            raise RuntimeError("Synthetic interruption")

        with mock.patch.object(model_class, "save_checkpoint", new=interrupted_save):
            with self.assertRaisesRegex(RuntimeError, "Synthetic interruption"):
                final.worker(self.manifest_path, variant, "fold_1")
        final._launch(self.manifest_path, variant, "fold_1")
        fit = engine.completed(self.manifest, variant, "fold_1")
        self.assertEqual(fit["resumed_from_trees"], 25)
        self.assertEqual(fit["trees"], 50)
        train, labels, valid, _ = engine._matrices(
            self.manifest, variant, "fold_1", loaded["preprocessing"])
        parameters = self.manifest["variants"][variant]["model_parameters"]
        self.assertEqual(parameters["max_features"], 256)
        self.assertEqual(parameters["n_bins"], 64)
        reference = model_class(**parameters).fit(train, labels)
        saved = np.load(Path(self.manifest["output"]) / variant / "fold_1/probabilities.npy")
        np.testing.assert_array_equal(saved, reference.predict_proba(valid)[:, 1])

    def test_frozen_scheduler_configuration_and_manifest_reject_changes(self):
        engine = final._engine(self.manifest)
        engine.frozen(self.manifest, inputs=True)
        for path in (Path(self.manifest["tuning_scheduler"]["path"]),
                     Path(self.manifest["tuning_wrapper"]["path"]),
                     self.root / final.REFERENCES["combined18_24_bins128"]):
            with self.subTest(path=path):
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"modified")
                    with self.assertRaisesRegex(ValueError, "[Ff]rozen"):
                        engine.frozen(self.manifest, inputs=True)
                finally:
                    path.write_bytes(original)
        changed = copy.deepcopy(self.manifest)
        changed["variants"]["combined18_24_bins128"]["model_parameters"]["max_features"] = 256
        final.write_json(self.manifest_path, changed)
        with self.assertRaisesRegex(ValueError, "manifest.*changed"):
            final.read_manifest(self.manifest_path)


if __name__ == "__main__":
    unittest.main()

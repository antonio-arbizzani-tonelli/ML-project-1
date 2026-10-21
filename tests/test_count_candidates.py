"""Verify documented count features and independently delivered A, B and A+B fits."""

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

from src.count_features import append_count_features
from tools import final_count_candidates as final


GROUP_A = ("BLIND", "DECIDE", "USEEQUIP")
GROUP_B = ("ADDEPEV2", "CHCCOPD1", "CHCKIDNY")
COUNT_A = ("FUNCTIONAL__yes_count", "FUNCTIONAL__answered_count")
COUNT_B = ("DIAGNOSES__yes_count", "DIAGNOSES__answered_count")
COUNT_NAMES = {"count_a": COUNT_A, "count_b": COUNT_B, "count_ab": (*COUNT_A, *COUNT_B)}


def preprocessing(root, suite):
    variants = final.read_json(root / suite["preprocessing_config_path"])["variants"]
    return next(item["preprocessing"] for item in variants
                if item["experiment_name"] == suite["preprocessing_variant"])


class TestDocumentedCountFeatures(unittest.TestCase):
    def test_yes_no_nonresponse_and_all_missing_have_distinct_meanings(self):
        metadata = {item["name"]: item for item in final.read_json(final.ROOT / "configs/feature_metadata.json")}
        for feature in (*GROUP_A, *GROUP_B):
            values = {str(item["code"]): item["label"] for item in metadata[feature]["documented_values"]}
            self.assertEqual(values["1"], "Yes")
            self.assertEqual(values["2"], "No")
            self.assertTrue({"7", "9"} <= {str(item["code"]) for item in metadata[feature]["special_values"]})
        raw_names = [*GROUP_A, *GROUP_B]
        raw = np.array([
            [1, 1, 1, 2, 2, 2],
            [1, 2, 7, 2, 1, 9],
            [7, 9, np.nan, np.nan, 7, 9],
            [2, np.nan, 2, 1, 2, np.nan],
            [-1, 1, 99, 1, np.nan, 1],
        ], dtype=np.float32)
        base = np.column_stack((raw[:, :2], [np.nan, 0, 1, 2, 3])).astype(np.float32)
        base_names = ["BLIND", "DECIDE", "existing_flag"]
        expected_a = np.array([[3, 3], [1, 2], [np.nan, 0], [0, 2], [1, 1]], dtype=np.float32)
        expected_b = np.array([[0, 3], [1, 2], [np.nan, 0], [1, 2], [2, 2]], dtype=np.float32)
        for group, expected, appended in (("A", expected_a, COUNT_A), ("B", expected_b, COUNT_B)):
            with self.subTest(group=group):
                transformed, names = append_count_features(base, base_names, raw, raw_names, (group,))
                np.testing.assert_array_equal(transformed[:, :base.shape[1]], base)
                np.testing.assert_array_equal(transformed[:, base.shape[1]:], expected)
                self.assertEqual(names, [*base_names, *appended])
        combined, names = append_count_features(base, base_names, raw, raw_names, ("A", "B"))
        np.testing.assert_array_equal(combined, np.column_stack((base, expected_a, expected_b)))
        self.assertEqual(names, [*base_names, *COUNT_A, *COUNT_B])
        np.testing.assert_array_equal(raw[:, :2], base[:, :2])
        corrupted = raw.copy()
        corrupted[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "infinite"):
            append_count_features(base, base_names, corrupted, raw_names, ("A",))


class TestCountCandidateRecipe(unittest.TestCase):
    def test_all_three_challengers_preserve_the_winning_base_and_fixed_model(self):
        final.validate_configuration(final.ROOT)
        base_suite = final.read_json(final.ROOT / "configs/experiments/combined18_24_l2_1.json")
        base_plan = preprocessing(final.ROOT, base_suite)
        for variant, expected_features in (("count_a", 299), ("count_b", 299), ("count_ab", 301)):
            with self.subTest(variant=variant):
                suite = final.read_json(final.ROOT / final.REFERENCES[variant])
                plan = copy.deepcopy(preprocessing(final.ROOT, suite))
                counts = plan.pop("derived_counts")
                self.assertEqual(counts["groups"], {"count_a": ["A"], "count_b": ["B"],
                                                   "count_ab": ["A", "B"]}[variant])
                self.assertEqual(plan, base_plan)
                self.assertEqual(suite["models"][0]["parameters"], base_suite["models"][0]["parameters"])
                self.assertEqual(suite["models"][0]["checkpoints"], [200])
                self.assertEqual(suite["models"][0]["parameters"]["l2_regularization"], 1.0)
                self.assertEqual(suite["seed"], base_suite["seed"])
                self.assertEqual(suite["fold_count"], 3)
                self.assertEqual(suite["data"], base_suite["data"])
                self.assertEqual(final.EXPECTED_FEATURES[variant], expected_features)

    def test_a_third_parallel_worker_is_rejected_before_loading_data(self):
        with self.assertRaisesRegex(ValueError, "two|2"):
            final.run(Path("does-not-exist.json"), workers=3)


class TestCountSyntheticSubmissions(unittest.TestCase):
    def setUp(self):
        (final.ROOT / ".local").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=final.ROOT / ".local")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for relative in (
            "src/numpy_boosting_phase14.py", "src/numpy_boosting.py",
            "src/feature_preprocessing.py", "src/conditional_imputation.py",
            "src/baseline_preprocessing.py", "src/evaluation.py", "src/count_features.py",
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
            "BLIND": np.tile([1, 2, 7, 9, np.nan, 1], 6),
            "DECIDE": np.tile([1, 2, 9, 7, np.nan, 2], 6),
            "USEEQUIP": np.tile([2, 2, np.nan, 9, np.nan, 1], 6),
            "ADDEPEV2": np.tile([2, 1, 9, 7, np.nan, 1], 6),
            "CHCCOPD1": np.tile([1, 1, np.nan, 9, np.nan, 2], 6),
            "CHCKIDNY": np.tile([1, 2, 7, 7, np.nan, 1], 6),
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
        for variant in final.VARIANTS:
            suite = final.read_json(final.ROOT / final.REFERENCES[variant])
            plans_path = Path(suite["preprocessing_config_path"])
            plans = final.read_json(final.ROOT / plans_path)
            for item in plans["variants"]:
                item["preprocessing"]["imputation"].update(
                    minimum_observed=2, anchor_bins=4,
                    maximum_continuous_training_rows=36, prediction_batch_size=12)
            final.write_json(self.root / plans_path, plans)
            suite["models"][0]["parameters"].update(
                max_depth=2, min_samples_leaf=1, n_bins=4, max_features=8)
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

    def test_two_slots_start_ab_as_one_frees_and_publish_each_csv_immediately(self):
        engine = final._engine(self.manifest)
        first, second, third = final.VARIANTS
        first_delivered = threading.Event()
        third_started = threading.Event()
        second_waiting = threading.Event()
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
                if partition == "fold_1" and variant in (first, second):
                    initial_pair.wait(timeout=10)
                if variant == second and partition == "fold_1":
                    second_waiting.set()
                    self.assertTrue(third_started.wait(timeout=20), "A+B did not use the free slot")
                if variant == third and partition == "fold_1":
                    self.assertTrue(second_waiting.is_set())
                    self.assertTrue(first_delivered.is_set())
                    third_started.set()
                final.worker(manifest_path, variant, partition)
            finally:
                with guard:
                    active -= 1

        def deliver(manifest, variant, record):
            result = original_deliver(manifest, variant, record)
            if variant == first:
                self.assertIsNone(engine.completed(manifest, second, "full"))
                self.assertIsNone(engine.completed(manifest, third, "full"))
                destination = Path(manifest["delivery"]) / f"submission_{first}_full_train_oof.csv"
                self.assertTrue(destination.is_file())
                first_delivered.set()
            return result

        with mock.patch.object(final, "_launch", side_effect=launch), \
                mock.patch.object(final, "_deliver_variant", side_effect=deliver):
            comparison = final.run(self.manifest_path, workers=2)
        self.assertEqual(maximum_active, 2)
        self.assertTrue(third_started.is_set())
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
                features = final.EXPECTED_FEATURES[variant]
                with np.load(directory / "full_train_oof.npz") as oof:
                    np.testing.assert_array_equal(oof["row_indices"], np.arange(36))
                    np.testing.assert_array_equal(oof["train_ids"], self.train_ids)
                    np.testing.assert_array_equal(oof["labels"], self.labels)
                    threshold = loaded["evaluation"].best_f1_threshold(
                        self.labels, oof["probabilities"]).threshold
                    self.assertEqual(threshold, summary["decision_threshold"])
                    for fold in (1, 2, 3):
                        rows = np.flatnonzero(oof["fold_ids"] == fold)
                        fit = engine.completed(self.manifest, variant, f"fold_{fold}")
                        self.assertEqual(fit["signature"]["training_shape"], [24, features])
                        np.testing.assert_array_equal(oof["probabilities"][rows],
                                                      np.load(directory / f"fold_{fold}/probabilities.npy"))
                full = engine.completed(self.manifest, variant, "full")
                self.assertEqual(full["signature"]["training_shape"], [36, features])
                self.assertEqual(full["signature"]["output_feature_names"][-len(COUNT_NAMES[variant]):],
                                 list(COUNT_NAMES[variant]))
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
        self.assertIn("three count-candidate", reproduced["next_action"])
        for variant in final.VARIANTS:
            self.assertEqual(comparison["variants"][variant]["submission"]["sha256"],
                             reproduced["variants"][variant]["submission"]["sha256"])

    def test_count_columns_match_ab_and_base_imputation_fits_training_rows_only(self):
        engine = final._engine(self.manifest)
        loaded = engine.modules(self.manifest)
        imputer = loaded["conditional_imputation"]
        fold_ids = np.load(self.manifest["folds_path"])
        train_rows, valid_rows = np.flatnonzero(fold_ids != 1), np.flatnonzero(fold_ids == 1)
        first = final.VARIANTS[0]
        plan = copy.deepcopy(self.manifest["variants"][first]["preprocessing"])
        plan.pop("derived_counts")
        anchors = plan["imputation"]["anchor_features"]
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
        base_train, base_valid, base_names = loaded["preprocessing"].tree_feature_matrices(
            self.train[train_rows], self.train[valid_rows], self.names,
            self.manifest["variants"][first]["metadata"], plan)
        self.assertEqual(len(base_names), 297)
        for variant in final.VARIANTS:
            transformed = engine._matrices(self.manifest, variant, "fold_1", loaded["preprocessing"])
            np.testing.assert_array_equal(transformed[0][:, :297], base_train)
            np.testing.assert_array_equal(transformed[2][:, :297], base_valid)
            self.assertEqual(transformed[3]["output_feature_names"][:297], base_names)
        combined = engine._matrices(self.manifest, "count_ab", "fold_1", loaded["preprocessing"])
        separate_b = engine._matrices(self.manifest, "count_b", "fold_1", loaded["preprocessing"])
        for matrix_index in (0, 2):
            np.testing.assert_array_equal(before[matrix_index][:, -2:], combined[matrix_index][:, 297:299])
            np.testing.assert_array_equal(separate_b[matrix_index][:, -2:], combined[matrix_index][:, 299:301])
        changed = self.train.copy()
        changed[valid_rows] = 999
        np.save(self.manifest["data"]["train_features"], changed)
        after = engine._matrices(self.manifest, first, "fold_1", loaded["preprocessing"])
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[3]["output_feature_names"], after[3]["output_feature_names"])

    def test_frozen_count_helper_runs_in_fresh_worker_and_resumes_exact_checkpoint(self):
        engine = final._engine(self.manifest)
        loaded = engine.modules(self.manifest)
        model_class = loaded["booster"].HistogramGradientBoostingClassifier
        original_save = model_class.save_checkpoint
        variant = "count_ab"

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
        reference = model_class(**self.manifest["variants"][variant]["model_parameters"]).fit(train, labels)
        saved = np.load(Path(self.manifest["output"]) / variant / "fold_1/probabilities.npy")
        np.testing.assert_array_equal(saved, reference.predict_proba(valid)[:, 1])

    def test_frozen_helper_configuration_and_manifest_reject_changes(self):
        engine = final._engine(self.manifest)
        engine.frozen(self.manifest, inputs=True)
        for path in (Path(self.manifest["sources"]["count_features"]["path"]),
                     self.root / final.REFERENCES["count_ab"]):
            with self.subTest(path=path):
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"modified")
                    with self.assertRaisesRegex(ValueError, "[Ff]rozen"):
                        engine.frozen(self.manifest, inputs=True)
                finally:
                    path.write_bytes(original)
        changed = copy.deepcopy(self.manifest)
        changed["variants"]["count_ab"]["expected_features"] = 299
        final.write_json(self.manifest_path, changed)
        with self.assertRaisesRegex(ValueError, "manifest.*changed"):
            final.read_manifest(self.manifest_path)


if __name__ == "__main__":
    unittest.main()

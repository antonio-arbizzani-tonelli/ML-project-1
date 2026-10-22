"""Exercise the three submission pipelines on small, independently ordered data."""

import copy
import csv
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from tools import final_promising_candidates as final


class TestPromisingCandidateSubmissions(unittest.TestCase):
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

        self.names = ["ANCHOR", "ANSWER", "FRUIT1"]
        self.labels = np.tile(np.array([0, 0, 1], dtype=np.int8), 12)
        self.train = np.column_stack((
            np.arange(36) % 4, np.tile([1, 2, 1, 2, 7, 9], 6),
            self.labels * 3 + np.arange(36) % 3,
        )).astype(np.float32)
        self.test = self.train[[15, 2, 25, 4, 18, 11, 24, 7, 29, 16, 35, 0]].copy()
        self.train_ids = np.arange(1, 37, dtype=np.int64)
        self.test_ids = np.array([115, 102, 125, 104, 118, 111, 124, 107, 129, 116, 135, 100], dtype=np.int64)
        raw = self.root / "data/raw/dataset"
        cache = self.root / "data/processed"
        raw.mkdir(parents=True)
        cache.mkdir(parents=True)
        for split, values, ids in (
            ("train", self.train, self.train_ids),
            ("test", self.test, self.test_ids),
        ):
            np.save(cache / f"x_{split}_float32.npy", values)
            np.save(cache / f"id_{split}_int64.npy", ids)
            self.write_csv(raw / f"x_{split}.csv", ["Id", *self.names],
                           [(int(row_id), *row) for row_id, row in zip(ids, values)])
        signed = (2 * self.labels - 1).astype(np.int8)
        np.save(cache / "y_train_int8.npy", signed)
        self.write_csv(raw / "y_train.csv", ["Id", "_MICHD"], zip(self.train_ids, signed))
        self.write_csv(raw / "sample_submission.csv", ["Id", "Prediction"],
                       zip(self.test_ids, np.ones(12, dtype=np.int8)))

        metadata = [
            {"name": "ANCHOR", "semantic_type": "continuous", "special_values": []},
            {"name": "ANSWER", "semantic_type": "categorical", "source_kind": "core_survey",
             "special_values": [{"code": 7, "label": "Don't know/Not Sure"},
                                {"code": 9, "label": "Refused"}]},
            {"name": "FRUIT1", "semantic_type": "continuous", "special_values": []},
        ]
        final.write_json(self.root / "configs/feature_metadata.json", metadata)
        base_plan = {"name": "codebook_aware"}
        self.plans = {
            "phase18": {**base_plan, "exclude_features": ["FRUIT1"]},
            "phase24": {**base_plan, "imputation": {
                "anchor_features": ["ANCHOR"], "nonresponse_features": ["ANSWER"],
                "minimum_observed": 2, "anchor_bins": 4, "fill_values": True,
            }},
            "phase25": base_plan,
        }
        self.parameters = {"learning_rate": 0.05, "max_depth": 2,
                           "min_samples_leaf": 1, "n_bins": 4,
                           "l2_regularization": 1.0, "max_features": 2,
                           "random_seed": 20, "n_estimators": 50}
        for variant in final.VARIANTS:
            preprocessing_path = f"configs/experiments/{variant}_mini_preprocessing.json"
            final.write_json(self.root / preprocessing_path, {"variants": [
                {"experiment_name": variant, "preprocessing": self.plans[variant]},
            ]})
            parameters = dict(self.parameters)
            parameters.pop("n_estimators")
            if variant == "phase25":
                parameters["l2_regularization"] = 5.0
            suite = {"fold_count": 3, "seed": 19,
                     "data": {"features_path": "data/processed/x_train_float32.npy",
                              "labels_path": "data/processed/y_train_int8.npy",
                              "x_train_csv_path": "data/raw/dataset/x_train.csv"},
                     "feature_metadata_path": "configs/feature_metadata.json",
                     "preprocessing_config_path": preprocessing_path,
                     "preprocessing_variant": variant,
                     "models": [{"parameters": parameters, "checkpoints": [50]}]}
            final.write_json(self.root / final.REFERENCES[variant], suite)
        self.feature_counts = {"phase18": 2, "phase24": 5, "phase25": 3}
        with mock.patch.object(final, "EXPECTED_SHAPE", (36, 12, 3)), \
                mock.patch.object(final, "EXPECTED_FEATURES", self.feature_counts), \
                mock.patch.object(final, "EXPECTED_FOLD_SEED", 19):
            self.manifest = final.prepare(self.root, self.root / "output")
        self.manifest_path = self.root / "output/manifest.json"

    @staticmethod
    def write_csv(path, header, rows):
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    def local_run(self):
        with mock.patch.object(final, "_launch", side_effect=final.worker):
            return final.run(self.manifest_path, workers=1)

    def test_three_complete_calibrations_and_signed_csvs_use_their_own_thresholds(self):
        comparison = self.local_run()
        self.assertIsNone(comparison["official_test_f1_comparison"])
        self.assertIn("not an unbiased test score", comparison["calibration_warning"])
        loaded = final.modules(self.manifest)
        delivery = final.read_json(comparison["delivery"]["manifest"])
        self.assertEqual(delivery["run_manifest_sha256"], final.checksum(self.manifest_path))
        self.assertEqual(delivery["historical_booster_sha256"], final.HISTORICAL_SHA256)
        self.assertFalse(delivery["uploaded_to_aicrowd"])
        self.assertEqual(set(delivery["variants"]), {"phase18", "phase24", "phase25"})
        self.assertEqual(set(comparison["variants"]), {"phase18", "phase24", "phase25"})
        for variant in final.VARIANTS:
            with self.subTest(variant=variant):
                directory = self.root / "output" / variant
                summary = comparison["variants"][variant]
                with np.load(directory / "full_train_oof.npz") as oof:
                    np.testing.assert_array_equal(oof["row_indices"], np.arange(36))
                    np.testing.assert_array_equal(oof["train_ids"], self.train_ids)
                    np.testing.assert_array_equal(oof["labels"], self.labels)
                    threshold = loaded["evaluation"].best_f1_threshold(
                        self.labels, oof["probabilities"]).threshold
                    self.assertEqual(threshold, summary["decision_threshold"])
                    for fold, (train, valid) in enumerate(final.full_train_folds(self.labels, 19), 1):
                        self.assertEqual(np.intersect1d(train, valid).size, 0)
                        np.testing.assert_array_equal(np.flatnonzero(oof["fold_ids"] == fold), valid)
                        np.testing.assert_array_equal(oof["probabilities"][valid],
                                                      np.load(directory / f"fold_{fold}/probabilities.npy"))
                        fit = final.completed(self.manifest, variant, f"fold_{fold}")
                        self.assertEqual(fit["signature"]["training_shape"], [24, self.feature_counts[variant]])
                with Path(summary["submission"]["path"]).open(newline="") as handle:
                    rows = list(csv.reader(handle))
                self.assertEqual(rows[0], ["Id", "Prediction"])
                self.assertEqual(len(rows), 13)
                np.testing.assert_array_equal([int(row[0]) for row in rows[1:]], self.test_ids)
                predictions = np.array([int(row[1]) for row in rows[1:]])
                self.assertTrue(np.all(np.isin(predictions, [-1, 1])))
                probabilities = np.load(directory / "full/probabilities.npy")
                np.testing.assert_array_equal(predictions, np.where(probabilities >= threshold, 1, -1))
                config = final.read_json(directory / "calibrated_final_config.json")
                self.assertEqual(config["threshold_training_rows"], 36)
                self.assertEqual(config["model"], self.manifest["variants"][variant]["model_parameters"])
                self.assertIn(f'--manifest "{self.manifest_path}"', config["reproduction_command"])
                self.assertIsNone(summary["official_test_f1"])
                delivered = delivery["variants"][variant]
                delivered_path = Path(comparison["delivery"]["directory"]) / delivered["csv"]
                self.assertEqual(delivered_path.read_bytes(), Path(summary["submission"]["path"]).read_bytes())
                self.assertEqual(delivered["sha256"], final.checksum(delivered_path))
                self.assertEqual(delivered["rows"], 12)
                self.assertEqual(delivered["decision_threshold"], threshold)
                self.assertEqual(delivered["threshold_training_rows"], 36)
                self.assertEqual(delivered["features"], self.feature_counts[variant])
                self.assertEqual(delivered["model_parameters"], config["model"])
                self.assertEqual(delivered["ids_sha256"], final.array_checksum(self.test_ids))
                self.assertIsNone(delivered["official_test_f1"])

        # Completed fits are reusable: a second run must not launch any model.
        with mock.patch.object(final, "_launch", side_effect=AssertionError("unexpected retraining")):
            repeated = final.run(self.manifest_path, workers=1)
        for variant in final.VARIANTS:
            self.assertEqual(comparison["variants"][variant]["submission"]["sha256"],
                             repeated["variants"][variant]["submission"]["sha256"])

    def test_each_candidate_uses_its_own_preprocessing_and_l2(self):
        loaded = final.modules(self.manifest)
        for variant in final.VARIANTS:
            with self.subTest(variant=variant):
                specification = self.manifest["variants"][variant]
                self.assertEqual(specification["preprocessing"], self.plans[variant])
                parameters = dict(self.parameters, l2_regularization=5.0 if variant == "phase25" else 1.0)
                self.assertEqual(specification["model_parameters"], parameters)
                train, labels, valid, signature = final._matrices(
                    self.manifest, variant, "full", loaded["preprocessing"])
                self.assertEqual(train.shape, (36, self.feature_counts[variant]))
                self.assertEqual(valid.shape, (12, self.feature_counts[variant]))
                np.testing.assert_array_equal(labels, self.labels)
                names = signature["output_feature_names"]
                if variant == "phase18":
                    self.assertEqual(names, ["ANCHOR", "ANSWER"])
                elif variant == "phase24":
                    self.assertEqual(names, [*self.names, "ANSWER__was_unknown", "ANSWER__was_refused"])
                    self.assertTrue(np.all(np.isfinite(train[:, 1])))
                else:
                    self.assertEqual(names, self.names)
                    self.assertTrue(np.any(np.isnan(train[:, 1])))

    def test_phase24_imputer_fits_on_fold_training_features_only(self):
        loaded = final.modules(self.manifest)
        imputer = loaded["conditional_imputation"]
        fold_ids = np.load(self.manifest["folds_path"])
        train_rows, valid_rows = np.flatnonzero(fold_ids != 1), np.flatnonzero(fold_ids == 1)
        fitted_anchors = []
        original_fit = imputer._AnchorBinner.fit

        def record_fit(binner, values):
            fitted_anchors.append(values.copy())
            return original_fit(binner, values)

        with mock.patch.object(imputer._AnchorBinner, "fit", new=record_fit), \
                mock.patch.object(imputer, "_categorical_predictions",
                                  wraps=imputer._categorical_predictions) as predict:
            before = final._matrices(self.manifest, "phase24", "fold_1", loaded["preprocessing"])
        self.assertEqual(len(fitted_anchors), 1)
        np.testing.assert_array_equal(fitted_anchors[0], self.train[train_rows, :1])
        self.assertEqual(predict.call_count, 1)
        training_bins, observed_values, observed_mask = predict.call_args.args[:3]
        self.assertEqual(training_bins.shape[0], len(train_rows))
        expected_observed = self.train[train_rows, 1].copy()
        expected_observed[np.isin(expected_observed, [7, 9])] = np.nan
        np.testing.assert_array_equal(observed_values, expected_observed)
        np.testing.assert_array_equal(observed_mask, np.isfinite(expected_observed))

        # Change held-out answers and anchors drastically; fitted training values
        # and learned feature names must still be identical.
        changed = self.train.copy()
        changed[valid_rows, 0] = 999
        changed[valid_rows, 1] = 99
        np.save(self.manifest["data"]["train_features"], changed)
        after = final._matrices(self.manifest, "phase24", "fold_1", loaded["preprocessing"])
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[3]["output_feature_names"], after[3]["output_feature_names"])
        self.assertTrue(np.all(after[2][:, 1] == 99))

    def test_interrupted_phase24_checkpoint_resumes_exactly_including_frozen_imputation(self):
        loaded = final.modules(self.manifest)
        model_class = loaded["booster"].HistogramGradientBoostingClassifier
        save = model_class.save_checkpoint

        def interrupted_save(model, path, metadata=None):
            save(model, path, metadata)
            raise RuntimeError("simulated interruption")

        with mock.patch.object(model_class, "save_checkpoint", new=interrupted_save):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                final.worker(self.manifest_path, "phase24", "fold_1")
        checkpoint = self.root / "output/phase24/fold_1/model.pkl"
        partial, metadata = model_class.load_checkpoint(checkpoint)
        self.assertEqual(len(partial.trees_), 25)
        self.assertEqual(metadata["variant"], "phase24")
        # Resume through the real frozen-driver subprocess used for long fits.
        final._launch(self.manifest_path, "phase24", "fold_1")
        completed = final.completed(self.manifest, "phase24", "fold_1")
        self.assertEqual(completed["resumed_from_trees"], 25)
        train, labels, valid, _ = final._matrices(
            self.manifest, "phase24", "fold_1", loaded["preprocessing"])
        reference = model_class(**self.manifest["variants"]["phase24"]["model_parameters"]).fit(train, labels)
        np.testing.assert_array_equal(np.load(checkpoint.parent / "probabilities.npy"),
                                      reference.predict_proba(valid)[:, 1])

        # A new interpreter must reconstruct the private package to unpickle it.
        code = ("import sys; from tools.final_promising_candidates import read_manifest,modules; "
                "m=modules(read_manifest(sys.argv[1])); "
                "model,metadata=m['booster'].HistogramGradientBoostingClassifier.load_checkpoint(sys.argv[2]); "
                "assert len(model.trees_)==50; assert metadata['variant']=='phase24'")
        process = subprocess.run([sys.executable, "-B", "-c", code,
                                  str(self.manifest_path), str(checkpoint)],
                                 cwd=final.ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_frozen_inputs_sources_and_configuration_are_enforced(self):
        final.frozen(self.manifest, inputs=True)
        checks = [Path(self.manifest["data"]["test_ids"]),
                  Path(self.manifest["sources"]["conditional_imputation"]["path"]),
                  self.root / final.REFERENCES["phase25"]]
        for path in checks:
            with self.subTest(path=path):
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"modified")
                    with self.assertRaisesRegex(ValueError, "[Ff]rozen"):
                        final.frozen(self.manifest, inputs=True)
                finally:
                    path.write_bytes(original)
        changed = copy.deepcopy(self.manifest)
        changed["variants"]["phase25"]["model_parameters"]["l2_regularization"] = 1.0
        final.write_json(self.manifest_path, changed)
        with self.assertRaisesRegex(ValueError, "manifest.*changed"):
            final.read_manifest(self.manifest_path)

    def test_changed_calibration_threshold_cannot_be_published(self):
        self.local_run()
        calibration_path = self.root / "output/phase25/calibration.json"
        calibration = final.read_json(calibration_path)
        calibration["decision_threshold"] += 0.1
        final.write_json(calibration_path, calibration)
        with self.assertRaisesRegex(ValueError, "threshold or provenance"):
            final.publish_submission(self.manifest, "phase25")


if __name__ == "__main__":
    unittest.main()

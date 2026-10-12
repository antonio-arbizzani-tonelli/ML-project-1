"""Small end-to-end calibration, CSV alignment and interrupted-fit checks."""

import csv
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from tools import final_phase14_phase34 as final


class TestFinalCalibration(unittest.TestCase):
    def setUp(self):
        (final.ROOT / ".local").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=final.ROOT / ".local")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for relative in (final.HISTORICAL, "src/numpy_boosting.py", "src/feature_preprocessing.py",
                         "src/evaluation.py", "src/run_preprocessing_cv.py", "run.py", "helpers.py"):
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(final.ROOT / relative, destination)
        raw = self.root / "data/raw/dataset"
        cache = self.root / "data/processed"
        raw.mkdir(parents=True)
        cache.mkdir(parents=True)
        self.labels = np.tile(np.array([0, 0, 1], dtype=np.int8), 12)
        self.train = np.column_stack([np.arange(36) % 7, self.labels * 2]).astype(np.float32)
        self.train[::5, 0] = np.nan
        self.test = self.train[:12].copy()
        self.train_ids = np.arange(1, 37)
        self.test_ids = np.arange(101, 113)
        for split, values, ids in (("train", self.train, self.train_ids), ("test", self.test, self.test_ids)):
            np.save(cache / f"x_{split}_float32.npy", values)
            np.save(cache / f"id_{split}_int64.npy", ids)
            with (raw / f"x_{split}.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Id", "A", "B"])
                writer.writerows((int(row_id), *row) for row_id, row in zip(ids, values))
        signed = (2 * self.labels - 1).astype(np.int8)
        np.save(cache / "y_train_int8.npy", signed)
        with (raw / "y_train.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Id", "_MICHD"])
            writer.writerows(zip(self.train_ids, signed))
        final._write_submission(raw / "sample_submission.csv", self.test_ids, np.ones(12, dtype=np.int8))
        metadata = [{"name": name, "semantic_type": "continuous", "special_values": []} for name in ("A", "B")]
        final.write_json(self.root / "configs/feature_metadata.json", metadata)
        final.write_json(self.root / "configs/experiments/preprocessing.json",
                         {"variants": [{"experiment_name": "phase14", "preprocessing": {"name": "codebook_aware"}}]})
        suite = {"fold_count": 3, "seed": 19,
                 "data": {"features_path": "data/processed/x_train_float32.npy", "labels_path": "data/processed/y_train_int8.npy",
                          "x_train_csv_path": "data/raw/dataset/x_train.csv"},
                 "preprocessing_config_path": "configs/experiments/preprocessing.json", "preprocessing_variant": "phase14",
                 "feature_metadata_path": "configs/feature_metadata.json",
                 "models": [{"parameters": {"learning_rate": 0.05, "max_depth": 2, "min_samples_leaf": 1,
                                             "n_bins": 4, "l2_regularization": 1.0, "max_features": 2, "random_seed": 20},
                             "checkpoints": [50]}]}
        final.write_json(self.root / final.REFERENCE, suite)
        with mock.patch.object(final, "EXPECTED_SHAPE", (36, 12, 2)), mock.patch.object(final, "EXPECTED_FEATURES", 2):
            self.manifest = final.prepare(self.root, self.root / "output")
        self.manifest_path = self.root / "output/manifest.json"

    def test_oof_calibration_and_two_csvs_preserve_ids_and_threshold_source(self):
        def local_worker(path, variant, partition):
            return final.worker(path, variant, partition)
        with mock.patch.object(final, "_launch", side_effect=local_worker):
            comparison = final.run(self.manifest_path, workers=1)
        self.assertIsNone(comparison["official_test_f1_comparison"])
        self.assertIn("not an unbiased test score", comparison["calibration_warning"])
        loaded = final.modules(self.manifest)
        self.assertNotEqual(loaded["phase14"].__name__, loaded["phase34"].__name__)
        self.assertNotEqual(self.manifest["sources"]["phase14"]["sha256"], self.manifest["sources"]["phase34"]["sha256"])
        for variant in final.VARIANTS:
            directory = self.root / "output" / variant
            with np.load(directory / "full_train_oof.npz") as oof:
                np.testing.assert_array_equal(oof["row_indices"], np.arange(36))
                np.testing.assert_array_equal(oof["train_ids"], self.train_ids)
                choice = loaded["evaluation"].best_f1_threshold(self.labels, oof["probabilities"])
                self.assertEqual(choice.threshold, comparison["variants"][variant]["decision_threshold"])
                for fold, (train, valid) in enumerate(final.full_train_folds(self.labels, 19), start=1):
                    self.assertEqual(np.intersect1d(train, valid).size, 0)
                    np.testing.assert_array_equal(np.flatnonzero(oof["fold_ids"] == fold), valid)
            with Path(comparison["variants"][variant]["submission"]["path"]).open(newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ["Id", "Prediction"])
            np.testing.assert_array_equal(np.array([int(row[0]) for row in rows[1:]]), self.test_ids)
            predictions = np.array([int(row[1]) for row in rows[1:]])
            self.assertTrue(np.all(np.isin(predictions, [-1, 1])))
            scores = np.load(directory / "full/probabilities.npy")
            threshold = comparison["variants"][variant]["decision_threshold"]
            np.testing.assert_array_equal(predictions, np.where(scores >= threshold, 1, -1))
            self.assertIn(f'--manifest "{self.manifest_path}"', final.read_json(directory / "calibrated_final_config.json")["reproduction_command"])
        calibration_path = self.root / "output/phase14/calibration.json"
        calibration = final.read_json(calibration_path)
        calibration["decision_threshold"] += 0.1
        final.write_json(calibration_path, calibration)
        with self.assertRaisesRegex(ValueError, "threshold or provenance"):
            final.publish_submission(self.manifest, "phase14")

    def test_interrupted_checkpoint_resumes_exactly_and_loads_in_fresh_process(self):
        loaded = final.modules(self.manifest)
        model_class = loaded["phase14"].HistogramGradientBoostingClassifier
        save = model_class.save_checkpoint
        def interrupted_save(model, path, metadata=None):
            save(model, path, metadata)
            raise RuntimeError("simulated interruption")
        with mock.patch.object(model_class, "save_checkpoint", new=interrupted_save):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                final.worker(self.manifest_path, "phase14", "fold_1")
        checkpoint = self.root / "output/phase14/fold_1/model.pkl"
        partial, _ = model_class.load_checkpoint(checkpoint)
        self.assertEqual(len(partial.trees_), 25)
        final.worker(self.manifest_path, "phase14", "fold_1")
        result = final.completed(self.manifest, "phase14", "fold_1")
        self.assertEqual(result["resumed_from_trees"], 25)
        train, labels, valid, _ = final._matrices(self.manifest, "fold_1", loaded["preprocessing"])
        reference = model_class(**self.manifest["model_parameters"]).fit(train, labels)
        np.testing.assert_array_equal(np.load(checkpoint.parent / "probabilities.npy"), reference.predict_proba(valid)[:, 1])
        with self.assertRaises(ValueError):
            loaded["phase34"].HistogramGradientBoostingClassifier.load_checkpoint(checkpoint)
        source = self.manifest["sources"]["phase14"]
        code = ("import sys; from tools.final_phase14_phase34 import load_source; "
                "m=load_source(sys.argv[1],sys.argv[2],sys.argv[3]); "
                "model,metadata=m.HistogramGradientBoostingClassifier.load_checkpoint(sys.argv[4]); "
                "assert len(model.trees_)==50; assert metadata['variant']=='phase14'")
        process = subprocess.run([sys.executable, "-B", "-c", code, source["module"], source["path"], source["sha256"], str(checkpoint)],
                                 cwd=final.ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_frozen_inputs_and_official_id_order_are_enforced(self):
        final.frozen(self.manifest, inputs=True)
        np.save(self.root / "data/processed/id_test_int64.npy", self.test_ids[::-1])
        with self.assertRaisesRegex(ValueError, "frozen input"):
            final.frozen(self.manifest, inputs=True)
        suite = final.read_json(self.root / final.REFERENCE)
        with mock.patch.object(final, "EXPECTED_SHAPE", (36, 12, 2)):
            with self.assertRaisesRegex(ValueError, "Cached IDs differ"):
                final._dataset(self.root, suite)


if __name__ == "__main__":
    unittest.main()

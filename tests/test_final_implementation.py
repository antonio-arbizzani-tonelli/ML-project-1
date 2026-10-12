"""Submission entry points must select the exact evaluated numerical source."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from run import run_pipeline
from src.boosting_implementation import get_booster_class
from src.run_boosting_cv import _fit_checkpoint_probabilities


class TestFinalImplementation(unittest.TestCase):
    def test_selected_source_is_the_byte_exact_phase14_implementation(self):
        root = Path(__file__).resolve().parents[1]
        final = json.loads((root / "configs/final_model.json").read_text())
        self.assertEqual(final["implementation"], "phase14")
        source = root / "src/numpy_boosting_phase14.py"
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), final["implementation_source_sha256"])
        self.assertEqual(get_booster_class("phase14").__module__, "src.numpy_boosting_phase14")
        with self.assertRaises(ValueError):
            get_booster_class("unknown")

    def test_cv_uses_the_historical_split_search_when_explicitly_selected(self):
        features = np.r_[np.arange(40), np.repeat(np.nan, 20)].reshape(-1, 1)
        labels = np.r_[np.zeros(40), np.ones(20)]
        parameters = {"max_depth": 1, "min_samples_leaf": 5, "n_bins": 4, "random_seed": 9}
        actual, _ = _fit_checkpoint_probabilities(features, labels, features, parameters, [1], implementation="phase14")
        historical = get_booster_class("phase14")(**parameters, n_estimators=1).fit(features, labels)
        corrected = get_booster_class("phase34")(**parameters, n_estimators=1).fit(features, labels)
        np.testing.assert_array_equal(actual[1], historical.predict_proba(features)[:, 1])
        self.assertFalse(np.array_equal(actual[1], corrected.predict_proba(features)[:, 1]))

    def test_run_pipeline_selects_phase14_and_writes_its_signed_predictions(self):
        parent = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="tmp_final_implementation_", dir=parent) as directory:
            root = Path(directory)
            self.assertEqual(root.resolve().parent, parent.resolve())
            data = root / "data"
            data.mkdir()
            features = np.r_[np.arange(40), np.repeat(np.nan, 20)].reshape(-1, 1)
            labels = np.r_[np.zeros(40), np.ones(20)].astype(np.int8)
            tests = np.array([[10.0], [np.nan]], dtype=np.float32)
            for name, header, rows in (
                ("x_train.csv", ["Id", "A"], [(i, float(x)) for i, x in enumerate(features[:, 0])]),
                ("y_train.csv", ["Id", "_MICHD"], [(i, int(2*y-1)) for i, y in enumerate(labels)]),
                ("x_test.csv", ["Id", "A"], [(100, 10), (101, "")]),
                ("sample_submission.csv", ["Id", "Prediction"], [(100, -1), (101, -1)]),
            ):
                with (data / name).open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(header)
                    writer.writerows(rows)
            (root / "metadata.json").write_text(json.dumps([{"name": "A", "semantic_type": "continuous", "special_values": []}]))
            (root / "plan.json").write_text(json.dumps({"variants": [{"experiment_name": "clean", "preprocessing": {"name": "codebook_aware"}}]}))
            parameters = {"n_estimators": 1, "max_depth": 1, "min_samples_leaf": 5, "n_bins": 4, "random_seed": 9}
            config = {"implementation": "phase14", "data": {"train_features": "x_train.csv", "train_labels": "y_train.csv", "test_features": "x_test.csv", "sample_submission": "sample_submission.csv"}, "preprocessing": {"config": "plan.json", "variant": "clean", "metadata": "metadata.json"}, "model": parameters, "decision_threshold": 0.34}
            (root / "config.json").write_text(json.dumps(config))
            result = run_pipeline(root, root / "config.json", data, root / "cache", root / "submission.csv", root / "model.pkl")
            model, _ = get_booster_class("phase14").load_checkpoint(root / "model.pkl")
            self.assertEqual(model.__class__.__module__, "src.numpy_boosting_phase14")
            expected = np.where(model.predict_proba(tests)[:, 1] >= 0.34, 1, -1)
            with (root / "submission.csv").open() as handle:
                self.assertEqual(list(csv.reader(handle)), [["Id", "Prediction"], ["100", str(expected[0])], ["101", str(expected[1])]])
            self.assertEqual(result["implementation"], "phase14")


if __name__ == "__main__":
    unittest.main()

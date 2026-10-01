"""Check the experimental full-data MLP entry point on small official-style CSVs."""

import csv
import json
import unittest

import numpy as np

from run_mlp import run_pipeline
from src.run_mlp_cv import load_pipeline, predict_pipeline
from tests.test_run_mlp_cv import fixture_directory


class TestRunMLP(unittest.TestCase):
    def test_full_training_saves_frozen_preprocessing_and_signed_submission(self):
        with fixture_directory() as root:
            data = root / "data"
            data.mkdir()
            training = np.column_stack((np.linspace(-2, 2, 40), np.tile([1, 2], 20)))
            testing = np.array([[-1, 1], [1, 2], [np.nan, 3]], dtype=np.float32)
            for name, values, start in (("x_train", training, 0), ("x_test", testing, 100)):
                with (data / f"{name}.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["Id", "A", "B"])
                    for position, row in enumerate(values, start):
                        writer.writerow([position, *["" if np.isnan(v) else v for v in row]])
            with (data / "y_train.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Id", "_MICHD"])
                writer.writerows(zip(range(40), np.where(training[:, 0] > 0, 1, -1)))
            metadata = [{"name": "A", "semantic_type": "continuous"}, {"name": "B", "semantic_type": "categorical"}]
            (root / "metadata.json").write_text(json.dumps(metadata))
            (root / "clean.json").write_text(json.dumps({"variants": [{"experiment_name": "clean", "preprocessing": {}}]}))
            suite = {"experiment_prefix": "fixture", "artifact_dir": "artifacts", "data": {},
                     "feature_metadata_path": "metadata.json", "cleaning_config_path": "clean.json", "cleaning_variant": "clean",
                     "split_id": "fixture", "fold_count": 3, "seed": 19, "inner_fold_count": 2, "inner_seed": 22,
                     "preprocessing": {"early_stopping_fraction": 0.2, "early_stopping_seed": 11},
                     "models": [{"name": "tiny", "parameters": {"hidden_layer_sizes": [4, 2], "max_epochs": 2}}]}
            config = root / "suite.json"
            config.write_text(json.dumps(suite))
            output = root / "submission.csv"
            result = run_pipeline(root, config, "tiny", 0.4, data, root / "cache", root / "artifacts", output, quiet=True)
            self.assertEqual(result["training_rows"], 40)
            self.assertEqual(result["predictions"], 3)
            payload = load_pipeline(result["checkpoint"])
            self.assertEqual(payload["metadata"]["threshold"], 0.4)
            scores = predict_pipeline(payload, testing)
            with output.open(newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ["Id", "Prediction"])
            self.assertEqual([int(row[0]) for row in rows[1:]], [100, 101, 102])
            np.testing.assert_array_equal([int(row[1]) for row in rows[1:]], np.where(scores >= 0.4, 1, -1))
            self.assertNotIn(3.0, payload["preprocessor"].states_[1].categories)

    def test_requires_an_explicit_valid_threshold(self):
        with self.assertRaises(ValueError):
            run_pipeline(None, None, None, np.nan, None, None, None, None)


if __name__ == "__main__":
    unittest.main()

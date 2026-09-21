"""Integration test for the executable all-negative experiment baseline."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.run_experiment import run_config


class TestRunExperiment(unittest.TestCase):
    """Run a tiny frozen split and inspect the metrics stored in its record."""

    def test_all_negative_baseline_evaluates_only_validation_indices(self):
        config = {
            "experiment_name": "fixture all negative",
            "hypothesis": "Fixture.",
            "data": {"dataset_id": "fixture", "labels_path": "data/labels.npy"},
            "features": {"included_columns": []},
            "missing_values": {"method": "not_applicable"},
            "split": {
                "id": "fixed",
                "indices_path": "splits/fixed.npz",
                "evaluation_partition": "validation",
            },
            "model": {"name": "all_negative", "parameters": {}},
            "threshold": {"value": 0.5, "selection_method": "not_applicable"},
        }

        class FixtureSplit:
            def __enter__(self):
                return {
                    "development_indices": np.array([3, 4]),
                    "validation_indices": np.array([0, 1, 2]),
                }

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        with (
            patch("src.run_experiment._load_config", return_value=config),
            patch(
                "src.run_experiment._resolve",
                side_effect=[Path("labels.npy"), Path("fixed.npz")],
            ),
            patch(
                "src.run_experiment.np.load",
                side_effect=[np.array([1, 0, 1, 0, 0], dtype=np.int8), FixtureSplit()],
            ),
            patch("src.run_experiment._sha256_file", return_value="fixture-sha"),
            patch(
                "src.run_experiment.save_experiment", return_value=Path("record.json")
            ) as save_record,
        ):
            record_path = run_config(Path("config.json"), Path("."), Path("records"))
        outcome = save_record.call_args.args[1]
        self.assertEqual(record_path, Path("record.json"))
        self.assertEqual(
            outcome["confusion_counts"],
            {
                "true_positives": 0,
                "false_positives": 0,
                "true_negatives": 1,
                "false_negatives": 2,
            },
        )
        self.assertEqual(outcome["metrics"]["f1"], 0.0)
        self.assertAlmostEqual(outcome["metrics"]["accuracy"], 1 / 3)

    def test_learned_baselines_use_saved_development_rows_and_write_metrics(self):
        """Run both learned branches on a small frozen split and feature array."""

        labels = np.array([-1, -1, 1, 1, -1, 1], dtype=np.int8)
        features = np.array(
            [
                [0.0, np.nan],
                [0.2, 1.0],
                [0.8, 1.0],
                [1.0, 2.0],
                [0.1, 2.0],
                [0.9, np.nan],
            ],
            dtype=np.float32,
        )
        common = {
            "hypothesis": "Fixture.",
            "data": {
                "dataset_id": "fixture",
                "labels_path": "data/labels.npy",
                "features_path": "data/features.npy",
            },
            "features": {"included_columns": "fixture"},
            "missing_values": {"method": "development-median"},
            "split": {"id": "fixed", "indices_path": "splits/fixed.npz"},
            "threshold": {"value": 0.5, "selection_method": "fixed"},
        }

        class FixtureSplit:
            def __enter__(self):
                return {
                    "development_indices": np.array([0, 1, 2, 3]),
                    "validation_indices": np.array([4, 5]),
                }

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        configurations = (
            {
                **common,
                "experiment_name": "fixture ridge",
                "model": {"name": "ridge_classifier", "parameters": {"lambda": 0.1}},
                "threshold": {"value": 0.0, "selection_method": "fixed"},
            },
            {
                **common,
                "experiment_name": "fixture logistic",
                "model": {
                    "name": "logistic_classifier",
                    "parameters": {"max_iters": 3, "gamma": 0.1},
                },
            },
        )
        for config in configurations:
            with (
                patch("src.run_experiment._load_config", return_value=config),
                patch(
                    "src.run_experiment._resolve",
                    side_effect=[
                        Path("labels.npy"),
                        Path("fixed.npz"),
                        Path("features.npy"),
                    ],
                ),
                patch(
                    "src.run_experiment.np.load",
                    side_effect=[labels, FixtureSplit(), features],
                ),
                patch("src.run_experiment._sha256_file", return_value="fixture-sha"),
                patch(
                    "src.run_experiment.save_experiment",
                    return_value=Path("record.json"),
                ) as save_record,
            ):
                run_config(Path("config.json"), Path("."), Path("records"))
            outcome = save_record.call_args.args[1]
            self.assertEqual(outcome["status"], "completed")
            self.assertEqual(outcome["evaluation_partition"], "validation")
            self.assertIn("f1", outcome["metrics"])
            self.assertEqual(outcome["model_details"]["weights"], 3)
            self.assertIn("features", outcome["input_sha256"])
        self.assertIn("log_loss", outcome["metrics"])


if __name__ == "__main__":
    unittest.main()

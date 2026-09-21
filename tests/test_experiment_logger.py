"""Tests for the small, file-based experiment ledger."""

import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiment_logger import save_experiment, validate_config


def example_config():
    """Return the smallest complete configuration used by logger tests."""

    return {
        "experiment_name": "unit baseline",
        "hypothesis": "A test hypothesis.",
        "data": {"dataset_id": "fixture"},
        "features": {"included_columns": []},
        "missing_values": {"method": "not_applicable"},
        "split": {"id": "fixture_split"},
        "model": {"name": "fixture_model", "parameters": {}},
        "threshold": {"value": 0.5, "selection_method": "fixed"},
    }


class TestExperimentLogger(unittest.TestCase):
    """Verify records retain details and the index exposes comparison fields."""

    def test_save_experiment_writes_complete_json_and_index_row(self):
        outcome = {
            "status": "completed",
            "metrics": {"f1": 0.44, "accuracy": 0.91, "precision": 0.5, "recall": 0.4},
            "runtime_seconds": 1.25,
            "memory_estimate": {"arrays_mebibytes": 12.5},
        }
        output_dir = Path(__file__).parent
        with (
            patch("src.experiment_logger._write_json_atomically") as write_record,
            patch("src.experiment_logger._append_index") as append_index,
            patch.object(Path, "read_bytes", return_value=b"record"),
        ):
            record_path = save_experiment(example_config(), outcome, output_dir)
        record = write_record.call_args.args[1]
        self.assertEqual(record["config"], example_config())
        self.assertEqual(record["outcome"], outcome)
        self.assertEqual(record["schema_version"], 1)
        index_row = append_index.call_args.args[1]
        self.assertEqual(index_row["f1"], 0.44)
        self.assertEqual(index_row["split_id"], "fixture_split")
        self.assertEqual(index_row["record_path"], record_path.name)

    def test_incomplete_config_is_rejected_before_writing(self):
        invalid = example_config()
        del invalid["model"]
        with self.assertRaisesRegex(ValueError, "model"):
            validate_config(invalid)


if __name__ == "__main__":
    unittest.main()

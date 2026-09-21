"""Tests for the final training and submission entry point."""

import csv
import unittest
from pathlib import Path

import numpy as np

from run import _read_feature_csv, _read_labels_csv, _write_submission


class TestFinalRunIO(unittest.TestCase):
    def test_reads_official_feature_and_label_shapes(self):
        features_path = Path("tests/_fixture_x_train.csv")
        labels_path = Path("tests/_fixture_y_train.csv")
        try:
            with features_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows([["Id", "A", "B"], [10, 1.5, ""], [11, 2.5, 3.5]])
            with labels_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows([["Id", "_MICHD"], [10, -1], [11, 1]])

            ids, features, names = _read_feature_csv(features_path)
            label_ids, labels = _read_labels_csv(labels_path)
        finally:
            features_path.unlink(missing_ok=True)
            labels_path.unlink(missing_ok=True)

        np.testing.assert_array_equal(ids, [10, 11])
        np.testing.assert_array_equal(label_ids, ids)
        np.testing.assert_array_equal(labels, [-1, 1])
        self.assertEqual(names, ["A", "B"])
        self.assertTrue(np.isnan(features[0, 1]))

    def test_writes_submission_header_ids_and_signed_predictions(self):
        path = Path("tests/_fixture_submission.csv")
        try:
            _write_submission(path, np.array([20, 21]), np.array([-1, 1]))
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(rows, [["Id", "Prediction"], ["20", "-1"], ["21", "1"]])


if __name__ == "__main__":
    unittest.main()

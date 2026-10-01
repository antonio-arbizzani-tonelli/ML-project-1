"""Integration checks for fold boundaries, nested thresholds and persisted inference."""

import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from src.evaluation import best_f1_threshold, classification_metrics
from src.run_mlp_cv import fit_partition, load_pipeline, predict_pipeline, run_suite, stratified_holdout


@contextmanager
def fixture_directory():
    """Use inherited Windows directory permissions for a disposable test fixture."""
    results = Path("results").resolve()
    directory = results / f"test_mlp_{uuid.uuid4().hex}"
    directory.mkdir()
    try:
        yield directory
    finally:
        if directory.resolve().parent != results:
            raise RuntimeError("Fixture cleanup escaped the results directory.")
        shutil.rmtree(directory)


def load_arrays(path):
    """Close compressed fixtures before Windows cleanup."""
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files}


class TestMLPCV(unittest.TestCase):
    def test_stratified_holdout_is_disjoint_complete_and_deterministic(self):
        y = np.r_[np.zeros(90), np.ones(10)]
        first = stratified_holdout(y, 0.1, 12)
        second = stratified_holdout(y, 0.1, 12)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(np.intersect1d(*first).size, 0)
        np.testing.assert_array_equal(np.sort(np.r_[first[0], first[1]]), np.arange(100))
        self.assertEqual(int(np.sum(y[first[1]])), 1)

    def test_evaluation_labels_cannot_affect_epoch_selection_or_fit(self):
        with fixture_directory() as temporary:
            x = np.arange(40, dtype=np.float32).reshape(-1, 1) / 10
            y = np.tile([0, 1], 20)
            arguments = (x, y, np.arange(30), np.arange(30, 40), ["N"],
                         [{"name": "N", "semantic_type": "continuous"}], {},
                         {"early_stopping_fraction": 0.2},
                         [{"name": "tiny", "parameters": {"hidden_layer_sizes": [3], "max_epochs": 2, "batch_size": 16}}])
            first, _ = fit_partition(*arguments, Path(temporary), "original", 4, quiet=True)
            y[30:] = 1 - y[30:]
            second, _ = fit_partition(*arguments, Path(temporary), "changed", 4, quiet=True)
            np.testing.assert_array_equal(first["tiny"], second["tiny"])

    def test_nested_suite_recomputes_thresholds_and_checkpoints(self):
        with fixture_directory() as temporary:
            root = Path(temporary).resolve()
            rng = np.random.default_rng(3)
            x = rng.normal(size=(120, 2)).astype(np.float32)
            y = (x[:, 0] + x[:, 1] > 0).astype(np.int8)
            np.save(root / "x.npy", x)
            np.save(root / "y.npy", y * 2 - 1)
            np.savez(root / "split.npz", development_indices=np.arange(120))
            (root / "header.csv").write_text("Id,A,B\n", encoding="utf-8")
            metadata = [{"name": n, "semantic_type": "continuous"} for n in ["A", "B"]]
            (root / "metadata.json").write_text(json.dumps(metadata))
            (root / "clean.json").write_text(json.dumps({"variants": [{"experiment_name": "clean", "preprocessing": {}}]}))
            suite = {"experiment_prefix": "fixture_mlp", "artifact_dir": "artifacts",
                     "data": {"features_path": "x.npy", "labels_path": "y.npy", "split_indices_path": "split.npz", "x_train_csv_path": "header.csv"},
                     "feature_metadata_path": "metadata.json", "cleaning_config_path": "clean.json", "cleaning_variant": "clean",
                     "split_id": "fixture", "fold_count": 3, "seed": 19, "inner_fold_count": 2, "inner_seed": 22,
                     "preprocessing": {"early_stopping_fraction": 0.2, "early_stopping_seed": 11},
                     "models": [{"name": "tiny", "parameters": {"hidden_layer_sizes": [4, 2], "max_epochs": 2, "batch_size": 32}}]}
            config = root / "suite.json"
            config.write_text(json.dumps(suite))
            record = json.loads(run_suite(config, root, root / "records", quiet=True)[0].read_text())
            outcome = record["outcome"]
            oof = load_arrays(root / outcome["oof_predictions_artifact"]["path"])
            metrics = classification_metrics(y, oof["nested_predictions"])[1]
            self.assertEqual(metrics["f1"], outcome["metrics"]["f1"])
            for fold in outcome["nested_threshold_evaluation"]["folds"]:
                number = fold["outer_fold"]
                inner = load_arrays(fold["inner_oof_predictions_artifact"]["path"])
                threshold = best_f1_threshold(inner["labels"], inner["probabilities"]).threshold
                self.assertEqual(threshold, fold["inner_threshold"])
                self.assertEqual(np.intersect1d(inner["training_row_indices"], inner["outer_validation_row_indices"]).size, 0)
                boundary = load_arrays(root / "artifacts" / f"outer{number}_boundaries.npz")
                np.testing.assert_array_equal(np.sort(np.r_[boundary["gradient_fit_rows"], boundary["early_stopping_rows"]]),
                                              np.sort(boundary["training_rows"]))
                self.assertEqual(np.intersect1d(boundary["early_stopping_rows"], boundary["evaluation_rows"]).size, 0)
                payload = load_pipeline(fold["outer_fit"]["checkpoint"]["path"])
                positions = oof["fold_ids"] == number
                replay = predict_pipeline(payload, x[oof["development_indices"][positions]])
                np.testing.assert_allclose(replay, oof["probabilities"][positions], atol=1e-7, rtol=1e-6)
                self.assertEqual(payload["model"].epochs_trained_, fold["outer_fit"]["selected_epochs"])
                self.assertEqual(payload["preprocessor"].states_[0].mean, float(np.mean(x[boundary["training_rows"], 0], dtype=np.float64)))


if __name__ == "__main__":
    unittest.main()

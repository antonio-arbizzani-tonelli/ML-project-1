"""Guard the costly seed trial against confounding and score misalignment."""

import copy
import contextlib
import json
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np

from src.evaluation import best_f1_threshold
from src.experiment_logger import save_experiment
from src.run_boosting_cv import _save_inner_oof_predictions, _save_oof_predictions
from src.summarize_phase19_transfer import _arrays
from tools.phase26_two_seeds import ARTIFACTS, REFERENCE_SUITE, SUITE, aligned_mean, evaluate_mean, preflight, validate_change, verify_feature_order


@contextlib.contextmanager
def artifact_fixture_directory():
    # Normal inherited Windows permissions; Python 3.14's mode=0700 temp
    # directories exclude the restricted runner token on this machine.
    parent = (Path(__file__).resolve().parents[1] / "results").resolve()
    path = parent / f"phase26_test_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        actual = path.resolve()
        if not actual.is_relative_to(parent) or actual == parent:
            raise ValueError("Fixture cleanup must remain inside the results directory.")
        shutil.rmtree(actual)


class TestTwoSeedTrial(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.reference = json.loads((root / REFERENCE_SUITE).read_text())
        self.candidate = json.loads((root / SUITE).read_text())
        self.first = {
            "training_row_indices": np.arange(4),
            "outer_validation_row_indices": np.array([4, 5]),
            "fold_ids": np.array([1, 2, 1, 2]),
            "labels": np.array([1, 0, 0, 1]),
            "outer_fold": np.array([1]), "inner_seed": np.array([20260923]),
            "probabilities": np.array([0.7, 0.5, 0.1, 0.4]),
        }
        self.second = copy.deepcopy(self.first)
        self.second["probabilities"] = np.array([0.3, 0.8, 0.2, 0.7])

    def test_accepts_only_requested_seed_change(self):
        validate_change(self.candidate, self.reference)

    def test_accepts_legacy_names_only_for_original_component(self):
        self.assertEqual(verify_feature_order(["a", "b"], {}, ["a", "b"], 20260920),
                         "legacy_checkpoint_order_verified_from_preflight_and_predictions")
        with self.assertRaises(ValueError):
            verify_feature_order(["a", "b"], {}, ["a", "b"], 20260921)

    def test_rejects_legacy_order_different_from_frozen_preprocessing(self):
        with self.assertRaises(ValueError):
            verify_feature_order(["b", "a"], {}, ["a", "b"], 20260920)

    def test_requires_new_checkpoint_names_to_match(self):
        saved = {"output_feature_names": ["a", "b"]}
        self.assertEqual(verify_feature_order(["a", "b"], saved, ["a", "b"], 20260921),
                         "checkpoint_feature_names_match")
        with self.assertRaises(ValueError):
            verify_feature_order(["a", "b"], {"output_feature_names": ["b", "a"]}, ["a", "b"], 20260921)

    def test_preserves_preflight_after_training_starts(self):
        with artifact_fixture_directory() as root:
            archive = root / ARTIFACTS / "executed_runner.py"
            archive.parent.mkdir(parents=True)
            archive.write_text("training snapshot")
            with self.assertRaisesRegex(ValueError, "Training snapshot"):
                preflight(root)

    def test_rejects_model_or_evaluation_changes(self):
        for key, value in (("random_seed", 20260922), ("l2_regularization", 5.0), ("learning_rate", 0.025)):
            changed = copy.deepcopy(self.candidate)
            changed["models"][0]["parameters"][key] = value
            with self.subTest(parameter=key), self.assertRaises(ValueError):
                validate_change(changed, self.reference)
        changed = copy.deepcopy(self.candidate)
        changed["nested_threshold"]["inner_seed"] = 1
        with self.assertRaises(ValueError):
            validate_change(changed, self.reference)

    def test_threshold_is_selected_on_mean_scores(self):
        mean = aligned_mean(self.first, self.second)
        np.testing.assert_allclose(mean, [0.5, 0.65, 0.15, 0.55])
        selected = best_f1_threshold(self.first["labels"], mean).threshold
        individual_average = sum(best_f1_threshold(a["labels"], a["probabilities"]).threshold
                                 for a in (self.first, self.second)) / 2
        self.assertEqual(selected, 0.5)
        self.assertNotEqual(selected, individual_average)

    def test_rejects_row_label_or_fold_mismatch(self):
        for key in ("training_row_indices", "outer_validation_row_indices", "fold_ids", "labels", "inner_seed"):
            changed = copy.deepcopy(self.second)
            changed[key][0] += 1
            with self.subTest(array=key), self.assertRaises(ValueError):
                aligned_mean(self.first, changed)

    def test_rejects_outer_rows_in_inner_selection(self):
        for arrays in (self.first, self.second):
            arrays["outer_validation_row_indices"][0] = 0
        with self.assertRaises(ValueError):
            aligned_mean(self.first, self.second)

    def test_rejects_invalid_component_scores(self):
        for value in (np.nan, np.inf, -0.01, 1.01):
            changed = copy.deepcopy(self.second)
            changed["probabilities"][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                aligned_mean(self.first, changed)

    def test_saved_ensemble_uses_mean_inner_scores_and_aligned_outer_scores(self):
        """Exercise the real artifact and evaluation path on three small folds."""
        with artifact_fixture_directory() as root:
            rows = np.arange(6)
            ids = np.repeat([1, 2, 3], 2)
            labels = np.tile([1, 0], 3)
            paths = []
            outer_scores = (np.array([0.9, 0.6, 0.3, 0.2, 0.7, 0.8]),
                            np.array([0.3, 0.2, 0.9, 0.8, 0.3, 0.2]))
            inner_scores = (np.array([0.7, 0.5, 0.4, 0.1]), np.array([0.3, 0.8, 0.7, 0.2]))
            for number in range(2):
                directory = root / str(number)
                op, fingerprint = _save_oof_predictions(directory, "component", rows, ids, labels,
                                                       outer_scores[number], 0.5, np.full(6, 0.5))
                folds = []
                for fold in range(1, 4):
                    train, valid = rows[ids != fold], rows[ids == fold]
                    info = _save_inner_oof_predictions(directory, "component", fold, 20260922 + fold,
                        train, valid, np.array([1, 2, 1, 2]), labels[train], inner_scores[number])
                    info["path"] = str(Path(info["path"]).relative_to(root))
                    folds.append({"inner_oof_predictions_artifact": info})
                config = {"experiment_name": "component", "hypothesis": "fixture", "data": {},
                    "features": {}, "missing_values": {}, "split": {}, "threshold": {"value": 0.5},
                    "model": {"name": "fixture", "parameters": {"random_seed": 20260920 + number}}}
                outcome = {"nested_threshold_evaluation": {"folds": folds}, "runtime_seconds": 0.1,
                    "source_sha256": {}, "input_sha256": {}, "environment": {}, "model_checkpoints": [],
                    "oof_predictions_artifact": {"path": str(op.relative_to(root)), "sha256": fingerprint}}
                paths.append(save_experiment(config, outcome, directory / "records"))
            mean_path = evaluate_mean(root, *paths, self.candidate, root / "mean")
            record = json.loads(mean_path.read_text())
            arrays = _arrays(root, record)
            np.testing.assert_array_equal(arrays["probabilities"], sum(outer_scores) / 2)
            np.testing.assert_array_equal(arrays["nested_thresholds"], np.full(6, 0.5))
            np.testing.assert_array_equal(arrays["nested_predictions"], arrays["probabilities"] >= 0.5)
            self.assertEqual(record["config"]["model"]["parameters"]["weights"], [0.5, 0.5])
            self.assertEqual(len(record["outcome"]["nested_threshold_evaluation"]["folds"]), 3)


if __name__ == "__main__":
    unittest.main()

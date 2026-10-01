"""Check permutation equivalence, selection boundaries and frozen ranking rules."""

import contextlib
import copy
import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.numpy_boosting import HistogramGradientBoostingClassifier
from src.run_boosting_cv import _metrics, _save_oof_predictions
from src.run_preprocessing_cv import _stratified_folds
from src.summarize_phase19_transfer import BASELINE
from tools.phase29_feature_selection import (CONFIG, PermutationPredictor, array_artifact,
    checked_ranking, configuration, evaluate, importance, load_arrays, ordered_columns, score,
    selector_partition, split_usage)


@contextlib.contextmanager
def fixture_directory():
    parent = (Path(__file__).resolve().parents[1] / "results").resolve()
    path = parent / f"phase29_test_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        actual = path.resolve()
        if actual == parent or not actual.is_relative_to(parent):
            raise ValueError("Cleanup must remain in the test fixture directory.")
        shutil.rmtree(actual)


class TestFeatureSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(29)
        cls.x = rng.normal(size=(400, 5)).astype(np.float32)
        cls.x[rng.random(cls.x.shape) < 0.08] = np.nan
        cls.y = (np.nan_to_num(cls.x[:, 0]) + 0.5 * np.nan_to_num(cls.x[:, 1]) > 0.4).astype(np.int8)
        cls.model = HistogramGradientBoostingClassifier(n_estimators=20, max_depth=3,
            min_samples_leaf=5, n_bins=16, max_features=None, random_seed=29).fit(cls.x, cls.y)

    def test_cached_original_is_exact(self):
        predictor = PermutationPredictor(self.model, self.x)
        np.testing.assert_array_equal(predictor.original_probabilities, self.model.predict_proba(self.x)[:, 1])

    def test_cached_single_permutation_is_exact_including_missingness(self):
        permutation = np.random.default_rng(30).permutation(self.x.shape[0])
        predictor = PermutationPredictor(self.model, self.x)
        before = predictor.binned.copy()
        for column in range(self.x.shape[1]):
            permuted = self.x.copy()
            permuted[:, column] = permuted[permutation, column]
            np.testing.assert_array_equal(predictor.probabilities([column], permutation), self.model.predict_proba(permuted)[:, 1])
            np.testing.assert_array_equal(predictor.binned, before)

    def test_group_uses_one_row_permutation_and_is_exact(self):
        permutation = np.random.default_rng(31).permutation(self.x.shape[0])
        predictor = PermutationPredictor(self.model, self.x)
        permuted = self.x.copy()
        permuted[:, [0, 1, 3]] = self.x[permutation][:, [0, 1, 3]]
        np.testing.assert_array_equal(predictor.probabilities([0, 1, 3], permutation), self.model.predict_proba(permuted)[:, 1])

    def test_invalid_permutation_does_not_mutate_binned_features(self):
        predictor = PermutationPredictor(self.model, self.x)
        before = predictor.binned.copy()
        with self.assertRaises(ValueError):
            predictor.probabilities([0], np.zeros(self.x.shape[0], dtype=int))
        np.testing.assert_array_equal(before, predictor.binned)

    def test_importance_holds_threshold_fixed(self):
        predictor = PermutationPredictor(self.model, self.x)
        threshold = 0.3
        original, _ = score(self.y, predictor.original_probabilities, threshold)
        permutations = [np.random.default_rng(seed).permutation(self.y.size) for seed in (31, 32, 33)]
        actual = importance(predictor, [0], self.y, threshold, permutations, original)
        expected = [original["f1"] - score(self.y, predictor.probabilities([0], p), threshold)[0]["f1"] for p in permutations]
        self.assertEqual(actual["mean_f1_drop"], float(np.mean(expected)))
        self.assertGreater(actual["mean_f1_drop"], 0)

    def test_selector_fit_holdout_and_sample_have_valid_boundaries(self):
        ids = np.arange(1000, 1400)
        fit, holdout, sample = selector_partition(self.y, ids, 20260930, 5, 50)
        self.assertEqual(np.intersect1d(fit, holdout).size, 0)
        np.testing.assert_array_equal(np.sort(np.concatenate([fit, holdout])), np.arange(400))
        self.assertTrue(np.all(np.isin(sample, holdout)))
        self.assertEqual(sample.size, 50)
        self.assertLess(abs(np.mean(self.y[sample]) - np.mean(self.y[holdout])), 1 / 50)
        again = selector_partition(self.y, ids, 20260930, 5, 50)
        for a, b in zip((fit, holdout, sample), again):
            np.testing.assert_array_equal(a, b)

    def test_rejects_duplicate_selector_rows(self):
        with self.assertRaises(ValueError):
            selector_partition(self.y, np.zeros(self.y.size), 1, 5, 50)

    def test_f1_is_first_and_ap_breaks_only_exact_ties(self):
        rows = [dict(column_index=i, mean_f1_drop=f1, mean_ap_drop=ap, mean_log_loss_increase=0, split_count=0)
                for i, (f1, ap) in enumerate(((0.1, 0.01), (0.1, 0.2), (0.2, -0.5), (-0.1, 1.0)))]
        selected, ordered = ordered_columns(rows, 2)
        self.assertEqual(selected, [1, 2])
        self.assertEqual([r["column_index"] for r in ordered], [2, 1, 0, 3])

    def test_ties_use_original_order_and_smaller_masks_are_nested(self):
        rows = [dict(column_index=i, mean_f1_drop=0, mean_ap_drop=0, mean_log_loss_increase=0, split_count=0) for i in range(150)]
        selected100, _ = ordered_columns(list(reversed(rows)), 100)
        selected60, _ = ordered_columns(rows, 60)
        self.assertEqual(selected100, list(range(100)))
        self.assertEqual(selected60, list(range(60)))
        with self.assertRaises(ValueError):
            ordered_columns(rows + [rows[0]], 100)

    def test_array_artifact_roundtrip_and_detects_corruption(self):
        with fixture_directory() as root:
            arrays = {"row_ids": np.array([1, 5, 8]), "probabilities": np.array([0.3, 0.5, 0.8])}
            info = array_artifact(root, root / "test.npz", arrays)
            loaded = load_arrays(root, info)
            for key in arrays:
                np.testing.assert_array_equal(arrays[key], loaded[key])
            invalid = copy.deepcopy(info)
            invalid["array_sha256"]["row_ids"] = "incorrect"
            with self.assertRaises(ValueError):
                load_arrays(root, invalid)

    def test_cached_selection_rejects_foreign_training_before_loading_artifacts(self):
        manifest = {"frozen_sha256": {"source": "a"}, "training_row_indices_sha256": "wrong"}
        with self.assertRaises(ValueError):
            checked_ranking(Path.cwd(), manifest, np.arange(10), np.arange(10, 20), {"source": "a"})

    def test_sampling_fraction_preserves_44_and_27_candidates(self):
        root = Path(__file__).resolve().parents[1]
        config, suite, _ = configuration(root)
        self.assertEqual(config["feature_counts"], [100, 60])
        self.assertEqual(suite["models"][0]["parameters"]["max_features"], 128)
        self.assertEqual(int(np.ceil(config["candidate_max_features_fraction"] * 100)), 44)
        self.assertEqual(int(np.ceil(config["candidate_max_features_fraction"] * 60)), 27)

    def test_usage_counts_agree_with_cached_tree_dependencies(self):
        counts, shallow, dependencies = split_usage(self.model)
        self.assertTrue(np.all(shallow <= counts))
        for column in range(self.x.shape[1]):
            self.assertEqual(counts[column] == 0, all(column not in group for group in dependencies))

    def test_nested_evaluation_uses_inner_thresholds_and_preserves_row_alignment(self):
        """Exercise the actual OOF/record writer; only expensive checkpoint inference is replaced."""
        project = Path(__file__).resolve().parents[1]
        config, suite, plan = configuration(project)
        y = (np.arange(120) % 4 == 0).astype(np.int8)
        development = np.arange(y.size, dtype=np.int64)
        outer = list(_stratified_folds(y, 3, suite["seed"]))
        fold_ids = np.zeros(y.size, dtype=np.int8)
        baseline_prob = np.where(y, 0.8, 0.1)
        baseline_metrics, baseline_counts = _metrics(y, baseline_prob, 0.5)
        selections, fitted, baseline_folds = {}, {}, []
        for fold, (train, valid) in enumerate(outer, 1):
            fold_ids[valid] = fold
            label = f"outer_{fold}"
            partitions = [(label, train, valid)]
            for inner, (it, iv) in enumerate(_stratified_folds(y[train], 2, suite["nested_threshold"]["inner_seed"] + fold), 1):
                partitions.append((f"outer_{fold}_inner_{inner}", train[it], train[iv]))
            metrics, counts = _metrics(y[valid], baseline_prob[valid], 0.5)
            baseline_folds.append({"outer_fold": fold, "outer_evaluation_metrics": metrics, "outer_evaluation_confusion_counts": counts})
            for label, part_train, part_valid in partitions:
                selections[label] = {"runtime_seconds": 1.0}
                for count in (100, 60):
                    inner_fit = "inner" in label
                    probability = np.where(y[part_valid], (0.70 if inner_fit else 0.80) + fold / 100,
                                           (0.20 if inner_fit else 0.10))
                    fitted[(label, count)] = {"checkpoint": {"path": "test.pkl", "sha256": "test"},
                        "feature_names": [f"f{i}" for i in range(count)], "profile": {"fit_seconds": 1.0},
                        "runtime_seconds": 1.0, "test_arrays": {"probabilities": probability,
                        "validation_row_indices": part_valid, "training_row_indices": part_train, "labels": y[part_valid]}}
        with fixture_directory() as root:
            baseline_path = root / BASELINE
            baseline_path.parent.mkdir(parents=True)
            oof, fingerprint = _save_oof_predictions(root / "baseline", "fixture", development,
                fold_ids, y, baseline_prob, 0.5, np.full(y.size, 0.5))
            record = {"config": json.loads((project / BASELINE).read_text())["config"], "outcome": {
                "nested_threshold_evaluation": {"pooled_metrics": baseline_metrics, "pooled_confusion_counts": baseline_counts,
                                                "folds": baseline_folds},
                "oof_predictions_artifact": {"path": oof.relative_to(root).as_posix(), "sha256": fingerprint}}}
            baseline_path.write_text(json.dumps(record), encoding="utf-8")
            with patch("tools.phase29_feature_selection.verify_candidate", side_effect=lambda *args: args[1]["test_arrays"]) as verifier:
                results = evaluate(root, config, suite, plan, {"frozen_sha256": {}}, np.zeros((y.size, 100)),
                                  y, development, [f"f{i}" for i in range(100)], [], selections, fitted)
            self.assertEqual(verifier.call_count, 18)
            for result in results:
                self.assertEqual(result["nested"]["pooled_metrics"]["f1"], 1.0)
                for fold, detail in enumerate(result["nested"]["folds"], 1):
                    self.assertEqual(detail["inner_threshold"], 0.70 + fold / 100)
                    info = detail["inner_oof_predictions_artifact"]
                    with np.load(root / info["path"]) as inner:
                        np.testing.assert_array_equal(inner["training_row_indices"], development[outer[fold-1][0]])
                        self.assertEqual(np.intersect1d(inner["training_row_indices"], inner["outer_validation_row_indices"]).size, 0)
                saved = json.loads((root / result["record"]).read_text())
                with np.load(root / saved["outcome"]["oof_predictions_artifact"]["path"]) as oof:
                    np.testing.assert_array_equal(oof["development_indices"], development)
                    np.testing.assert_array_equal(oof["fold_ids"], fold_ids)
                    np.testing.assert_array_equal(oof["nested_predictions"], y)
                    for fold in (1, 2, 3):
                        np.testing.assert_array_equal(oof["nested_thresholds"][fold_ids == fold], np.full(np.sum(fold_ids == fold), 0.70 + fold / 100))


if __name__ == "__main__":
    unittest.main()

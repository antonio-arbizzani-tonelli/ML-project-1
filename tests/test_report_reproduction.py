"""Regression checks for fresh report runs, scientific boundaries and receipts."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

import numpy as np

from src.feature_preprocessing import FeaturePreprocessor
from tools.report_reproduction import HistoricalMedianPreprocessor, ReportRun, fingerprint, sampling_positions, split_rows, write
from tools.reproduce_report import checked_metrics, selected_experiments


@contextmanager
def fixture():
    parent = Path(__file__).resolve().parents[1] / "results"
    directory = parent / ("test_report_" + uuid.uuid4().hex)
    directory.mkdir()
    try:
        yield directory
    finally:
        if directory.resolve().parent != parent.resolve():
            raise RuntimeError("Test cleanup escaped its directory.")
        shutil.rmtree(directory)


class ReportReproductionTests(unittest.TestCase):
    def test_historical_categories_use_median_and_leave_default_unchanged(self):
        metadata = [{"name": "C", "semantic_type": "categorical"}]
        values = np.asarray([[1], [1], [2], [3], [3], [np.nan]])
        plan = {"name": "codebook_aware"}
        modern = FeaturePreprocessor(["C"], metadata, plan).fit(values)
        historical = HistoricalMedianPreprocessor(["C"], metadata, plan).fit(values)
        self.assertEqual(modern._states[0].fill_value, 1)
        self.assertEqual(historical._states[0].fill_value, 2)
        imputed = np.asarray([1, 1, 2, 3, 3, 2])
        np.testing.assert_allclose(historical.transform(values)[:, 1], (imputed-imputed.mean())/imputed.std())
        self.assertEqual(FeaturePreprocessor(["C"], metadata, plan).fit(values)._states[0].fill_value, 1)

    def test_sampling_preserves_the_original_choice_and_hybrid_rules(self):
        from tools.phase36_undersampling import select_positions
        labels = np.r_[np.ones(10), np.zeros(100)].astype(np.int8)
        scores = np.linspace(0, 1, len(labels))
        for method in ("random", "hybrid_hard"):
            actual = sampling_positions(labels, method, 3, 17, scores)
            np.testing.assert_array_equal(actual, select_positions(labels, method, 17, scores))
            self.assertEqual(int(labels[actual].sum()), 10)

    def test_sampling_ratios_share_the_historical_permutation_prefix(self):
        y = np.r_[np.ones(10), np.zeros(100)].astype(np.int8)
        five = sampling_positions(y, "random_permutation_prefix", 5, 42)
        seven = sampling_positions(y, "random_permutation_prefix", 7, 42)
        expected = np.sort(np.r_[np.arange(10), np.random.default_rng(42).permutation(np.arange(10, 110))[:50]])
        np.testing.assert_array_equal(five, expected)
        self.assertTrue(set(five) <= set(seven))

    def test_split_covers_rows_once_in_signed_class_order(self):
        y = np.tile(np.asarray([-1, 1], dtype=np.int8), 30)
        train, valid = split_rows(y)
        np.testing.assert_array_equal(np.sort(np.r_[train, valid]), np.arange(60))
        self.assertEqual(np.intersect1d(train, valid).size, 0)
        self.assertEqual(len(valid), 12)
        self.assertEqual(int(np.sum(y[valid] == 1)), 6)

    def test_full_mode_never_excludes_sparse_columns(self):
        run = object.__new__(ReportRun)
        run.smoke = False
        plan = {"name": "codebook_aware", "exclude_features": ["A"]}
        self.assertIs(run.smoke_plan(plan), plan)
        self.assertEqual(run.adjusted_parameters({"n_estimators": 200, "min_samples_leaf": 200}),
                         {"n_estimators": 200, "min_samples_leaf": 200})

    def test_fit_cache_rejects_changed_probabilities(self):
        from src.run_preprocessing_cv import _sha256
        with fixture() as directory:
            run = object.__new__(ReportRun)
            run.output, run.run_id, run.smoke = directory, "fixture", False
            x = np.arange(120, dtype=np.float32).reshape(40, 3)
            y = np.tile([0, 1], 20)
            parameters = {"n_estimators": 3, "max_depth": 2, "min_samples_leaf": 2, "max_features": 2, "random_seed": 1}
            first, _ = run.fit_model(x[:30], y[:30], x[30:], parameters, "phase14", {"rows": "first"})
            second, _ = run.fit_model(x[:30], y[:30], x[30:], parameters, "phase14", {"rows": "first"})
            np.testing.assert_array_equal(first, second)
            artifact = next(directory.glob("fits/*/probabilities.npy"))
            np.save(artifact, first + 0.01)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                run.fit_model(x[:30], y[:30], x[30:], parameters, "phase14", {"rows": "first"})

    def test_nested_fit_and_selector_seeds_preserve_both_original_orders(self):
        with fixture() as directory:
            run = object.__new__(ReportRun)
            run.output, run.development = directory, np.arange(90)
            run.y = np.tile([0, 1], 45).astype(np.int8)
            run.suite = lambda path: ({"seed": 20260919, "nested_threshold": {"inner_seed": 20260922}}, {})
            calls = []
            def fit(suite, plan, train, valid, number, selector_index, entry):
                self.assertEqual(np.intersect1d(train, valid).size, 0)
                calls.append((number, selector_index))
                # Valid probabilities intentionally exceed typical F1 thresholds;
                # shifted scores would make binary log loss reject this fixture.
                return np.full(len(valid), 0.9)
            run.tree_fit = fit
            result = run.nested_tree({"id": "fixture", "recipe": "unused"})
            self.assertEqual(calls, [(1, 0), (4, 1), (5, 2), (2, 3), (6, 4), (7, 5), (3, 6), (8, 7), (9, 8)])
            self.assertTrue(np.isfinite(result["metrics"]["log_loss"]))

    def test_ensemble_averages_probabilities_before_threshold_selection(self):
        run = object.__new__(ReportRun)
        run.y = np.tile([0, 1], 20)
        run.matrix_pair = lambda *a: (np.ones((20, 2)), np.ones((20, 2)), ["A", "B"])
        def fit(train, labels, valid, parameters, backend, identity):
            return np.full(len(valid), 0.2 if parameters["random_seed"] == 1 else 0.8), None
        run.fit_model = fit
        suite = {"models": [{"parameters": {}, "checkpoints": [200]}]}
        predicted = run.tree_fit(suite, {}, np.arange(20), np.arange(20, 40), 1, 0, {"ensemble": [1, 2]})
        np.testing.assert_array_equal(predicted, np.full(20, 0.5))

    def test_catalog_has_every_target_and_distinguishes_pdf_scope(self):
        root = Path(__file__).resolve().parents[1]
        catalog = json.loads((root / "configs/report_experiments.json").read_text())
        self.assertEqual(len(catalog["experiments"]), 45)
        self.assertEqual(len(selected_experiments(catalog, ["report"])), 39)
        for entry in catalog["experiments"]:
            self.assertTrue((root / entry["recipe"]).is_file())
            self.assertTrue(entry["expected"])
            self.assertIn("f1", entry["expected"])
            self.assertTrue((root / entry["historical_record"]).is_file())
        self.assertEqual(len([e for e in catalog["experiments"] if e["kind"] == "calibration"]), 18)

    def test_checker_recomputes_metrics_and_rejects_a_forged_success(self):
        from src.run_boosting_cv import _metrics
        from src.run_preprocessing_cv import _sha256
        with fixture() as directory:
            y = np.tile([0, 1], 15).astype(np.int8)
            p = np.linspace(0.01, 0.99, 30)
            np.savez_compressed(directory / "oof.npz", rows=np.arange(30), labels=y, probabilities=p,
                                thresholds=np.full(30, 0.5), folds=np.repeat([1, 2, 3], 10))
            result = {"experiment": "fixture", "oof": "oof.npz", "oof_sha256": _sha256(directory / "oof.npz"),
                      "metrics": _metrics(y, p, 0.5)[0], "comparison": {"status": "matched"}}
            checked_metrics(directory, result)
            result["metrics"]["f1"] = 1.0
            with self.assertRaisesRegex(ValueError, "Stored metric"):
                checked_metrics(directory, result)


if __name__ == "__main__":
    unittest.main()

"""Compare resumed parallel CV with a complete standard synthetic CV run."""

import copy
import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from src import run_boosting_cv as runner
from tools import phase33_parallel_resume as resume


@contextmanager
def _temporary_directory():
    parent = Path(__file__).resolve().parents[1] / "results"
    directory = parent / f"test_parallel_resume_{uuid.uuid4().hex}"
    directory.mkdir()
    try:
        yield directory
    finally:
        if not directory.resolve().is_relative_to(parent.resolve()):
            raise ValueError("Fixture cleanup escaped the test workspace.")
        shutil.rmtree(directory)


def _fixture(root):
    rng = np.random.default_rng(77)
    x = rng.normal(size=(90, 5)).astype(np.float32)
    x[::7, 1] = np.nan
    x[:, 3] = rng.choice([1, 2, 7, 9], size=90)
    x[:, 4] = rng.choice([0, 5, 77, 88, 99], size=90)
    labels = np.where(x[:, 0] + rng.normal(scale=0.5, size=90) > 0, 1, -1).astype(np.int8)
    np.save(root / "features.npy", x)
    np.save(root / "labels.npy", labels)
    np.savez(root / "split.npz", development_indices=rng.permutation(90)[:72])
    names = ["ID", "DATE", "DESIGN", "ANSWER", "DAYS"]
    (root / "features.csv").write_text("Id," + ",".join(names) + "\n", encoding="utf-8")
    metadata = [{"name": name, "semantic_type": kind} for name, kind in zip(names,
        ("identifier", "date", "survey_design", "binary", "count"))]
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "plan.json").write_text(json.dumps({"variants": [{
        "experiment_name": "raw", "preprocessing": {"name": "raw_tree_passthrough"}}]}), encoding="utf-8")
    suite = {"experiment_prefix": "tiny_raw", "artifact_dir": "artifacts",
        "data": {"features_path": "features.npy", "labels_path": "labels.npy",
                 "split_indices_path": "split.npz", "x_train_csv_path": "features.csv"},
        "feature_metadata_path": "metadata.json", "preprocessing_config_path": "plan.json",
        "preprocessing_variant": "raw", "split_id": "tiny", "fold_count": 3, "seed": 19,
        "fixed_threshold": 0.5,
        "nested_threshold": {"inner_fold_count": 2, "inner_seed": 22,
            "model_checkpoints": [{"model": "tiny", "checkpoints": [5]}]},
        "models": [{"name": "tiny", "checkpoints": [5], "parameters": {
            "learning_rate": 0.1, "max_depth": 2, "min_samples_leaf": 2,
            "n_bins": 8, "max_features": 3, "random_seed": 20}}]}
    path = root / "suite.json"
    path.write_text(json.dumps(suite), encoding="utf-8")
    return path


def _arrays(root, record):
    with np.load(root / record["outcome"]["oof_predictions_artifact"]["path"]) as data:
        return {key: data[key].copy() for key in data.files}


class TestParallelResume(unittest.TestCase):
    def test_cached_outer_and_parallel_inner_fits_exactly_match_standard_cv(self):
        with _temporary_directory() as temporary:
            root = Path(temporary).resolve()
            suite = _fixture(root)
            reference_path = runner.run_suite(suite, root, root / "reference_records")[0]
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            expected_arrays = _arrays(root, reference)
            # Keep outer 1; remove two disposable fixture checkpoints so both
            # newly fitted outer and inner models exercise the parallel path.
            for fold in (2, 3):
                (root / "artifacts/model_checkpoints" / f"tiny_raw_tiny_fold{fold}_5trees.pkl").unlink()
            manifest = resume.prepare(root, suite, root / "cache")
            self.assertEqual(len(manifest["tasks"]), 9)
            self.assertEqual(sum(resume._cached(manifest, task) is not None
                                 for task in manifest["tasks"]), 1)
            missing = resume.train_missing(manifest, workers=2)
            self.assertEqual(len(missing), 8)
            resumed_path = resume.assemble(manifest, root / "resumed_records", workers=2)[0]
            resumed = json.loads(resumed_path.read_text(encoding="utf-8"))
            actual_arrays = _arrays(root, resumed)
            for key, values in expected_arrays.items():
                np.testing.assert_array_equal(actual_arrays[key], values, err_msg=key)
            for key in ("metrics", "confusion_counts", "fixed_threshold_metrics"):
                self.assertEqual(reference["outcome"][key], resumed["outcome"][key])
            before = reference["outcome"]["nested_threshold_evaluation"]
            after = resumed["outcome"]["nested_threshold_evaluation"]
            self.assertEqual(before["pooled_metrics"], after["pooled_metrics"])
            for first, second in zip(before["folds"], after["folds"]):
                for key in ("inner_threshold", "inner_selection_metrics", "outer_evaluation_metrics"):
                    self.assertEqual(first[key], second[key])
            self.assertEqual(len(resumed["outcome"]["model_checkpoints"]), 3)
            self.assertEqual(len(resumed["outcome"]["parallel_resume"]["fit_caches"]), 9)
            self.assertEqual(resume.train_missing(manifest, workers=2), [])

            changed = copy.deepcopy(manifest)
            changed["tasks"][0]["signature"]["train_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "partition/order/parameters"):
                resume.assemble(changed, root / "invalid_records")
            task = manifest["tasks"][0]
            _, _, probabilities, _ = resume._cache_files(manifest, task)
            scores = np.load(probabilities)
            scores[0] = 0.123
            np.save(probabilities, scores)
            with self.assertRaisesRegex(ValueError, "Invalid completed cache"):
                resume._cached(manifest, task)


if __name__ == "__main__":
    unittest.main()

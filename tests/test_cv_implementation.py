"""Explicit historical CV selects and records a backend without changing recipes."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.boosting_implementation import get_booster_class
from src.run_boosting_cv import main, run_suite


class TestCVImplementation(unittest.TestCase):
    def test_default_and_explicit_backends_record_source_and_isolate_checkpoints(self):
        with tempfile.TemporaryDirectory(prefix="tmp_cv_implementation_", dir="tests") as temporary:
            root = Path(temporary).resolve()
            x = np.zeros((90, 321), dtype=np.float32)
            x[:, 0] = np.r_[np.arange(60), np.repeat(np.nan, 30)]
            names = ["A" + str(index) for index in range(321)]
            y = np.r_[-np.ones(60), np.ones(30)].astype(np.int8)
            np.save(root / "x.npy", x)
            np.save(root / "y.npy", y)
            np.savez(root / "split.npz", development_indices=np.arange(90))
            (root / "x.csv").write_text("Id," + ",".join(names) + "\n", encoding="utf-8")
            (root / "metadata.json").write_text(json.dumps([
                {"name": name, "semantic_type": "continuous", "special_values": []}
                for name in names
            ]), encoding="utf-8")
            (root / "preprocessing.json").write_text(json.dumps({"variants": [{
                "experiment_name": "clean", "preprocessing": {"name": "codebook_aware"}
            }]}), encoding="utf-8")
            suite = {
                "data": {"features_path": "x.npy", "labels_path": "y.npy",
                         "split_indices_path": "split.npz", "x_train_csv_path": "x.csv"},
                "feature_metadata_path": "metadata.json",
                "preprocessing_config_path": "preprocessing.json",
                "preprocessing_variant": "clean", "artifact_dir": "artifacts",
                "split_id": "synthetic", "fold_count": 3, "seed": 4,
                "fixed_threshold": 0.5,
                "models": [{"name": "tiny", "parameters": {
                    "max_depth": 1, "min_samples_leaf": 3, "n_bins": 4,
                    "random_seed": 9}, "checkpoints": [2]}],
            }
            config = root / "suite.json"
            config.write_text(json.dumps(suite), encoding="utf-8")
            original = config.read_bytes()
            for requested in (None, "phase14", "phase34"):
                with self.subTest(implementation=requested), contextlib.redirect_stdout(io.StringIO()):
                    records = run_suite(config, root, root / ("records_" + str(requested)), requested)
                    result = json.loads(records[0].read_text(encoding="utf-8"))
                    selected = requested or "phase34"
                    expected_source = Path("src") / (
                        "numpy_boosting_phase14.py" if selected == "phase14" else "numpy_boosting.py")
                    expected_hash = hashlib.sha256(expected_source.read_bytes()).hexdigest()
                    self.assertEqual(result["config"]["model"]["implementation"], selected)
                    self.assertEqual(result["config"]["model"]["implementation_source_sha256"], expected_hash)
                    for artifact in result["outcome"]["model_checkpoints"]:
                        path = root / artifact["path"]
                        model, metadata = get_booster_class(selected).load_checkpoint(path)
                        self.assertEqual(metadata["implementation"], selected)
                        self.assertEqual(metadata["implementation_source_sha256"], expected_hash)
                        if requested:
                            self.assertIn("reproduction_" + requested, path.parts)
                        else:
                            self.assertNotIn("reproduction_phase34", path.parts)
                        self.assertEqual(model.__class__.__module__, "src." + expected_source.stem)
                    self.assertEqual(config.read_bytes(), original)

    def test_unknown_override_is_rejected_before_loading_data(self):
        with tempfile.TemporaryDirectory(prefix="tmp_cv_backend_", dir="tests") as temporary:
            config = Path(temporary) / "suite.json"
            config.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "implementation"):
                run_suite(config, Path(temporary), Path(temporary), "unknown")

    def test_cli_forwards_an_explicit_backend(self):
        with patch("sys.argv", ["run_boosting_cv", "recipe.json", "--implementation", "phase14"]), \
                patch("src.run_boosting_cv.run_suite", return_value=[]) as runner:
            main()
        self.assertEqual(runner.call_args.args[-1], "phase14")


if __name__ == "__main__":
    unittest.main()

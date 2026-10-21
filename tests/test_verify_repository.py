"""Check that final validation follows its selected evidence and configuration."""

import hashlib
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

from tools.verify_repository import (
    BASELINE, BASELINE_SUITE, INTEGRITY_RECEIPT, PIPELINE_FILES,
    verify, verify_pipeline_integrity,
)


class TestSelectedValidation(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / ("tmp_verify_repository_" + uuid4().hex)
        self.root.mkdir()
        self.files = set()
        for name in ("README.md", "run.py", "helpers.py", "implementations.py", "docs/FINAL_REPORT.md",
                     "docs/FINAL_AUDIT.md", "docs/ARCHIVE.md", "docs/FINAL_REPORT.pdf",
                     "docs/FINAL_REPORT.tex", ".github/workflows/tests.yml",
                     "official_material/project1/brfss_2015_codebook.txt",
                     "results/eda/analysis/splits/stratified_seed_20260918.npz"):
            self._write(name, "")
        self._write("requirements.txt", "numpy==1.23.1\n")
        self.preprocessing = {"config": "configs/cleaning.json", "variant": "clean",
                              "metadata": "configs/feature_metadata.json"}
        self._write(self.preprocessing["config"], {
            "variants": [{"experiment_name": "clean", "preprocessing": {}}]})
        self._write(self.preprocessing["metadata"], {})
        self.parameters = {"max_depth": 5, "n_estimators": 200}
        self.suite = {"models": [{"parameters": {"max_depth": 5}, "checkpoints": [200]}],
                      "preprocessing_config_path": self.preprocessing["config"],
                      "preprocessing_variant": "clean",
                      "feature_metadata_path": self.preprocessing["metadata"]}
        self.final = {"model": self.parameters.copy(), "preprocessing": self.preprocessing.copy(),
                      "decision_threshold": 0.2}
        self._write(BASELINE_SUITE, self.suite)
        self._write(BASELINE, self._record(0.2, 0.44))
        self._write("configs/final_model.json", self.final)

    def tearDown(self):
        if self.root.resolve().parent != Path(__file__).resolve().parent:
            raise ValueError("Fixture cleanup escaped the tests directory.")
        shutil.rmtree(self.root)

    def _write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) if isinstance(value, dict) else value, encoding="utf-8")
        self.files.add(name)

    def _record(self, threshold, f1):
        return {"config": {"model": {"parameters": self.parameters.copy()},
                           "features": {"preprocessing_source": self.preprocessing["config"],
                                        "preprocessing_variant": self.preprocessing["variant"]}},
                "outcome": {"nested_threshold_evaluation": {
                    "folds": [{"inner_threshold": threshold}] * 3,
                    "pooled_metrics": {"f1": f1}}}}

    def _select_phase34(self):
        self.final["validation"] = {"name": "Phase 34", "suite": "configs/phase34.json",
                                    "record": "results/experiments/phase34/record.json"}
        self.final["decision_threshold"] = 0.3
        self._write(self.final["validation"]["suite"], self.suite)
        self._write(self.final["validation"]["record"], self._record(0.3, 0.45))
        self._write("configs/final_model.json", self.final)

    def _verify(self):
        with patch("tools.verify_repository.published_files", return_value=self.files), \
                patch("tools.verify_repository.verify_pipeline_integrity", return_value=([], 0)):
            return verify(self.root)

    def _select_full_train_calibration(self):
        source = "src/numpy_boosting_phase14.py"
        self._write(source, '"""Frozen numerical fixture."""\n')
        fingerprint = hashlib.sha256((self.root / source).read_bytes()).hexdigest()
        self.suite["implementation"] = "phase14"
        self._write(BASELINE_SUITE, self.suite)
        self.final.update({"implementation": "phase14",
                           "implementation_source_sha256": fingerprint,
                           "decision_threshold": 0.25,
                           "calibration": {"record": "results/eda/calibration.json"}})
        self.calibration = {"model": self.parameters.copy(), "implementation": "phase14",
                            "implementation_sha256": fingerprint, "training_rows": 30,
                            "threshold": {"value": 0.25, "method": "pooled_3_fold_full_train_oof_f1",
                                          "training_rows": 30, "fold_count": 3}}
        self._write(self.final["calibration"]["record"], self.calibration)
        self._write("configs/final_model.json", self.final)

    def test_full_train_calibration_can_replace_development_mean_threshold(self):
        self._select_full_train_calibration()
        result = self._verify()
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["submission_threshold"], 0.25)
        self.assertEqual(result["nested_development_f1"], 0.44)

    def _select_calibration_only(self):
        self._select_full_train_calibration()
        self.final["validation"] = {
            "name": "Combined full-train candidate", "type": "full_train_oof_calibration",
            "suite": BASELINE_SUITE, "record": self.final["calibration"]["record"],
        }
        self.calibration["preprocessing"] = self.preprocessing.copy()
        self._write(self.final["calibration"]["record"], self.calibration)
        self._write("configs/final_model.json", self.final)

    def test_calibration_only_candidate_has_no_invented_nested_score(self):
        self._select_calibration_only()
        result = self._verify()
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["validation_type"], "full_train_oof_calibration")
        self.assertIsNone(result["nested_development_f1"])
        self.assertEqual(result["submission_threshold"], 0.25)

    def test_calibration_only_candidate_rejects_parameter_and_preprocessing_mismatch(self):
        self._select_calibration_only()
        self.calibration["model"]["max_depth"] = 7
        self.calibration["preprocessing"]["variant"] = "other"
        self._write(self.final["calibration"]["record"], self.calibration)
        result = self._verify()
        self.assertFalse(result["passed"])
        self.assertIn("Full-train calibration and final model parameters differ.", result["errors"])
        self.assertIn("Full-train calibration and final preprocessing differ.", result["errors"])

    def test_calibration_only_candidate_rejects_different_record_references(self):
        self._select_calibration_only()
        self._write("results/eda/other_calibration.json", self.calibration)
        self.final["validation"]["record"] = "results/eda/other_calibration.json"
        self._write("configs/final_model.json", self.final)
        self.assertIn("Selected validation and full-train calibration records differ.",
                      self._verify()["errors"])

    def test_calibration_only_candidate_requires_a_calibration_reference(self):
        self._select_calibration_only()
        del self.final["calibration"]
        self._write("configs/final_model.json", self.final)
        self.assertIn("Full-train validation requires a calibration record.", self._verify()["errors"])

    def test_unknown_validation_evidence_type_is_rejected(self):
        self.final["validation"] = {
            "name": "Unknown", "type": "unknown", "suite": BASELINE_SUITE, "record": BASELINE,
        }
        self._write("configs/final_model.json", self.final)
        self.assertIn("Unknown selected validation evidence type.", self._verify()["errors"])

    def test_final_threshold_must_match_published_full_train_calibration(self):
        self._select_full_train_calibration()
        self.final["decision_threshold"] = 0.3
        self._write("configs/final_model.json", self.final)
        result = self._verify()
        self.assertFalse(result["passed"])
        self.assertIn("Final threshold differs from its full-train OOF calibration record.",
                      result["errors"])
        self.files.remove(self.final["calibration"]["record"])
        self.assertIn("Missing published full-train calibration record: results/eda/calibration.json",
                      self._verify()["errors"])

    def test_legacy_config_keeps_phase14_fallback(self):
        result = self._verify()
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["selected_model"], "Phase 14")
        self.assertEqual(result["nested_development_f1"], 0.44)

    def test_nonselected_recipe_requires_published_preprocessing_dependency(self):
        recipe = "configs/experiments/comparison.json"
        dependency = "configs/experiments/comparison_preprocessing.json"
        self._write(recipe, {"preprocessing_config_path": dependency})
        expected = f"Missing published configuration dependency: {recipe}: {dependency}"
        self.assertIn(expected, self._verify()["errors"])
        self._write(dependency, {})
        self.files.remove(dependency)
        self.assertIn(expected, self._verify()["errors"])

    def test_nested_configuration_dependencies_are_checked_transitively(self):
        recipe = "configs/experiments/comparison.json"
        dependency = "configs/experiments/comparison_preprocessing.json"
        metadata = "configs/comparison_metadata.json"
        self._write(recipe, {"variants": [{"preprocessing_config_path": dependency}]})
        self._write(dependency, {"feature_metadata_path": metadata})
        expected = f"Missing published configuration dependency: {dependency}: {metadata}"
        self.assertIn(expected, self._verify()["errors"])
        self._write(metadata, {})
        result = self._verify()
        self.assertTrue(result["passed"], result["errors"])

    def test_selected_evidence_replaces_legacy_metrics_and_threshold(self):
        self._select_phase34()
        result = self._verify()
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["selected_model"], "Phase 34")
        self.assertEqual(result["nested_development_f1"], 0.45)
        self.assertEqual(result["submission_threshold"], 0.3)

    def test_selected_reference_must_exist_and_be_published(self):
        self._select_phase34()
        record = self.final["validation"]["record"]
        self.files.remove(record)
        result = self._verify()
        self.assertFalse(result["passed"])
        self.assertIn(f"Missing published validation record: {record}", result["errors"])
        self.assertIsNone(result["nested_development_f1"])
        (self.root / record).unlink()
        self.files.add(record)
        self.assertFalse(self._verify()["passed"])

    def test_selected_suite_and_record_reject_mismatched_parameters_and_threshold(self):
        self._select_phase34()
        self.final["model"]["max_depth"] = 7
        self.final["decision_threshold"] = 0.2
        self.final["preprocessing"]["metadata"] = "configs/other_metadata.json"
        self._write("configs/other_metadata.json", {})
        self._write("configs/final_model.json", self.final)
        result = self._verify()
        self.assertFalse(result["passed"])
        self.assertIn("Final model parameters differ from the selected validation suite.", result["errors"])
        self.assertIn("Final model parameters differ from the selected validation record.", result["errors"])
        self.assertIn("Final threshold differs from the mean of selected validation inner thresholds.", result["errors"])
        self.assertIn("Final preprocessing metadata differs from the selected validation suite.", result["errors"])

    def test_selected_record_cannot_mix_a_different_preprocessing_with_the_suite(self):
        self._select_phase34()
        record = self._record(0.3, 0.45)
        record["config"]["features"]["preprocessing_source"] = "configs/raw.json"
        self._write(self.final["validation"]["record"], record)
        result = self._verify()
        self.assertFalse(result["passed"])
        self.assertIn("Final preprocessing config differs from the selected validation record.", result["errors"])


class TestPipelineIntegrity(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / ("tmp_integrity_" + uuid4().hex)
        self.root.mkdir()
        self.files = {INTEGRITY_RECEIPT, *PIPELINE_FILES}
        self.receipt = {"schema_version": 1, "files": {}}
        for name in PIPELINE_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            content = ("Frozen fixture: " + name + "\n").encode()
            path.write_bytes(content)
            self.receipt["files"][name] = {
                "sha256": hashlib.sha256(content).hexdigest(), "newline_policy": "preserve"}
        self._save_receipt()

    def tearDown(self):
        if self.root.resolve().parent != Path(__file__).resolve().parent:
            raise ValueError("Fixture cleanup escaped the tests directory.")
        shutil.rmtree(self.root)

    def _save_receipt(self):
        path = self.root / INTEGRITY_RECEIPT
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.receipt), encoding="utf-8")

    def test_checks_all_required_delivery_fingerprints(self):
        errors, count = verify_pipeline_integrity(self.root, self.files)
        self.assertEqual(errors, [])
        self.assertEqual(count, len(PIPELINE_FILES))

    def test_modified_metadata_or_preprocessing_is_rejected(self):
        for name in ("configs/feature_metadata.json", "src/feature_preprocessing.py"):
            with self.subTest(file=name):
                path = self.root / name
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                errors, _ = verify_pipeline_integrity(self.root, self.files)
                self.assertIn("Pipeline fingerprint mismatch: " + name, errors)
                path.write_bytes(original)

    def test_only_explicit_git_normalized_text_accepts_crlf(self):
        normalized = "configs/final_model.json"
        self.receipt["files"][normalized]["newline_policy"] = "lf"
        self._save_receipt()
        for name in (normalized, "src/numpy_boosting_phase14.py"):
            path = self.root / name
            path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        errors, _ = verify_pipeline_integrity(self.root, self.files)
        self.assertNotIn("Pipeline fingerprint mismatch: " + normalized, errors)
        self.assertIn("Pipeline fingerprint mismatch: src/numpy_boosting_phase14.py", errors)

    def test_missing_fingerprint_or_unpublished_file_is_rejected(self):
        del self.receipt["files"]["configs/feature_metadata.json"]
        self._save_receipt()
        self.files.remove("run.py")
        errors, _ = verify_pipeline_integrity(self.root, self.files)
        self.assertIn("Missing pipeline fingerprint: configs/feature_metadata.json", errors)
        self.assertIn("Missing fingerprinted pipeline file: run.py", errors)

    def test_missing_receipt_and_outside_path_are_rejected(self):
        errors, _ = verify_pipeline_integrity(self.root, self.files - {INTEGRITY_RECEIPT})
        self.assertIn("Missing pipeline integrity receipt: " + INTEGRITY_RECEIPT, errors)
        self.receipt["files"]["../outside.py"] = {"sha256": "", "newline_policy": "preserve"}
        self._save_receipt()
        errors, _ = verify_pipeline_integrity(self.root, self.files)
        self.assertIn("Pipeline fingerprint escapes the repository: ../outside.py", errors)
        self.receipt["files"]["run.py"]["newline_policy"] = "unknown"
        self._save_receipt()
        errors, _ = verify_pipeline_integrity(self.root, self.files)
        self.assertIn("Invalid pipeline fingerprint policy: run.py", errors)


if __name__ == "__main__":
    unittest.main()

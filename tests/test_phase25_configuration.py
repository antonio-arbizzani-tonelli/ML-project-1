"""Reject unintended confounders in the L2-only trial."""

import copy
import json
import unittest
from pathlib import Path

from tools.phase25_l2_trial import SUITE, validate_change


class TestL2OnlyConfiguration(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.reference = json.loads((root / "configs/experiments/phase14_boosting_codebook_corrections_only.json").read_text())
        self.candidate = json.loads((root / SUITE).read_text())

    def test_accepts_requested_change(self):
        validate_change(self.candidate, self.reference)

    def test_rejects_additional_parameter_change(self):
        self.candidate["models"][0]["parameters"]["min_samples_leaf"] = 150
        with self.assertRaises(ValueError):
            validate_change(self.candidate, self.reference)

    def test_rejects_changed_preprocessing_seed_or_threshold_protocol(self):
        for key, value in (("preprocessing_variant", "different"), ("seed", 1), ("nested_threshold", None)):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.candidate)
                changed[key] = value
                with self.assertRaises(ValueError):
                    validate_change(changed, self.reference)


if __name__ == "__main__":
    unittest.main()

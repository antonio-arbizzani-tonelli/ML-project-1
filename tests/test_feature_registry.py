"""Tests for codebook parsing and feature classification."""

from pathlib import Path
import unittest

from src.build_feature_registry import (
    candidate_duplicate_groups,
    classify_semantic_type,
    parse_codebook,
    special_values,
)


class TestFeatureRegistry(unittest.TestCase):
    """Check the transformations used to build the feature registry."""

    def test_parse_codebook_record(self):
        """Extract the description, response labels, and skip note."""

        path = Path("official_material/project1/brfss_2015_codebook.txt")
        record = parse_codebook(path)["GENHLTH"]

        self.assertEqual(record["title"], "General Health")
        self.assertEqual(record["section"], "1.1")
        self.assertEqual(record["source_kind"], "core_survey")
        self.assertEqual(
            record["description"],
            "Would you say that in general your health is:",
        )
        self.assertEqual(
            special_values(record["documented_values"]),
            [
                {"code": "7", "label": "Don’t know/Not Sure"},
                {"code": "9", "label": "Refused"},
                {"code": "BLANK", "label": "Not asked or Missing"},
            ],
        )

    def test_parse_codebook_retains_values_with_omitted_labels(self):
        """Keep sequence codes whose labels are blank in the printed table."""

        path = Path("official_material/project1/brfss_2015_codebook.txt")
        records = parse_codebook(path)
        interview_year_codes = {
            value["code"] for value in records["IYEAR"]["documented_values"]
        }
        adult_count_codes = {
            value["code"] for value in records["NUMADULT"]["documented_values"]
        }

        self.assertIn("2016", interview_year_codes)
        self.assertIn("2", adult_count_codes)

    def test_parse_codebook_keeps_split_label_fragments(self):
        """Join label fragments printed before the numeric frequency fields."""

        path = Path("official_material/project1/brfss_2015_codebook.txt")
        record = parse_codebook(path)["MAXVO2_"]

        self.assertIn(
            "two implied decimal places",
            record["documented_values"][0]["label"].lower(),
        )

    def test_semantic_type_uses_documented_responses(self):
        """Recognize a binary variable after excluding missing-value codes."""

        codebook = {
            "title": "Example diagnosis",
            "description": "Have you ever received the diagnosis?",
            "source_kind": "core_survey",
            "documented_values": [
                {"code": "1", "label": "Yes"},
                {"code": "2", "label": "No"},
                {"code": "7", "label": "Don't know"},
                {"code": "9", "label": "Refused"},
            ],
        }
        profile = {"unique_count": 4}

        semantic_type, _ = classify_semantic_type("EXAMPLE", codebook, profile)

        self.assertEqual(semantic_type, "binary")

    def test_duplicate_candidates_share_basic_statistics(self):
        """Compare only columns that could still be identical."""

        profiles = [
            {
                "name": "A",
                "missing_count": 2,
                "min": 1.0,
                "max": 3.0,
                "unique_count": 4,
            },
            {
                "name": "B",
                "missing_count": 2,
                "min": 1.0,
                "max": 3.0,
                "unique_count": 4,
            },
            {
                "name": "C",
                "missing_count": 0,
                "min": 1.0,
                "max": 3.0,
                "unique_count": 3,
            },
        ]

        self.assertEqual(candidate_duplicate_groups(profiles), [["A", "B"]])


if __name__ == "__main__":
    unittest.main()

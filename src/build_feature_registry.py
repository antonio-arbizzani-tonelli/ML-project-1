"""Build the feature registry from the CDC codebook and dataset audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

CODEBOOK_URL = "https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf"

IDENTIFIERS = {"Id", "SEQNO", "_PSU"}

SURVEY_DESIGN_FEATURES = {
    "DISPCODE",
    "MSCODE",
    "QSTLANG",
    "QSTVER",
    "_CHISPNC",
    "_CLLCPWT",
    "_CPRACE",
    "_CRACE1",
    "_DUALCOR",
    "_DUALUSE",
    "_LLCPWT",
    "_RAWRAKE",
    "_STSTR",
    "_STRWT",
    "_WT2RAKE",
}

CONTINUOUS_FEATURES = {
    "FC60_",
    "HEIGHT3",
    "HTIN4",
    "HTM4",
    "MAXVO2_",
    "METVL11_",
    "METVL21_",
    "WEIGHT2",
    "WTKG3",
    "_BMI5",
    "_CLLCPWT",
    "_LLCPWT",
    "_RAWRAKE",
    "_STRWT",
    "_WT2RAKE",
}

DATE_FEATURES = {"FLSHTMY2", "HIVTSTD3", "IDATE"}

CATEGORICAL_FEATURES = {
    "CAREGIV1",
    "EMPLOY1",
    "EXRACT11",
    "EXRACT21",
    "FMONTH",
    "IDAY",
    "IMONTH",
    "IYEAR",
    "MARITAL",
    "_STATE",
}

ORDINAL_FEATURES = {
    "CHECKUP1",
    "CHOLCHK",
    "CRGVHRS1",
    "CRGVLNG1",
    "HOWLONG",
    "HPLSTTST",
    "LASTPAP2",
    "LASTSIG3",
    "LASTSMK2",
    "LENGEXAM",
    "LSTBLDS3",
    "PSATIME",
    "SMOKDAY2",
    "USENOW3",
    "_PA150R2",
    "_PA300R2",
}

ORDINAL_KEYWORDS = {
    "age category",
    "age group",
    "degree of",
    "difficulty",
    "education",
    "employment",
    "general health",
    "how often",
    "income",
    "level of",
    "limited",
    "marital status",
    "pain rating",
    "satisfaction",
    "severity",
}

COUNT_KEYWORDS = {
    "age at",
    "age when",
    "days",
    "how long",
    "how many",
    "hours",
    "minutes",
    "number of",
    "times",
}

MISSING_LABEL_KEYWORDS = {
    "don't know",
    "don’t know",
    "missing",
    "not asked",
    "not sure",
    "refused",
}

HIGH_LEAKAGE = {
    "HAREHAB1": (
        "The question is asked about rehabilitation following a heart attack "
        "and is skipped when CVDINFR4 is not positive."
    )
}

STAFF_QUESTIONS = {
    "HAREHAB1": (
        "Is this variable valid for competition use even though the question is "
        "asked only after a reported heart attack?"
    )
}

POSSIBLE_LEAKAGE = {
    "CVDASPRN": "Aspirin advice may reflect previously identified cardiovascular risk.",
    "RDUCHART": (
        "Aspirin use to prevent heart attack may follow a cardiovascular diagnosis."
    ),
}


def parse_args(arguments: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("codebook_text", type=Path)
    parser.add_argument("dataset_audit", type=Path)
    parser.add_argument("x_train", type=Path)
    parser.add_argument("metadata_output", type=Path)
    parser.add_argument("registry_json_output", type=Path)
    parser.add_argument("registry_csv_output", type=Path)
    parser.add_argument("summary_output", type=Path)
    return parser.parse_args(arguments)


def normalize_space(text: str) -> str:
    """Collapse repeated whitespace into single spaces."""

    return " ".join(text.replace("�", "'").split())


def is_page_header(line: str) -> bool:
    """Return whether a line belongs to the repeated PDF page header."""

    text = normalize_space(line)
    return (
        not text
        or text.startswith("BEHAVIORAL RISK FACTOR SURVEILLANCE SYSTEM")
        or text.startswith("CODEBOOK REPORT")
        or text == "Land-Line and Cell-Phone data"
        or bool(re.match(r"^\d+ of \d+", text))
    )


def previous_content_line(lines: Sequence[str], start: int) -> str:
    """Return the closest non-header line before a position."""

    for index in range(start, -1, -1):
        if not is_page_header(lines[index]):
            return normalize_space(lines[index])
    return ""


def extract_description(lines: Sequence[str], variable_line: int) -> str:
    """Extract the description following a codebook variable heading."""

    for index in range(variable_line + 1, min(variable_line + 12, len(lines))):
        match = re.search(r"Description:\s*(.*)", lines[index])
        if not match:
            continue

        parts = [match.group(1).strip()]
        for continuation in lines[index + 1 :]:
            normalized = normalize_space(continuation)
            if "Value Label" in normalized or "Weighted" in normalized:
                break
            if normalized and not is_page_header(continuation):
                parts.append(normalized)
        return normalize_space(" ".join(parts))
    return ""


def extract_value_labels(lines: Sequence[str]) -> List[Dict[str, str]]:
    """Extract documented codes and labels from one codebook block."""

    values: List[Dict[str, str]] = []
    code_pattern = re.compile(r"^(BLANK|\d+(?:\s*-\s*\d+)?)$")

    pending_code = ""
    for line in lines:
        parts = re.split(r"\s{2,}", line.strip())
        if pending_code and parts and not code_pattern.fullmatch(parts[0]):
            label = normalize_space(parts[0]).strip("-")
            if label and not re.fullmatch(r"[\d,.-]+", label):
                values.append({"code": pending_code, "label": label})
                pending_code = ""
            continue
        if len(parts) < 2 or not code_pattern.fullmatch(parts[0]):
            continue
        label_parts = []
        for part in parts[1:]:
            if re.fullmatch(r"[\d,.-]+", part):
                break
            label_parts.append(part)
        label = normalize_space(" ".join(label_parts))
        if label == "-":
            pending_code = parts[0]
            continue
        if not label or re.fullmatch(r"[\d,.-]+", label):
            # The CDC codebook often prints a label only on the first row of a
            # simple sequence (days 1--31, household counts, interview years).
            # The following rows still document valid codes, but their second
            # whitespace-delimited field is already the frequency.  Retain the
            # code with a neutral label so downstream domain validation does
            # not mistake these values for anomalies.
            values.append(
                {"code": parts[0], "label": "Documented value (label omitted)"}
            )
            continue
        values.append({"code": parts[0], "label": label})

    unique: Dict[tuple, Dict[str, str]] = {}
    for value in values:
        unique[(value["code"], value["label"])] = value
    return list(unique.values())


def extract_notes(lines: Sequence[str]) -> List[str]:
    """Extract codebook notes that explain derivations and skip patterns."""

    notes = []
    for index, line in enumerate(lines):
        match = re.search(r"Notes:\s*(.*)", line)
        if not match:
            continue
        parts = [match.group(1).strip()]
        for continuation in lines[index + 1 : index + 3]:
            text = normalize_space(continuation)
            if text and not re.match(r"^(\d+|BLANK)\b", text):
                parts.append(text)
        notes.append(normalize_space(" ".join(parts)))
    return list(dict.fromkeys(note for note in notes if note))


def parse_codebook(path: Path) -> Dict[str, Dict[str, object]]:
    """Parse the CDC text export into records keyed by variable name."""

    pages = path.read_text(encoding="utf-8").split("\f")
    records: Dict[str, Dict[str, object]] = {}

    for page_number, page in enumerate(pages, start=1):
        lines = page.splitlines()
        variable_positions = [
            index for index, line in enumerate(lines) if "SAS Variable Name:" in line
        ]

        for position_index, variable_line in enumerate(variable_positions):
            match = re.search(r"SAS Variable Name:\s*(\S+)", lines[variable_line])
            if match is None:
                continue
            name = match.group(1)
            block_end = (
                variable_positions[position_index + 1]
                if position_index + 1 < len(variable_positions)
                else len(lines)
            )
            block = lines[variable_line:block_end]

            descriptor_index = variable_line - 1
            while descriptor_index >= 0:
                descriptor = normalize_space(lines[descriptor_index])
                if "Type:" in descriptor:
                    break
                descriptor_index -= 1
            descriptor = (
                normalize_space(lines[descriptor_index])
                if descriptor_index >= 0
                else ""
            )
            title = previous_content_line(lines, descriptor_index - 1)

            if "Calculated" in descriptor:
                source_kind = "derived"
            elif "Module:" in descriptor:
                source_kind = "optional_module"
            elif "Section:" in descriptor:
                source_kind = "core_survey"
            elif "Weighting:" in descriptor:
                source_kind = "survey_design"
            else:
                source_kind = "administrative"

            section_match = re.search(
                r"(?:Section:|Module:|LandLine:|CellPhone:|Weighting:)\s*([\d.]+)",
                descriptor,
            )
            if section_match is None and "Calculated" in descriptor:
                section_match = re.search(r"([\d.]+)", descriptor)

            record = records.setdefault(
                name,
                {
                    "name": name,
                    "title": title,
                    "description": extract_description(lines, variable_line),
                    "source_kind": source_kind,
                    "section": section_match.group(1) if section_match else "",
                    "codebook_pages": [],
                    "documented_values": [],
                    "notes": [],
                },
            )
            record["codebook_pages"].append(page_number)
            record["documented_values"].extend(extract_value_labels(block))
            record["notes"].extend(extract_notes(block))

    for record in records.values():
        record["codebook_pages"] = sorted(set(record["codebook_pages"]))
        value_keys = set()
        unique_values = []
        for value in record["documented_values"]:
            key = (value["code"], value["label"])
            if key not in value_keys:
                value_keys.add(key)
                unique_values.append(value)
        record["documented_values"] = unique_values
        record["notes"] = list(dict.fromkeys(record["notes"]))

    return records


def special_values(values: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return codes documented as missing, unknown, or refused."""

    result = []
    for value in values:
        label = value["label"].lower()
        if any(keyword in label for keyword in MISSING_LABEL_KEYWORDS):
            result.append(value)
    return result


def substantive_values(values: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return documented values that represent actual answers."""

    special = {(value["code"], value["label"]) for value in special_values(values)}
    return [
        value
        for value in values
        if (value["code"], value["label"]) not in special and value["code"] != "BLANK"
    ]


def classify_semantic_type(
    name: str, codebook: Dict[str, object], profile: Dict[str, object]
) -> tuple[str, str]:
    """Assign a conservative initial type using codebook and data properties."""

    if name in IDENTIFIERS:
        return "identifier", "Explicit identifier"
    if name in SURVEY_DESIGN_FEATURES:
        return "survey_design", "Survey administration or weighting variable"
    if name in DATE_FEATURES:
        return "date", "Calendar month and year or complete interview date"
    if name in CATEGORICAL_FEATURES:
        return "categorical", "Named categories without a useful numeric distance"
    if name in ORDINAL_FEATURES:
        return "ordinal", "Documented categories follow a meaningful order"
    if name in CONTINUOUS_FEATURES:
        return "continuous", "Measured or calculated continuous quantity"

    description = str(codebook["description"]).lower()
    title = str(codebook["title"]).lower()
    combined_text = f"{title} {description}"
    values = substantive_values(codebook["documented_values"])
    codes = {value["code"] for value in values}

    if len(codes) == 2 and not any("-" in code for code in codes):
        return "binary", "Two documented substantive responses"
    if any(keyword in combined_text for keyword in ORDINAL_KEYWORDS):
        return "ordinal", "The documented responses have a natural order"
    if any(keyword in combined_text for keyword in COUNT_KEYWORDS):
        return "count", "The variable records a number, duration, or frequency"

    unique_count = profile["unique_count"]
    if isinstance(unique_count, int) and unique_count <= 20:
        return "categorical", "A small set of documented response categories"
    if any("-" in code for code in codes):
        return "continuous", "The codebook documents a numeric value range"
    if codebook["source_kind"] == "derived":
        return "continuous", "Calculated variable with many possible values"
    return "unresolved", "The codebook does not imply a safe numeric treatment"


def leakage_annotation(name: str) -> tuple[str, str, str]:
    """Return temporal role, risk level, and supporting reason."""

    if name in HIGH_LEAKAGE:
        return "post_outcome", "high", HIGH_LEAKAGE[name]
    if name in POSSIBLE_LEAKAGE:
        return "contemporaneous", "possible", POSSIBLE_LEAKAGE[name]
    return "contemporaneous", "none", ""


def candidate_duplicate_groups(
    profiles: Sequence[Dict[str, object]],
) -> List[List[str]]:
    """Group columns that could be exact duplicates from their summaries."""

    groups: Dict[tuple, List[str]] = defaultdict(list)
    for profile in profiles:
        key = (
            profile["missing_count"],
            profile["min"],
            profile["max"],
            str(profile["unique_count"]),
        )
        groups[key].append(profile["name"])
    return [group for group in groups.values() if len(group) > 1]


def find_exact_duplicates(
    csv_path: Path, profiles: Sequence[Dict[str, object]]
) -> Dict[str, str]:
    """Return duplicate columns mapped to the first matching column."""

    candidate_groups = candidate_duplicate_groups(profiles)
    candidate_set = {name for group in candidate_groups for name in group}
    candidate_names = [
        profile["name"] for profile in profiles if profile["name"] in candidate_set
    ]
    digests = {name: hashlib.blake2b(digest_size=16) for name in candidate_names}

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        indexes = {name: header.index(name) for name in candidate_names}
        for row in reader:
            for name, digest in digests.items():
                digest.update(row[indexes[name]].encode("utf-8"))
                digest.update(b"\x00")

    hash_groups: Dict[bytes, List[str]] = defaultdict(list)
    for name, digest in digests.items():
        hash_groups[digest.digest()].append(name)
    possible_matches = [names for names in hash_groups.values() if len(names) > 1]
    if not possible_matches:
        return {}

    comparisons = {
        name: [reference, True]
        for names in possible_matches
        for reference in names[:1]
        for name in names[1:]
    }
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        indexes = {name: header.index(name) for name in candidate_names}
        for row in reader:
            for name, comparison in comparisons.items():
                if comparison[1] and row[indexes[name]] != row[indexes[comparison[0]]]:
                    comparison[1] = False

    return {
        name: comparison[0] for name, comparison in comparisons.items() if comparison[1]
    }


def build_metadata(
    feature_names: Sequence[str],
    codebook_records: Dict[str, Dict[str, object]],
    train_profiles: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    """Build semantic metadata in dataset column order."""

    metadata = [
        {
            "name": "Id",
            "title": "Project row identifier",
            "description": "Identifier assigned to each project observation.",
            "semantic_type": "identifier",
            "semantic_type_basis": "Project file structure",
            "source_kind": "project_identifier",
            "section": "",
            "codebook_pages": [],
            "documented_values": [],
            "special_values": [],
            "blank_meaning": "Not blank",
            "temporal_role": "not_applicable",
            "leakage_risk": "none",
            "leakage_reason": "",
            "competition_use": "include",
            "staff_question": "",
            "notes": [],
            "source": "Project CSV files",
        }
    ]

    for name in feature_names:
        codebook = codebook_records[name]
        semantic_type, type_basis = classify_semantic_type(
            name, codebook, train_profiles[name]
        )
        temporal_role, risk, risk_reason = leakage_annotation(name)
        documented_special_values = special_values(codebook["documented_values"])
        blank_labels = [
            value["label"]
            for value in codebook["documented_values"]
            if value["code"] == "BLANK"
        ]
        metadata.append(
            {
                "name": name,
                "title": codebook["title"],
                "description": codebook["description"],
                "semantic_type": semantic_type,
                "semantic_type_basis": type_basis,
                "source_kind": codebook["source_kind"],
                "section": codebook["section"],
                "codebook_pages": codebook["codebook_pages"],
                "documented_values": codebook["documented_values"],
                "special_values": documented_special_values,
                "blank_meaning": (
                    "; ".join(dict.fromkeys(blank_labels))
                    if blank_labels
                    else "No documented blank category"
                ),
                "temporal_role": temporal_role,
                "leakage_risk": risk,
                "leakage_reason": risk_reason,
                "competition_use": "include",
                "staff_question": STAFF_QUESTIONS.get(name, ""),
                "notes": codebook["notes"],
                "source": CODEBOOK_URL,
            }
        )
    return metadata


def build_registry(
    metadata: Sequence[Dict[str, object]],
    audit: Dict[str, object],
    duplicates: Dict[str, str],
) -> List[Dict[str, object]]:
    """Combine semantic metadata with observed train and test statistics."""

    train_columns = {column["name"]: column for column in audit["x_train"]["columns"]}
    test_columns = {column["name"]: column for column in audit["x_test"]["columns"]}
    registry = []

    for position, entry in enumerate(metadata):
        name = entry["name"]
        if name == "Id":
            observed = {
                "position": position,
                "train_missing_fraction": 0.0,
                "test_missing_fraction": 0.0,
                "missing_fraction_difference": 0.0,
                "train_unique_count": audit["x_train"]["row_count"],
                "train_top_values": [],
                "test_top_values": [],
                "train_min": audit["x_train"]["first_id"],
                "train_max": audit["x_train"]["last_id"],
                "exact_duplicate_of": "",
            }
        else:
            train = train_columns[name]
            test = test_columns[name]
            observed = {
                "position": position,
                "train_missing_fraction": train["missing_fraction"],
                "test_missing_fraction": test["missing_fraction"],
                "missing_fraction_difference": abs(
                    train["missing_fraction"] - test["missing_fraction"]
                ),
                "train_unique_count": train["unique_count"],
                "train_top_values": train["top_values"],
                "test_top_values": test["top_values"],
                "train_min": train["min"],
                "train_max": train["max"],
                "exact_duplicate_of": duplicates.get(name, ""),
            }
        registry.append({**entry, **observed})
    return registry


def write_json(path: Path, value: object) -> None:
    """Write formatted JSON, creating its parent directory if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_registry_csv(path: Path, registry: Sequence[Dict[str, object]]) -> None:
    """Write the main registry fields as a reviewable table."""

    fields = [
        "position",
        "name",
        "title",
        "description",
        "semantic_type",
        "semantic_type_basis",
        "source_kind",
        "section",
        "codebook_pages",
        "special_values",
        "blank_meaning",
        "temporal_role",
        "leakage_risk",
        "leakage_reason",
        "competition_use",
        "staff_question",
        "train_missing_fraction",
        "test_missing_fraction",
        "missing_fraction_difference",
        "train_unique_count",
        "train_top_values",
        "test_top_values",
        "train_min",
        "train_max",
        "exact_duplicate_of",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for entry in registry:
            row = dict(entry)
            row["codebook_pages"] = json.dumps(row["codebook_pages"])
            row["special_values"] = json.dumps(
                row["special_values"], ensure_ascii=False
            )
            row["train_top_values"] = json.dumps(row["train_top_values"])
            row["test_top_values"] = json.dumps(row["test_top_values"])
            writer.writerow(row)


def write_summary(path: Path, registry: Sequence[Dict[str, object]]) -> None:
    """Write a compact overview of the completed registry."""

    type_counts = Counter(entry["semantic_type"] for entry in registry)
    source_counts = Counter(entry["source_kind"] for entry in registry)
    risk_counts = Counter(entry["leakage_risk"] for entry in registry)
    duplicates = [entry for entry in registry if entry["exact_duplicate_of"]]
    high_risk = [entry for entry in registry if entry["leakage_risk"] == "high"]
    staff_questions = [entry for entry in registry if entry["staff_question"]]
    unresolved = [entry for entry in registry if entry["semantic_type"] == "unresolved"]

    lines = [
        "# Feature Registry Summary",
        "",
        f"The registry contains {len(registry)} columns including `Id`.",
        "",
        "## Semantic types",
        "",
        "| Type | Columns |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| `{name}` | {count} |" for name, count in sorted(type_counts.items())
    )
    lines.extend(
        [
            "",
            "## Source kinds",
            "",
            "| Source | Columns |",
            "| --- | ---: |",
        ]
    )
    lines.extend(
        f"| `{name}` | {count} |" for name, count in sorted(source_counts.items())
    )
    lines.extend(
        [
            "",
            "## Leakage review",
            "",
            "All supplied variables remain included in the competition feature set. "
            "The leakage labels are used for interpretation and controlled ablation.",
            "",
            "| Risk | Columns |",
            "| --- | ---: |",
        ]
    )
    lines.extend(
        f"| `{name}` | {count} |" for name, count in sorted(risk_counts.items())
    )
    lines.extend(
        [
            "",
            "High-risk variables:",
            "",
        ]
    )
    lines.extend(
        f"- `{entry['name']}`: {entry['leakage_reason']}" for entry in high_risk
    )
    lines.extend(
        [
            "",
            "Questions for the teaching staff:",
            "",
        ]
    )
    lines.extend(
        f"- `{entry['name']}`: {entry['staff_question']}" for entry in staff_questions
    )
    lines.extend(
        [
            "",
            "## Exact duplicate columns",
            "",
        ]
    )
    lines.extend(
        f"- `{entry['name']}` duplicates `{entry['exact_duplicate_of']}`."
        for entry in duplicates
    )
    lines.extend(
        [
            "",
            "## Unresolved semantic types",
            "",
        ]
    )
    if unresolved:
        lines.extend(
            f"- `{entry['name']}`: {entry['description']}" for entry in unresolved
        )
    else:
        lines.append("No unresolved semantic types.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(arguments: Optional[Sequence[str]] = None) -> None:
    """Build all feature-registry artifacts."""

    args = parse_args(arguments)
    audit = json.loads(args.dataset_audit.read_text(encoding="utf-8"))
    codebook = parse_codebook(args.codebook_text)
    train_profiles = {column["name"]: column for column in audit["x_train"]["columns"]}
    feature_names = list(train_profiles)

    metadata = build_metadata(feature_names, codebook, train_profiles)
    duplicates = find_exact_duplicates(args.x_train, list(train_profiles.values()))
    registry = build_registry(metadata, audit, duplicates)

    write_json(args.metadata_output, metadata)
    write_json(args.registry_json_output, registry)
    write_registry_csv(args.registry_csv_output, registry)
    write_summary(args.summary_output, registry)

    print(f"Wrote metadata for {len(metadata)} columns")
    print(f"Exact duplicate columns: {len(duplicates)}")
    print(
        "Unresolved semantic types: "
        f"{sum(entry['semantic_type'] == 'unresolved' for entry in metadata)}"
    )


if __name__ == "__main__":
    main()

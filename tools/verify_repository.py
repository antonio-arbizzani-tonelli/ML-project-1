"""Check intended Git contents, documentation links and frozen experiment evidence.

This administrative check requires Git, but no official CSV data or checkpoints.
Run from the repository root with python -m tools.verify_repository.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
BASELINE = (
    "results/experiments/20260921T105139261213Z_"
    "phase14-boosting-codebook-corrections-only-depth5-features128-200trees.json"
)
BASELINE_SUITE = "configs/experiments/phase14_boosting_codebook_corrections_only.json"
LINK = re.compile(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)")
LOCAL_PREFIXES = (
    "data/raw/dataset/", "data/processed/", "results/boosting_artifacts/",
    "results/mlp_artifacts/", "results/oof_predictions/", ".local/", ".venv/",
)


def published_files(root: Path) -> set[str]:
    """Include tracked files and eligible untracked files, never ignored files."""
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
    )
    return set(filter(None, output.decode("utf-8").split("\0")))


def verify(root: Path = ROOT) -> dict:
    """Fail on inconsistent delivery contents without modifying the repository."""
    files = published_files(root)
    errors = []
    counts = {"eligible_files": len(files), "python_files": 0,
              "json_files": 0, "markdown_links": 0, "indexed_records": 0}
    required = (
        "README.md", "run.py", "helpers.py", "implementations.py", "requirements.txt",
        "configs/final_model.json", "docs/FINAL_REPORT.md", "docs/FINAL_AUDIT.md",
        "docs/GITHUB_RELEASE.md", ".github/workflows/tests.yml",
        "official_material/project1/brfss_2015_codebook.txt",
        "results/eda/analysis/splits/stratified_seed_20260918.npz", BASELINE,
    )
    for name in required:
        if name not in files or not (root / name).is_file():
            errors.append(f"Missing delivery file: {name}")

    for name in sorted(files):
        path = root / name
        if not path.is_file():
            errors.append(f"Tracked or eligible file is absent: {name}")
            continue
        if (name.startswith(LOCAL_PREFIXES) or path.suffix in {".pkl", ".npy", ".log"}
                or path.name.startswith("submission") and path.suffix == ".csv"
                or "__pycache__" in path.parts):
            errors.append(f"Local generated artifact eligible for publication: {name}")
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f"Delivery file exceeds 10 MiB: {name}")
        if path.suffix == ".py":
            counts["python_files"] += 1
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=name,
                          feature_version=(3, 9))
            except (SyntaxError, UnicodeError) as exc:
                errors.append(f"Python 3.9 syntax check failed: {name}: {exc}")
        if path.suffix == ".json":
            counts["json_files"] += 1
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, UnicodeError) as exc:
                errors.append(f"Invalid JSON: {name}: {exc}")
        if path.suffix == ".md":
            for match in LINK.finditer(path.read_text(encoding="utf-8")):
                target = match.group(1).strip("<>")
                if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
                    continue
                target = unquote(target.split("#", 1)[0].split("?", 1)[0])
                destination = (path.parent / target).resolve()
                counts["markdown_links"] += 1
                try:
                    relative = destination.relative_to(root.resolve()).as_posix()
                except ValueError:
                    errors.append(f"Documentation link escapes the repository: {name}: {target}")
                    continue
                if destination.is_dir():
                    if not any(item.startswith(relative + "/") for item in files):
                        errors.append(f"Unpublished directory link: {name}: {target}")
                elif relative not in files or not destination.is_file():
                    errors.append(f"Missing published documentation link: {name}: {target}")

    for index in sorted((root / "results/experiments").rglob("index.csv")):
        if index.relative_to(root).as_posix() not in files:
            continue
        with index.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            counts["indexed_records"] += 1
            record = index.parent / row["record_path"]
            if (not record.is_file()
                    or record.relative_to(root).as_posix() not in files):
                errors.append(f"Missing indexed record: {record.relative_to(root)}")
            elif hashlib.sha256(record.read_bytes()).hexdigest() != row["record_sha256"]:
                errors.append(f"Indexed record hash mismatch: {record.relative_to(root)}")

    final = json.loads((root / "configs/final_model.json").read_text(encoding="utf-8"))
    validation = final.get("validation", {
        "name": "Phase 14", "suite": BASELINE_SUITE, "record": BASELINE,
    })
    validation_type = validation.get("type", "nested_development")
    if validation_type not in {"nested_development", "full_train_oof_calibration"}:
        errors.append("Unknown selected validation evidence type.")
    calibration_only = validation_type == "full_train_oof_calibration"
    if calibration_only:
        if "calibration" not in final:
            errors.append("Full-train validation requires a calibration record.")
        elif validation["record"] != final["calibration"]["record"]:
            errors.append("Selected validation and full-train calibration records differ.")
    selected = {}
    for kind in ("suite", "record"):
        name = validation[kind]
        if name not in files or not (root / name).is_file():
            errors.append(f"Missing published validation {kind}: {name}")
        else:
            selected[kind] = json.loads((root / name).read_text(encoding="utf-8"))
    if "suite" in selected:
        suite = selected["suite"]
        parameters = dict(suite["models"][0]["parameters"])
        parameters["n_estimators"] = suite["models"][0]["checkpoints"][-1]
        if final["model"] != parameters:
            errors.append("Final model parameters differ from the selected validation suite.")
        for field, suite_field in (("config", "preprocessing_config_path"),
                                   ("variant", "preprocessing_variant"),
                                   ("metadata", "feature_metadata_path")):
            if final["preprocessing"][field] != suite[suite_field]:
                errors.append(f"Final preprocessing {field} differs from the selected validation suite.")
    nested_f1 = None
    if "record" in selected and not calibration_only:
        record = selected["record"]
        if final["model"] != record["config"]["model"]["parameters"]:
            errors.append("Final model parameters differ from the selected validation record.")
        for field, record_field in (("config", "preprocessing_source"),
                                    ("variant", "preprocessing_variant")):
            if final["preprocessing"][field] != record["config"]["features"][record_field]:
                errors.append(f"Final preprocessing {field} differs from the selected validation record.")
        nested = record["outcome"]["nested_threshold_evaluation"]
        nested_f1 = nested["pooled_metrics"]["f1"]
        mean_threshold = sum(f["inner_threshold"] for f in nested["folds"]) / len(nested["folds"])
        if "calibration" not in final and abs(final["decision_threshold"] - mean_threshold) > 1e-15:
            errors.append("Final threshold differs from the mean of selected validation inner thresholds.")
    if "implementation" in final:
        implementation = final["implementation"]
        source_name = {"phase14": "src/numpy_boosting_phase14.py",
                       "phase34": "src/numpy_boosting.py"}.get(implementation)
        if source_name is None:
            errors.append("Unknown selected booster implementation.")
        elif source_name not in files or not (root / source_name).is_file():
            errors.append(f"Missing published booster implementation: {source_name}")
        elif hashlib.sha256((root / source_name).read_bytes()).hexdigest() != final.get("implementation_source_sha256"):
            errors.append("Selected booster source differs from the frozen implementation.")
        if "suite" in selected and selected["suite"].get("implementation", "phase34") != implementation:
            errors.append("Selected CV suite uses a different booster implementation.")
    if "calibration" in final:
        name = final["calibration"]["record"]
        if name not in files or not (root / name).is_file():
            errors.append(f"Missing published full-train calibration record: {name}")
        else:
            calibration = json.loads((root / name).read_text(encoding="utf-8"))
            if calibration["model"] != final["model"]:
                errors.append("Full-train calibration and final model parameters differ.")
            if (calibration_only or "preprocessing" in calibration) and calibration.get("preprocessing") != final["preprocessing"]:
                errors.append("Full-train calibration and final preprocessing differ.")
            if abs(calibration["threshold"]["value"] - final["decision_threshold"]) > 1e-15:
                errors.append("Final threshold differs from its full-train OOF calibration record.")
            if calibration["implementation"] != final.get("implementation"):
                errors.append("Full-train calibration uses a different booster implementation.")
            if calibration["implementation_sha256"] != final.get("implementation_source_sha256"):
                errors.append("Full-train calibration source fingerprint differs from the selected source.")
            if (calibration["threshold"]["method"] != "pooled_3_fold_full_train_oof_f1"
                    or calibration["threshold"]["training_rows"] != calibration["training_rows"]
                    or calibration["threshold"]["fold_count"] != 3):
                errors.append("Full-train threshold calibration has inconsistent scope or protocol.")
    for field in ("config", "metadata"):
        name = final["preprocessing"][field]
        if name not in files or not (root / name).is_file():
            errors.append(f"Missing published preprocessing {field}: {name}")
    preprocessing_path = root / final["preprocessing"]["config"]
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    plan = next(v["preprocessing"] for v in preprocessing["variants"]
                if v["experiment_name"] == final["preprocessing"]["variant"])
    if "HAREHAB1" in plan.get("exclude_features", []):
        errors.append("HAREHAB1 is excluded despite the documented retention decision.")
    if (root / "requirements.txt").read_text().strip() != "numpy==1.23.1":
        errors.append("The official training dependency pin differs from NumPy 1.23.1.")

    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version()},
        "passed": not errors, "counts": counts, "errors": errors,
        "selected_model": validation["name"], "nested_development_f1": nested_f1,
        "validation_type": validation_type,
        "submission_threshold": final["decision_threshold"],
        "scope": "Git-eligible working-tree files, links, syntax, JSON, ledger hashes and frozen configuration. No training or official-runtime execution.",
    }


def main() -> None:
    """Print verification and optionally save its compact result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify()
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

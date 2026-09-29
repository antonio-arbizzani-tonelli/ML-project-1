"""Run the existing booster CV with progress and environment provenance."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from src import run_boosting_cv as runner
from src.run_preprocessing_cv import _sha256


def _logged_fit(original, profiles: list[dict], outer_fold_count: int):
    """Add progress checkpoints while keeping requested predictions unchanged."""

    def fit(train_features, train_labels, validation_features, parameters,
            checkpoints, checkpoint_callback=None):
        number = len(profiles) + 1
        stage = "outer" if number <= outer_fold_count else "inner"
        started = time.perf_counter()
        entry = {"fit_number": number, "stage": stage,
                 "training_rows": int(train_labels.size),
                 "validation_rows": int(validation_features.shape[0])}
        profiles.append(entry)
        print(f"START fit={number} stage={stage} train={train_labels.size} "
              f"validation={validation_features.shape[0]}", flush=True)
        steps = sorted(set(checkpoints) | set(range(25, checkpoints[-1] + 1, 25)))

        def progress(step, model):
            print(f"PROGRESS fit={number} stage={stage} trees={step}/{checkpoints[-1]} "
                  f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
            if checkpoint_callback is not None and step in checkpoints:
                checkpoint_callback(step, model)

        probabilities, profile = original(
            train_features, train_labels, validation_features, parameters,
            steps, progress,
        )
        entry["elapsed_seconds"] = time.perf_counter() - started
        entry["tree_fit_seconds"] = profile["total_seconds"]
        print(f"DONE fit={number} stage={stage} elapsed={entry['elapsed_seconds']:.1f}s",
              flush=True)
        return {step: probabilities[step] for step in checkpoints}, profile

    return fit


def run_logged_suite(config_path: Path, root: Path, output_dir: Path) -> list[Path]:
    """Keep the experiment implementation intact and record each fit's cost."""

    suite = json.loads(config_path.read_text(encoding="utf-8"))
    if len(suite["models"]) != 1:
        raise ValueError("Logged cost accounting requires one model per suite.")
    original_fit = runner._fit_checkpoint_probabilities
    original_save = runner.save_experiment
    profiles = []
    source_names = (
        "src/numpy_boosting.py", "src/feature_preprocessing.py",
        "src/evaluation.py", "src/run_boosting_cv.py",
        "src/run_logged_boosting_cv.py",
    )
    source_hashes = {name: _sha256(root / name) for name in source_names}

    def save(config, outcome, directory):
        if "hypothesis" in suite:
            config["hypothesis"] = suite["hypothesis"]
        outcome["environment"] = {
            "python": platform.python_version(), "numpy": np.__version__,
        }
        outcome["source_sha256"] = source_hashes
        outcome["individual_fit_profiles"] = list(profiles)
        outcome["progress_note"] = (
            "Extra checkpoints only report progress and compute intermediate "
            "validation predictions. They do not alter trees or random draws."
        )
        return original_save(config, outcome, directory)

    runner._fit_checkpoint_probabilities = _logged_fit(
        original_fit, profiles, suite["fold_count"]
    )
    runner.save_experiment = save
    try:
        return runner.run_suite(config_path, root, output_dir)
    finally:
        runner._fit_checkpoint_probabilities = original_fit
        runner.save_experiment = original_save


def main() -> None:
    """Run a logged suite from the project root."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    for record in run_logged_suite(args.config, root, args.output_dir):
        print(f"Saved experiment record: {record}", flush=True)


if __name__ == "__main__":
    main()

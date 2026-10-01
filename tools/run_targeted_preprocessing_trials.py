"""Run both authorized trials in order and save verified results after each."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

from src.run_logged_boosting_cv import run_logged_suite
from tools.targeted_preprocessing_analysis import TRIALS, compare, write_report


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    root = Path.cwd()
    summaries = []
    for phase, prefix in TRIALS.items():
        directory = root / "results/boosting_artifacts" / prefix
        directory.mkdir(parents=True, exist_ok=True)
        if list((directory / "records").glob("*.json")):
            raise ValueError(f"Phase {phase} already has a record; inspect it before retraining.")
        with (directory / "run_stdout.log").open("w", encoding="utf-8") as log:
            tee = _Tee(sys.stdout, log)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                print(f"BEGIN phase={phase}: {prefix}", flush=True)
                records = run_logged_suite(root / f"configs/experiments/{prefix}.json",
                                           root, directory / "records")
                if len(records) != 1:
                    raise ValueError("Expected one record per trial.")
                summaries.append(compare(root, phase, records[0]))
                print(f"COMPLETE phase={phase}; verified report saved", flush=True)
    write_report(root, summaries)
    print("COMPLETE both trials; combined report saved", flush=True)


if __name__ == "__main__":
    main()

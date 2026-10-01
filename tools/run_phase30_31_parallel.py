"""Launch both verified trials concurrently and publish records serially."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.summarize_phase19_transfer import _publish
from tools.phase30_31_trials import PREFIXES, configuration, paths, read, verify_frozen


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, status):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    root = Path.cwd()
    directory = root / "results/boosting_artifacts/phase30_31_parallel"
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / "status.json"
    if status_path.exists():
        raise ValueError("Existing parallel run: inspect status before relaunching.")
    for phase in (30, 31):
        suite, _, _ = configuration(root, phase)
        verify_frozen(root, phase, suite, read(root, paths(phase)[1]))
        if (root / suite["artifact_dir"] / "executed_runner.py").exists():
            raise ValueError(f"Phase {phase} already started.")
    status = {"started_at_utc": now(), "status": "running", "trials": {}}
    jobs = {}
    streams = []
    try:
        for phase in (30, 31):
            out = (directory / f"phase{phase}_stdout.log").open("w", encoding="utf-8")
            err = (directory / f"phase{phase}_stderr.log").open("w", encoding="utf-8")
            streams.extend((out, err))
            process = subprocess.Popen([sys.executable, "-u", "-m", "tools.phase30_31_trials", str(phase), "run"],
                                       cwd=root, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
            jobs[phase] = process
            status["trials"][str(phase)] = {"pid": process.pid, "status": "running", "started_at_utc": now(),
                                            "artifact_dir": f"results/boosting_artifacts/{PREFIXES[phase]}"}
            save(status_path, status)
            print(f"STARTED phase={phase} pid={process.pid}", flush=True)
        remaining = set(jobs)
        while remaining:
            for phase in sorted(remaining):
                code = jobs[phase].poll()
                if code is None:
                    continue
                item = status["trials"][str(phase)]
                item.update(exit_code=code, completed_at_utc=now(), status="completed" if code == 0 else "failed")
                if code == 0:
                    source = next((root / item["artifact_dir"] / "records").glob("*.json"))
                    # Only this controller writes the common ledger.
                    item["published_record"] = str(_publish(source, root).relative_to(root)).replace("\\", "/")
                    comparison = read(root, f"{item['artifact_dir']}/comparison.json")
                    item.update(nested_f1=comparison["candidate_nested"]["pooled_metrics"]["f1"],
                                delta_f1=comparison["metric_deltas"]["f1"],
                                meets_plan_promotion_rule=comparison["meets_plan_promotion_rule"])
                remaining.remove(phase)
                save(status_path, status)
                print(f"FINISHED phase={phase} exit_code={code}", flush=True)
            if remaining:
                time.sleep(10)
        status.update(status="completed" if all(job.returncode == 0 for job in jobs.values()) else "failed", completed_at_utc=now())
        save(status_path, status)
    except BaseException as error:
        status.update(status="controller_failed", error=repr(error))
        save(status_path, status)
        raise
    finally:
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    main()

# Final delivery audit - 21 October 2026

The project release selects **combined18_24_features256**, with 297 processed
columns, 200 trees of depth 5, 256 candidate features per node and threshold
**0.22007772525099117**. AIcrowd feedback reported on 14 October gives
**F1 0.455 and accuracy 0.883**. The [selected submission record](../results/eda/final_combined18_24_submission.json)
and [17-candidate comparison](../results/eda/final_combined_experiments_summary.json)
preserve its provenance. The previous Phase 14 release remains historical evidence.

## Requirements and final contents

| Requirement | Verified delivery |
| --- | --- |
| Six required methods | `implementations.py`; all applicable public numerical tests pass |
| Executable reproduction | `run.py` and `configs/final_model.json` select the combined preprocessing and frozen historical booster |
| Permitted libraries | Training uses NumPy and Python standard libraries; visualization uses Matplotlib |
| Written report | English LaTeX report, two A4 pages, no appendix; PDF compiled and both pages checked visually |
| Reproducible settings and evidence | Selected config, historical configs, fixed development split, 114 historical ledger records, 17 later candidate summaries and dated aggregate feedback |
| Public code repository | GitHub API confirms `ML-project-1` is public on 21 October; hand in the precise commit URL |
| Local material excluded | Official datasets, caches, checkpoints, individual OOF/test probabilities, CSV submissions, logs and temporary scheduling helpers |

The delivery contains approximately **22.1 MiB**. Original local experiment
manifests remain untouched. Compact aggregate summaries retain their hashes;
15 published feedback/verification fingerprints and five selected numerical
source fingerprints were checked against the raw staged blobs. Git attributes
preserve the exact original bytes of frozen sources and recipes, including
original CRLF files. The two preprocessing files have no logical code change.

## Selected prediction replay

Current preprocessing rebuilt the recorded training and test matrices exactly:
328,135 x 297 and 109,379 x 297. The trusted saved checkpoint was replayed
through the normal `src.numpy_boosting_phase14` module. Maximum probability
difference is **zero**, and the normal submission writer reproduces the chosen
CSV byte for byte:

`d01c3949dea567dc9aa929ba232cbe77395d236b1c4a4492624e0bec5305e883`

The CSV has 109,379 unique IDs in official sample order, header `Id,Prediction`,
labels -1/+1, **13,895 positive** and **95,484 negative** predictions. Nine
input fingerprints and five numerical source fingerprints were checked.
This audit replays the completed full fit; it does not repeat expensive training.
Synthetic end-to-end tests also exercise training, calibration and CSV output.

## Checks actually performed

- **281 project tests passed**, with zero skips or failures, on an export of
  the exact staged Git blobs without datasets, caches or saved models.
- **26 public course tests passed** against that clean export. The commit URL
  format check is separate; the explicitly optional Black check is excluded.
- After the full suite, two existing report writers were made compatible with
  Python 3.9 by using `Path.open(..., newline=...)`. Their six existing tests
  passed in a targeted follow-up. No model calculation changed.
- `tools.verify_repository` passed: Python 3.9 syntax, JSON, local Markdown links,
  selected model/preprocessing/calibration agreement, frozen booster and all
  **114 historical indexed record hashes**. Calibration-only selection returns
  no invented nested score.
- Training, MLP and data-preparation help entry points work in the clean export.
- The 17 archived candidate CSV hashes agree with the aggregate evidence.
  Submission CSVs themselves remain local.
- `git diff --cached --check` passed. The final report compiles without text
  outside the margins; both rendered pages have been inspected.
- No matches were found for the private-key, GitHub/OpenAI token and AWS access
  key patterns scanned in Git-eligible text. This is a scoped pattern check.

[Machine-readable readiness](../results/eda/final_delivery_readiness_20261021.json)
records environment, scope and results. Earlier readiness JSONs remain dated
historical records and do not define the current model.

## Interpretation and remaining external steps

The exact combined pipeline has **no completed nested development evaluation**.
Its OOF F1 **0.444071** calibrates the threshold on the same labels. Historical
Phase 14 nested F1 **0.442155** belongs to a different pipeline. Reported AIcrowd
F1/accuracy are rounded team feedback, without independent platform retrieval.
Development and leaderboard feedback both contributed to model selection.

Local checks use **Python 3.14.3 / NumPy 2.4.2**. Python 3.9 syntax checking does
not replace full runtime execution. GitHub Actions is configured for
**Python 3.9 / NumPy 1.23.1**; check the workflow for the published commit.
The team must still submit that commit link and the PDF through the course
system. The [release guide](GITHUB_RELEASE.md) describes the hand-in format
and fresh-checkout reproduction. No AIcrowd upload or course hand-in is
performed by the repository push.

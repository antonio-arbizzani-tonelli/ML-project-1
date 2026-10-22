# Delivery integrity and recorded validation

This package organizes the existing report evidence and retains optional
procedures for repeating it. The selected model, required methods, numerical
sources and configuration/metadata are unchanged. The report has the editorial
clarifications described below; its scientific results are unchanged. The 114
historical execution records retain their original bytes and indexed hashes.

## Report detail clarification on 22 October 2026

The two-page report now specifies the MLP preprocessing, architecture, optimizer,
training settings and inner epoch selection. It also defines the linear baselines'
four age-product features on preprocessed channels and their training-only scaling.
Existing text was compacted to keep both pages; no result or model setting changed.

The saved LaTeX source was compiled to PDF and both rendered pages were inspected.
The current PDF/LaTeX fingerprints and their preceding hashes are recorded in the
pipeline integrity receipt. This was document compilation and editorial/static
review only: no local project tests, training or experiments were run. The numerical
checks below remain evidence for the earlier tested sources, not a new execution
of the report experiments.

## Cache and EDA instruction corrections on 22 October 2026

After the execution checks below, `run.py` was updated to require per-split
CSV/array SHA-256 receipts before reusing feature caches. Missing, partial,
modified or obsolete caches are rejected without deleting existing files.
The EDA instructions now distinguish the current descriptive-count command
from the detailed historical workflow and its archived prerequisites.

These changes were reviewed statically. No project commands, tests, training or
experiments were run for this correction. The pipeline integrity receipt records
the new loader hash and retains its earlier hash as historical provenance;
the checks below describe the earlier tested commit, not this loader revision.
Model algorithms, preprocessing, settings and historical records were unchanged
by that cache correction; the later report clarification is described above.

## Independent checks on 22 October 2026

Validation used a real shallow Git clone, initially containing only published
files and the four official CSVs, without previous models, numeric caches,
predictions or local source snapshots. The tested source commit is
[`f0741b0`](https://github.com/antonio-arbizzani-tonelli/ML-project-1/commit/f0741b0f137cf34bc5da0eef8c09aa7f32c0d14e).
The immediate delivery commit `0e161d7` only organized documentation and validation
receipts. The later cache-loader correction is described separately above.

- **271 project tests passed**, without failures, errors or skips (184.659 seconds),
  using Python 3.9.13 / NumPy 1.23.1 / Matplotlib 3.6.3 on Windows. Matplotlib's
  cache was placed inside the clone. One existing test fixture now explicitly
  uses int64 IDs, consistently with official caches, on both NumPy versions.
- **45/45 reduced workflows passed**, including a separate receipt check: the
  39 PDF studies and six supporting comparisons. The deterministic subset has
  5,999 training rows, 4,799 development rows and 100 test rows, with three trees,
  ten logistic updates or at most two neural epochs. Rare-column exclusions are recorded
  and apply only to reduced workflows.
- Seven independent negative checks rejected forged metrics, modified OOF
  predictions, modified numeric caches, changed inputs/configuration, crossed
  full/reduced scope and omitted required studies.
- The repository verifier passed: 114 indexed records, 133 configuration
  dependencies and 15 pipeline fingerprints at the tested commit. All 13
  protected files and all four official CSV hashes were independently checked.
  Help entry points and the Git whitespace check also passed.

The [compact validation receipt](../results/eda/report_reproduction_validation_20261022.json)
records these outcomes without machine-specific paths or model artifacts.

## Scientific coverage

The full-data descriptive counts were regenerated exactly: 328,135 labelled
rows, 109,379 test rows, 321 raw predictors, 28,975 positives, 47,175,779 blank
cells, 262,508 development rows and 295 cleaned predictors. The 1,095,421
nonresponse recodes, 899,687 zero recodes and HAREHAB1 counts (620 raw responses,
all positive; 607 valid responses) also match the recorded evidence.

Only **ridge completed a full scientific rerun**: F1 `0.41950783283563026`,
with all historical metrics matching within 1e-6 (maximum observed difference
about 1.4e-16). Independent checks confirmed official-label alignment, confusion
counts and exclusion of the reserved 20% partition.

Logistic regression and the Phase 18 calibration were interrupted at the user's
request. They have no completed result from this rerun. **The full numerical
reproduction of all 45 studies was not completed and is not claimed.** Reduced
workflow success establishes execution coverage, not reproduction of historical
scores. Existing report numbers and all original scientific records are retained.

The [optional reproduction guide](REPRODUCIBILITY.md) maps each study to its
configuration, backend and future command. No additional fitting is required
to inspect the delivered evidence. AIcrowd values remain external team-reported
feedback, not offline-reproduced scores. The selected combined pipeline still
has no completed independent nested evaluation; its OOF value is calibration.

## Earlier organization audit

The 21 October slimming audit checked the then-current 337-file package:
261 project tests, 26 public course tests, 95 configuration dependencies and
15 pipeline fingerprints passed. Those counts describe that historical package,
not the current test total. Its unchanged selected-model checkpoint replay is
documented in the [submission record](../results/eda/final_combined18_24_submission.json).

Official data, numeric caches, models, individual predictions, logs and submission
CSVs remain outside Git. GitHub Actions maintains the repository checks with
Python 3.9 / NumPy 1.23.1. A push does not submit to AIcrowd or the course system.

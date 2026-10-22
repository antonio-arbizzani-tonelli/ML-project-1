# Optional reproduction of the recorded report experiments

The repository organizes the existing results and preserves their original
records. The commands below are optional procedures for future execution.
During the 22 October audit, all 45 workflows passed reduced checks and only
the ridge comparison completed a full numerical rerun. Further fitting was
stopped at the user's request. The [delivery audit](FINAL_AUDIT.md) and its
[compact validation receipt](../results/eda/report_reproduction_validation_20261022.json)
record that scope; the remaining historical metrics are not newly revalidated.

The current report runner needs the published checkout, the four official CSVs
and the documented Python 3.9 / NumPy 1.23.1 environment. It does not read old
models, predictions, `.local` snapshots, Git history or historical preflight
files. Historical result JSONs supply comparison targets only.

Place the CSVs in `data/raw/dataset/` and run from the repository root:

```bash
python -m tools.reproduce_report list
python -m tools.reproduce_report data
python -m tools.reproduce_report run --experiment report --require-complete
python -m tools.reproduce_report check --require-complete
```

The [machine-readable catalog](../configs/report_experiments.json) maps every
experiment to its configuration, backend, historical target and command. It
contains 39 experiments cited in the two-page PDF, plus six additional comparisons
in the supporting notes. `--experiment report` runs the PDF set; `all` includes
the supporting comparisons. `historical` selects nested studies and `calibration`
selects the 18 full-training calibrations (the earlier Phase14 reference plus
the 17 later candidates). Every study can also run independently:

```bash
python -m tools.reproduce_report run --experiment ridge logistic
python -m tools.reproduce_report run --experiment two_seeds top100 top60
python -m tools.reproduce_report run --experiment weight1p25 weight1p5 weight2
python -m tools.reproduce_report run --experiment random3 hard3 random5 random7
python -m tools.reproduce_report run --experiment cal_combined18_24_features256
```

Use `--data-dir PATH` for official inputs elsewhere. `--output-dir PATH` chooses
a new directory inside this checkout; the default is `results/reproduction/report`.
Full and reduced checks always use separate `full/` and `smoke/` subdirectories.
No command writes to the historical ledger, selected configuration or report.

## Report-to-command map

| Report evidence | Experiment IDs |
| --- | --- |
| First nested-results table | `ridge`, `logistic`, `raw321`, `phase14`, `phase34`, `mlp64_32` |
| Activity one-hot, exact bins, restricted imputation, stronger L2 | `activity_one_hot`, `exact_bins`, `restricted_imputation`, `l2_5` |
| Positive weights and two-seed averaging | `weight1p25`, `weight1p5`, `weight2`, `two_seeds` |
| Selection of 100/60 columns | `top100`, `top60` |
| Random/hybrid3:1 and random5:1/7:1 sampling | `random3`, `hard3`, `random5`, `random7` |
| HAREHAB1 ablation | `harehab_ablation` |
| Full-training candidate table and three response-count candidates | All IDs beginning `cal_`; the catalog lists every candidate separately |
| Split, prevalence, blanks, cleaning counts and raw HAREHAB1 counts | `data` command; `data_claims.json` |
| Raw/clean and split-correction intervals/count differences | Automatically regenerated in `paired_comparisons.json` once their constituent studies finish |
| Selected output size, signs, ordering, positive count and hash | `cal_combined18_24_features256` and its generated CSV |
| AIcrowd scores | External team-reported feedback retained in Git; offline execution cannot reproduce leaderboard scoring |

## Preserved scientific procedures

Nested studies regenerate the original 80/20 split (seed20260918), three
development folds (seed20260919) and two inner threshold folds. The usual inner
seed is20260922 plus the outer-fold number; ridge uses20260923. The reserved20%
is not used to tune or score these comparisons. Training-only preprocessing,
imputation, selection and sampling are refitted within each fitting partition.

Linear studies explicitly use the historical median fill, including coded
categorical inputs, in a separate preprocessor. The maintained type-aware
preprocessor and selected submission runtime retain their current behavior.
Ridge keeps its inner-OOF score calibration. The two-seed study averages each
partition's probabilities before selecting its inner-OOF F1 threshold.

Feature selection repeats its80/20 selector split, training-only permutation
ranking and full-partition refit separately for every eligible outer and inner
partition. Top60 is contained in top100; the candidate-feature fraction is128/295.
The selector seed follows the original interleaved partition order. Sampling
instead preserves the original outer-first fitting order for its seed offsets.
Phase36 uses the corrected backend; Phase37 uses the frozen historical backend.

Calibrations fit three folds on all labelled rows, select the pooled OOF threshold
by F1 with accuracy tie breaking, then fit all labelled rows and generate a CSV.
The new runner constructs each candidate directly from its published plan and
parameters. Native-category and structural variants use their separate numerical
backends. A300-tree fresh fit follows the same training sequence as continuation
from200 trees. All intermediate models can be regenerated; no original tuning
manifest is required by this runner.

## Outputs, resuming and acceptance

`run.json` fingerprints published numerical/configuration sources, the four
official inputs, environment and execution mode. `data/receipt.json` fingerprints
the caches generated by this run. `fits/` contains parameter/row/matrix-specific
models and predictions; a completed fit is reused only when its fingerprints
match. Partial tree checkpoints resume the same training sequence. Changing the
fingerprinted sources, data, environment or smoke settings requires a new output
directory. The selected loader and CSV writer are separately protected by the
pipeline integrity receipt and repository verifier. Linear/MLP
studies can be rerun after interruption; completed tree fits are shared across
identical controls and the ensemble.

Each new record identifies its scope and original comparison target. Full runs
compare all available metrics with an absolute tolerance of1e-6, check calibration
thresholds and the original CSV hash/count when available. This tolerance detects
a single changed prediction in these datasets while allowing insignificant
floating-point differences in continuous metrics. A mismatch is recorded as
`different`; it is never converted into a successful reproduction or used to
rewrite historical results. The full checker recomputes metrics from generated
OOF arrays, checks generated files and verifies required coverage, descriptive
counts and paired comparisons. AIcrowd feedback is always marked external.

Full execution involves hundreds of fits and can take days. To check every
workflow quickly before running scientific comparisons:

```bash
python -m tools.reproduce_report run --smoke --experiment all --require-complete
python -m tools.reproduce_report check --smoke --require-complete
```

Smoke uses a deterministic subset of about 6,000 rows, three trees, ten logistic updates
and at most two MLP epochs. Its derived preprocessing plans exclude columns with fewer
than50 cleaned observations in that subset, recorded in `smoke_fixture.json`,
so rare questionnaire branches cannot become entirely missing inside tiny folds.
Feature limits are bounded by the resulting width. Imputation anchors, category
and response-count families still execute. **Smoke success verifies execution
and boundaries; it does not reproduce or validate the report's numerical results.**
Full mode never applies these reductions or exclusions. `validation.json`
explicitly distinguishes full scientific metrics from a reduced workflow check.

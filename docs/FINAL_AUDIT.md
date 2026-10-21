# Delivery integrity after repository slimming

The current tree is a smaller report-oriented package. The selected model,
`run.py`, required methods, five numerical sources, preprocessing/metadata and
report PDF/LaTeX are unchanged from the [archived release](ARCHIVE.md).
The [separate integrity receipt](../results/eda/final_pipeline_integrity.json)
checks those files and the compact final outcomes. Original experiment records
and their 114 indexed SHA-256 values remain unchanged.

Independent validation on 21 October 2026 used an export of the exact staged Git
blobs, without official datasets, caches, models or individual predictions:

- **261 project tests passed**, with no failures or skips (112.3 seconds).
- **26 public course tests passed**; the commit-URL format check is separate and
  the explicitly optional Black check is excluded.
- All five help entry points passed, along with the repository verifier:
  114 historical records, 95 configuration dependencies and 15 integrity fingerprints.
- All 13 protected files match the archived Git blobs exactly. The five numerical
  source hashes and every historical ledger record are unchanged.
- Six negative checks rejected changes to metadata, preprocessing recipes and
  numerical sources. Compact dataset evidence and its three source fingerprints
  were independently checked against the original records.

The maintained package contains 337 files, about 7.9 MiB of Git blobs, compared
with 550 files and 21.6 MiB in the archived tree. Older test totals belong to that
release and are not reused as counts for the smaller suite.

## Required checks

```bash
python -m tools.verify_repository
python -m unittest discover -s tests -q
python run.py --help
python -m tools.prepare_data --help
git diff --cached --check
```

The verifier includes Python 3.9 syntax, published local links, JSON, historical
ledger hashes, selected parameters/backend/calibration and 15 pipeline/report
fingerprints. It verifies every numerical `source_sha256` declared by the selected
submission record and the contents of preprocessing/metadata through the receipt.
Negative tests cover changed preprocessing/metadata, missing fingerprints and
unsupported normalization policies. Numerical sources and frozen recipes preserve
exact bytes; only explicitly Git-normalized text accepts CRLF/LF equivalence.

The CV runner's optional backend override leaves the default behavior unchanged,
records backend/source in models and results, and writes a backend-specific
artifact directory. It does not silently reuse models from the other search.

No model was retrained for this organization change. The prior selected-model
checkpoint replay and CSV identity check remain documented in the unchanged
[submission record](../results/eda/final_combined18_24_submission.json).
The report remains two pages. The exact combined pipeline has no completed
nested evaluation; calibration and team-reported AIcrowd feedback remain distinct.

CI runs the maintained checks with Python 3.9 /NumPy 1.23.1. Local checks use
Python 3.14.3 /NumPy 2.4.2. Official data, local models, probabilities and submission
CSVs are excluded. Hand in the PDF and the precise public Git commit URL through
the course system; a repository push does not submit to AIcrowd or the course.

# Development archive

The complete development tree is preserved at [commit `d491f15`](https://github.com/antonio-arbizzani-tonelli/ML-project-1/tree/d491f15211dd63d5a6888acae75e90a82b39ea4f).
The current tree retains the final pipeline, maintained tests, report comparisons,
compact evidence and the unchanged 114-record historical ledger.

Intermediate screening recipes, superseded orchestration and resume scripts,
large detailed EDA audits, preflights, partial rankings and repeated release
notes were removed from the current tree. The official data and ignored local
checkpoints were not changed.

To inspect a removed file without changing this checkout:

```bash
git show d491f15211dd63d5a6888acae75e90a82b39ea4f:path/to/file
```

For the full historical environment, use a separate clone checked out at that
commit. Historical JSON records are immutable evidence: paths, source fingerprints
and configurations inside them describe the original execution, not a guarantee
that every referenced file or local checkpoint exists in this smaller checkout.
The [configuration guide](../configs/README.md) distinguishes fresh execution
from replay requiring original local artifacts. Numerical sources and selected
configuration remain byte-identical to the archived release. The report PDF/LaTeX
subsequently gained MLP and age-interaction details without changing results;
their current and preceding hashes are recorded in the pipeline integrity receipt.
The `run.py` cache loader was subsequently hardened with CSV/array receipts;
its updated source hash is recorded separately in the current pipeline integrity
receipt. This administrative change has not been used to rerun historical results.

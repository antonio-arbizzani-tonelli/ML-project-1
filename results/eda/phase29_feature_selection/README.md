# Historical feature-selection comparison

The100/60-column candidates score nested F1 0.439224 / 0.437584 against historical
Phase 14's0.442155 and lose in every fold. The
[comparison JSON](comparison.json) and [study summary](../phase29_feature_selection_summary.md)
retain the result and protocol. This is not an evaluation of the final combined model.

The selector uses an 80/20 split inside each eligible training partition and
permutation importance on up to 10,000 rows, three permutations per column.
Reduced models are refitted on the complete training partition; their thresholds
use inner OOF predictions. Top 60 is contained in top 100.

Detailed rankings, partial verification output and group diagnostics were archived.
The [complete development commit](../../../docs/ARCHIVE.md) preserves them.
`tools.phase29_feature_selection` requires original Phase 14 models, OOF arrays,
control manifests and matching source snapshots. It is a local replay utility;
these prerequisites are not supplied by a source-only checkout.

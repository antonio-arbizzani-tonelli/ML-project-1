# Feature Registry Summary

The registry contains 322 columns including `Id`.

## Semantic types

| Type | Columns |
| --- | ---: |
| `binary` | 95 |
| `categorical` | 78 |
| `continuous` | 17 |
| `count` | 66 |
| `date` | 3 |
| `identifier` | 3 |
| `ordinal` | 45 |
| `survey_design` | 15 |

## Source kinds

| Source | Columns |
| --- | ---: |
| `administrative` | 26 |
| `core_survey` | 90 |
| `derived` | 91 |
| `optional_module` | 109 |
| `project_identifier` | 1 |
| `survey_design` | 5 |

## Leakage review

All supplied variables remain included in the competition feature set. The leakage labels are used for interpretation and controlled ablation.

| Risk | Columns |
| --- | ---: |
| `high` | 1 |
| `none` | 319 |
| `possible` | 2 |

High-risk variables:

- `HAREHAB1`: The question is asked about rehabilitation following a heart attack and is skipped when CVDINFR4 is not positive.

Questions for the teaching staff:

- `HAREHAB1`: Is this variable valid for competition use even though the question is asked only after a reported heart attack?

## Exact duplicate columns

- `_PSU` duplicates `SEQNO`.
- `CELLFON2` duplicates `CTELNUM1`.
- `_CPRACE` duplicates `_CRACE1`.

## Unresolved semantic types

No unresolved semantic types.

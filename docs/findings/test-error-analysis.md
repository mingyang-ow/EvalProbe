# Frozen TEST error-analysis protocol

Status: **ready for human review; 0/44 adjudications completed**.

## Bounded populations

| Review group | Selection | Items |
|---|---|---:|
| `WHOLE_DISAGREEMENTS` | All 11 official false positives and the single false negative | 12 |
| `LOCAL_REFERENCE_ONLY` | All official unsupported units missed by the local judge | 12 |
| `LOCAL_JUDGE_ONLY_SAMPLE` | Fixed sample from 71 judge-only units | 20 |

These groups must be summarized separately after review because only the judge-only population is
sampled. Human classifications are explanatory and never replace official RAGTruth metrics.

## Deterministic sampling

Seed `20260828` is applied to safe IDs and numeric/label metadata, never question, evidence, answer,
span text, or sentence text. A greedy selection first favors previously unrepresented coverage and
new records. Coverage dimensions are official whole reference, whole agreement/disagreement,
TRAIN-frozen burden stratum, locality, and whether the record has few or many judge-only units.
SHA-256 of seed, record ID, and sentence ID breaks ties. The algorithm permits at most two units per
record; this realized sample contains 20 records and 20 sources for 20 units.

The realized sample includes supported and unsupported official responses, whole agreements and
disagreements, low/medium/high unsupported burden, localized/distributed unsupported responses,
and both sparse and dense judge-only records. Selection occurred before reviewing corpus text.

## Review order

1. `WHOLE_DISAGREEMENTS`
2. `LOCAL_REFERENCE_ONLY`
3. `LOCAL_JUDGE_ONLY_SAMPLE`

Use the existing five-way taxonomy. The only whole unsupported miss carries a diagnostic priority
flag, but it remains one observation and should not be generalized. Stop after these 44 items unless
a separate methodological question is explicitly authorized.

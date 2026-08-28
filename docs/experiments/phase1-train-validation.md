# Phase 1: TRAIN methodology validation

These six-record results are development diagnostics, not scientific performance estimates.

## Baseline and audit

Phase 1A used three supported and three unsupported source-unique TRAIN records with independent
whole and sentence-v1 local requests. Local exact agreement was 2/6, with 8 false-positive and 2
false-negative sentence IDs. Human review of the broader 20-record sentence audit produced 14 PASS
and 6 FAIL; every failure was a detached list-marker segmentation defect.

Historical Phase 1A disagreement classifications were:

```text
JUDGE_ERROR 0
SEGMENTATION_DEFECT 3
REFERENCE_MAPPING_ARTIFACT 0
BENCHMARK_AMBIGUITY 8
RUBRIC_AMBIGUITY 0
```

## Deterministic repair and revalidation

Sentence-v2 merges formatting-only enumeration markers into the immediately following textual
unit without rewriting text or offsets. Corpus units changed 50,541 → 40,236 and marker-only units
10,305 → 0; all 2,927 annotation offsets remained exact. Independent review of the same 20 records
then produced 20 PASS and 0 FAIL.

The same six TRAIN records were rerun locally with unchanged model, reasoning, prompt, and labels.
Exact agreement remained 2/6; false positives changed 8 → 7 and false negatives 2 → 0. The two
record-12839 misses were artificial sentence-v1 marker units and mapped to `BOTH` under v2.

All seven current v2 disagreements were human-classified `BENCHMARK_AMBIGUITY`. No current
segmentation, reference-mapping, or rubric defect remained, which justified stopping methodology
tuning. Safe details are in the [sentence-v2 comparison](../../reports/phase1c/train-canary-segmentation-v2/comparison.md)
and [review summary](../../reports/review/review_summary.json).

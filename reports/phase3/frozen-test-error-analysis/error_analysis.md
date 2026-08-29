# Frozen TEST human error analysis

Status: **COMPLETE**. Reviewed: **44 / 44**.
Official Sol-versus-RAGTruth metrics remain unchanged; no human-corrected primary metric is created.

## Whole disagreements

| Official mismatch | Judge error | Benchmark ambiguity | Segmentation | Mapping | Rubric |
|---|---:|---:|---:|---:|---:|
| False positive (n=11) | 1 | 10 | 0 | 0 | 0 |
| False negative (n=1) | 1 | 0 | 0 | 0 | 0 |

- Apparent whole false positives: 10/11 benchmark ambiguity; 1 judge error; 0 segmentation defect.
- Whole false negatives: 1/1 judge error.

## Local reference-only misses

| Population | Judge error | Benchmark ambiguity | Segmentation | Mapping | Rubric |
|---|---:|---:|---:|---:|---:|
| Reference only (n=12) | 8 | 4 | 0 | 0 | 0 |

Of 12 official unsupported units missed by the local judge, 8 were classified as judge errors and 4 as benchmark ambiguity.

## Sampled local judge-only flags

| Population | Judge error | Benchmark ambiguity | Segmentation | Mapping | Rubric |
|---|---:|---:|---:|---:|---:|
| Judge-only sample (n=20) | 0 | 20 | 0 | 0 | 0 |

Of 20 sampled judge-only units, 20 were classified as benchmark ambiguity and 0 as a segmentation defect. This was a deterministic coverage sample, not a probability sample; its exact rate
must not be projected onto all 71 judge-only units.

## Interpretation boundaries

- The review does not support treating every apparent whole false positive or local
  judge-only flag as judge over-calling; benchmark incompleteness was the dominant human
  classification in both reviewed populations.
- Local judging also made genuine misses: 8/12 local reference-only items were classified as judge
  errors.
- The frozen burden counts (9/10, 10/10, 10/10) remain weak evidence for a burden effect.
- Granularity recovery remains 0/1 and therefore underpowered and inconclusive.
- No bounded TEST item remains classified as a segmentation, reference-mapping, or rubric defect.
  Phase 3 changes no frozen result.

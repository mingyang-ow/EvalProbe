# Frozen TEST findings

The frozen 60-response QA TEST pilot produced 48/60 whole-response agreement with official
RAGTruth. The judge detected 29/30 officially unsupported responses, while also marking 11/30
officially supported responses unsupported. Thus unsupported recall was high (29/30), but
unsupported precision was 29/40.

Detection was 9/10 in the low-burden stratum and 10/10 in both medium and high strata. These are
small descriptive strata; their Wilson intervals are recorded in the machine-readable analysis.

Sentence-v2 local exact-set agreement was 24/60. Across local sentence units, unsupported precision
was 60/131 and recall was 60/72. The single whole-response unsupported miss had no overlapping local
detection, so the prespecified granularity-recovery result was 0/1. This null result does not show
that local judging caused or prevented whole-response errors.

These are official benchmark comparisons, not human-corrected scores.

## What the benchmark score hid

Human review did not support interpreting all apparent false positives as judge over-calling. Ten
of 11 whole false positives were classified as plausible benchmark ambiguity, versus one judge
error. All 20 sampled local judge-only flags were classified
as benchmark ambiguity. Because this was a deterministic coverage sample rather than a probability
sample, 20/20 must not be projected to all 71 judge-only units.

The judge also made genuine misses: 8/12 local reference-only items and the single whole false
negative were classified as judge errors. The most defensible interpretation is therefore mixed:
official local precision is depressed by plausible annotation incompleteness, while local recall
still hides substantive judge misses.

The burden result remains weak or null at this scale, and granularity recovery remains
underpowered at 0/1. Human review does not create corrected headline metrics.

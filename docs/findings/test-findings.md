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

These are official benchmark comparisons, not human-corrected scores. Ninety-five disagreement
items await explanatory review, and no TEST adjudication has occurred. Claims about benchmark
ambiguity or judge error should wait for that review.

## Error analysis pending

The apparent whole false positives and local judge-only flags cannot yet be interpreted as judge
errors. Phase 3 therefore freezes a bounded 44-item review set before any text inspection: all
whole disagreements, all local reference-only misses, and a diverse deterministic sample of 20
local judge-only flags. Findings will be updated only after human review is complete.

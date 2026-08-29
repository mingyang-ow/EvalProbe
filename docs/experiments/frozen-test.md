# Frozen TEST experiment

## Pre-execution record

- Run ID: `frozen-test-v1`
- Frozen pilot: `reports/phase0/pilot_manifest.jsonl`
- Manifest SHA-256: `393f48b2f7a6cc6d4b7e9fc55e57f19ed87fdff4767041f8c5b4370c44595e11`
- Records: 60 TEST (30 supported, 30 unsupported; 10/10/10 unsupported burden strata)
- Requests: 60 whole + 60 sentence-v2 local = 120 maximum
- Judge: `gpt-5.6-sol`, low reasoning
- Prompts: `whole-grounding-v1`, `local-grounding-v1`
- Retry/fallback: 0/0
- Original Phase 2A hard cap: $1.50

## Pre-execution outcome

The complete dry-run planned all 120 independent requests and made zero network calls.

```text
Approximate input: 411,103 characters / 143,073 tokens
Conservative estimated maximum cost: $2.108292
Hard cap: $1.50
Cost gate: FAIL
```

All manifest-shape gates passed: 60 TEST records, 30/30 labels, 10/10/10 unsupported burden
strata, 60 unique record and source IDs, and the expected SHA-256. Sentence-v2, both prompt
versions, the model and reasoning effort, zero retry/fallback, documentation presence, safe DTOs,
and absence of any pre-existing TEST results were also verified. Ruff and the API-free test suite
passed.

Status: **BLOCKED BEFORE TEST**. No provider call was made and no `results.jsonl` exists. The
machine-readable dry-run is at
[`reports/phase2/frozen-test-v1/dry_run.json`](../../reports/phase2/frozen-test-v1/dry_run.json).

The cap and output allowances were not changed to force execution. Consequently there are no TEST
metrics, plots, or human-review queue. A future human decision is required before any new execution
authorization; it must not be inferred from this blocked pass.

## Phase 2B authorization

On 2026-08-28, the human operator explicitly increased only the hard cap to **$3.00** and
authorized the already frozen 120-call TEST plan. The sample, manifest hash, prompts, schemas,
model, reasoning effort, sentence-v2 methodology, reference construction, output allowances,
no-retry/no-fallback behavior, and no-tuning rule remain unchanged. The original $1.50 blocked
preflight above is retained as historical operational evidence.

The rerun passed every pre-execution gate with the same $2.108292 conservative maximum and made
zero network calls during validation. The managed execution approval guard then rejected the paid
corpus-derived data export before the first provider request. After the human operator supplied the
required direct confirmation, the exact frozen run proceeded without any methodology change.

## Execution outcome

- Provider attempts/completed calls: 120/120
- Operational failures: 0
- Input tokens: 89,798 (0 cached; 6,852 cache-write tokens reported)
- Output tokens: 11,918, including 9,578 reasoning tokens
- Total tokens: 101,716
- Estimated actual cost: $0.604404 against the $3.00 hard cap

## Official TEST metrics

Whole-response confusion counts were 19 supported→supported, 11 supported→unsupported,
1 unsupported→supported, and 29 unsupported→unsupported. Accuracy and balanced accuracy were both
48/60 = 0.800. Unsupported precision was 29/40 = 0.725 and recall was 29/30 = 0.967; the unsupported
false-negative rate was 1/30 = 0.033.

Unsupported detection by frozen burden stratum was 9/10 low, 10/10 medium, and 10/10 high. The
sentence-v2 local result had exact-set agreement on 24/60 responses, unsupported-unit precision of
60/131 = 0.458, and recall of 60/72 = 0.833, with 71 false-positive and 12 false-negative local
sentence units. The one whole-response unsupported miss was recovered by overlapping local
detection in 0/1 cases. Descriptively, whole detection was 13/14 for localized and 16/16 for
distributed unsupported responses.

The safe queue contains 95 items: 1 whole false negative, 11 whole false positives, 12 local
reference-only units, and 71 local judge-only units. Human TEST adjudication has not started;
official metrics remain Sol versus unchanged RAGTruth. Status: **TEST COMPLETE — READY FOR ERROR
ANALYSIS**.

## Phase 3 review preparation

Error-analysis setup made no provider calls and changed no frozen artifact or official metric. The
bounded human workload contains all 12 whole disagreements, all 12 local `REFERENCE_ONLY` units,
and 20 of the 71 local `JUDGE_ONLY` units, for 44 items total. The judge-only sample uses seed
`20260828` and safe metadata only. A deterministic greedy pass prioritizes new records and then
coverage of official reference class, whole agreement/disagreement, burden stratum, locality, and
per-record judge-only density; stable hashes break ties. The selected sample uses 20 unique records
and sources, with at most one selected unit from each in this realized set. At preparation time,
human adjudication had not started.

## Phase 3 review outcome

All 44 bounded items were reviewed. Among the 11 official whole false positives, classifications
were 10 `BENCHMARK_AMBIGUITY` and 1 `JUDGE_ERROR`; the single whole false
negative was `JUDGE_ERROR`. The 12 local reference-only misses comprised 8 `JUDGE_ERROR` and 4
`BENCHMARK_AMBIGUITY`. All 20 items in the local judge-only sample were classified
`BENCHMARK_AMBIGUITY`.

The previously inconsistent whole-view segmentation classification was human-re-reviewed and
changed to `BENCHMARK_AMBIGUITY`; no TEST item remains classified as a segmentation,
reference-mapping, or rubric defect. The judge-only sample was coverage-oriented rather than probabilistic, so 20/20 is not an
estimate for all 71 judge-only units. Official metrics, labels, predictions, and methodology remain
unchanged. Phase 3 used zero provider calls and $0 API spend.

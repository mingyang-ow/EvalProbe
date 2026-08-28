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
corpus-derived data export before the first provider request. No `results.jsonl` exists; calls,
usage, and cost remain zero. Status remains **BLOCKED BEFORE TEST** pending direct approval accepted
by the execution environment.

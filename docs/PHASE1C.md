# Phase 1C: deterministic local-unit repair

Phase 1C changes only local-unit construction. The judge model, prompt text, output schemas,
official RAGTruth labels, and span-overlap reference rule remain unchanged.

## Sentence-v2 rule

After the existing splitter runs, a unit is merged with its immediately following unit when all
of these conditions hold:

- the complete first unit is only a conservative numeric, alphabetic, parenthesized, or bullet
  list marker;
- the intervening source range contains only whitespace;
- the following unit is not another marker and contains alphabetic text.

The merged unit spans one contiguous range of the original answer. Text is not rewritten or
normalized. A terminal or consecutive marker is retained and emitted as a diagnostic warning.
The overlap rule remains: any RAGTruth character span touching a unit makes that unit
`UNSUPPORTED`.

## Safe artifacts

- `reports/phase1c/corpus_segmentation_diagnostics.json` contains aggregate TRAIN/TEST counts and
  span-validation statuses only.
- `reports/phase1c/phase0_v2_summary.json` contains IDs and numeric before/after mappings only.
- `reports/phase0/manual_audit_v2.jsonl` contains corpus text and is gitignored.
- `reports/phase1c/train-canary-segmentation-v2/` contains the safe dry-run, manifest, provider
  envelopes, and comparison report.

## Execution discipline

The Phase 1C configuration pins the six Phase 1A TRAIN record IDs, `gpt-5.6-sol` with low
reasoning, zero retries, zero fallbacks, six local calls, and a $0.25 cap. Phase 1A whole results
are reused only after their request hashes and contract metadata match a freshly rebuilt Phase 1A
plan. No TEST judge call is planned or authorized.

Human v2 adjudications are stored under `phase0-segmentation-v2` or
`train-canary-segmentation-v2`, preserving historical v1 decisions separately. Completion of the
new review round is required before freezing the methodology.

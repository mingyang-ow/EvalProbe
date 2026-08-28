# Phase 1A judge contract

Phase 1A validates one judge before any frozen TEST evaluation. It uses `gpt-5.6-sol` with low
reasoning through the OpenAI Responses API, no tools, no fallback model, no automatic retries, and
no shared conversation state.

## Semantic contracts

The whole-response judge asks whether every substantive factual assertion is supported by, or
directly and reasonably inferable from, supplied evidence. Any contradiction or unsupported added
information makes the response `UNSUPPORTED`, even if outside knowledge says it is true. Harmless
paraphrases are allowed.

The strict model-owned output is:

```json
{"verdict": "SUPPORTED"}
```

or the corresponding `UNSUPPORTED` verdict.

The local judge receives the same question/evidence with deterministic numbered Phase 0 sentences.
It returns only:

```json
{"unsupported_sentence_ids": [2, 5]}
```

The application rejects duplicate or unknown IDs and derives sentence verdicts. No rationale or
chain-of-thought is requested.

Exact versioned instructions live in `src/evalprobe/phase1/prompts.py`; JSON schemas and safe DTOs
live in `src/evalprobe/phase1/contracts.py`. Static instructions and dynamic corpus content remain
separate.

## Operational contract

Each of six source-unique TRAIN records receives one independent whole request and one independent
local request. Provider status, refusal, incomplete output, invalid JSON, contract failures, and
budget guards remain distinct from semantic disagreement.

Safe JSONL results are flushed after each call. Resume matches completion and input hash, so a
completed unchanged request is not paid for twice. Corpus-bearing requests and provider debug
artifacts are never persisted in tracked reports.

Pricing is application-owned configuration with an effective date and source. Cost uses
provider-reported input, cached-input, cache-write, output, and reasoning breakdowns; reasoning is
already included in output tokens and is not double-counted. Missing usage stays missing.

## Stage gate

The Phase 0 manual sentence-conversion audit has no recorded human approval. That approval and
human review of the TRAIN canary remain prerequisites before the frozen TEST pilot.

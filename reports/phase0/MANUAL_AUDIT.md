# Manual span-to-sentence conversion audit

Running `uv run evalprobe phase0 audit` creates `manual_audit.jsonl` beside this file. The artifact is deterministic (seed `20260828`), text-bearing, and therefore gitignored.

It samples 20 eligible QA responses and attempts to cover supported, one-span, multiple-span, low-burden, high-burden, localized, distributed, and `implicit_true` cases when available. Each row contains source offsets, sentence labels, and empty human-review fields.

Review each row locally and record whether:

1. annotation offsets identify the intended response text;
2. deterministic sentence boundaries are sensible;
3. every sentence touched by a hallucination span is marked unsupported;
4. locality agrees with the number of affected sentences.

This is a human audit. No LLM approval is used.

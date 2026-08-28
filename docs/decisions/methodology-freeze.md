# Methodology freeze

Status: frozen before TEST. Baseline TRAIN checkpoint: Git commit `6410f39`. Frozen TEST run ID:
`frozen-test-v1`.

| Component | Frozen decision |
|---|---|
| Dataset | RAGTruth, `good` QA only |
| Official reference | Unchanged RAGTruth annotations; `implicit_true=true` remains unsupported |
| Whole construction | Zero annotations → `SUPPORTED`; otherwise `UNSUPPORTED` |
| TEST sampling | Existing 60-row manifest; 30/30 labels; 10/10/10 unsupported burden strata; unique sources |
| Frozen manifest | `reports/phase0/pilot_manifest.jsonl` |
| Manifest SHA-256 | `393f48b2f7a6cc6d4b7e9fc55e57f19ed87fdff4767041f8c5b4370c44595e11` |
| Burden | Union span coverage / response characters; TRAIN-derived tertiles |
| Local units | `sentence-v2` deterministic list-marker repair |
| Local reference | Any span overlap makes the unit unsupported |
| Judge | `gpt-5.6-sol`; `reasoning.effort=low` |
| Prompts | `whole-grounding-v1`; `local-grounding-v1` |
| Outputs | Whole verdict object; local unsupported-sentence-ID object |
| Requests | Independent, no tools, no shared state, provider storage disabled |
| Failures | Persist operational status and available usage; fail fast; no fabricated semantics or usage |
| Retry/fallback | Zero automatic retries; zero fallback models |
| Budget | Hard cap $1.50; at most 120 semantic calls |
| Human review | Explanatory only; never rewrites benchmark labels or primary metrics |

## No-Tuning rule

From the first TEST request, prompts, segmentation, span mapping, thresholds, sample, judge,
reasoning effort, schemas, and reference construction cannot change based on TEST performance. A
genuine implementation defect stops the run and requires a human decision. Semantic disagreement,
poor performance, and surprising results are not implementation defects.

RAGTruth predates this judge, so training exposure cannot be ruled out. Exact annotations are a
benchmark reference rather than infallible truth; TRAIN review found plausible omissions. Results
are QA-only, use one judge, and have limited power at n=60.

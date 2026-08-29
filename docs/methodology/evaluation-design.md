# Evaluation design

## Research question

Does an LLM judge miss a small unsupported claim inside an otherwise grounded RAG answer, and can
local evidence judging find unsupported content when whole-response judging approves the answer?

## Dataset and references

EvalProbe uses RAGTruth `good` QA responses only. Official annotations are not rewritten.

- Whole response: no annotation is `SUPPORTED`; one or more annotations is `UNSUPPORTED`.
- `implicit_true=true` remains unsupported because grounding is evaluated only against supplied
  evidence.
- Local: a sentence-v2 unit is `UNSUPPORTED` when any official span overlaps its source offsets.

Human adjudication explains disagreements but never replaces benchmark metrics.

## Reference hierarchy

EvalProbe keeps three distinct kinds of evidence:

```text
official RAGTruth reference ≠ human adjudication ≠ judge prediction
```

Official performance always compares the frozen judge prediction with the unchanged RAGTruth
reference. Human adjudication asks why the two disagree; it is explanatory error analysis, not a
replacement gold label and not a basis for a human-corrected headline metric.

## Frozen sample and burden

The TEST pilot contains 60 source-unique responses: 30 supported and 30 unsupported. Unsupported
responses are split 10/10/10 across low, medium, and high burden using thresholds derived only from
eligible TRAIN unsupported responses.

Hallucination burden is:

```text
characters covered by the union of hallucination spans / response character count
```

The frozen thresholds are low ≤ 0.10594594594594595 and medium ≤ 0.24954296160877515.

## Judge views

Whole and local requests are independent. Whole owns only a `SUPPORTED | UNSUPPORTED` verdict.
Local owns only an array of unsupported sentence-v2 IDs. The application owns identifiers,
references, versions, usage, cost, and operational status. No rationale or chain-of-thought is
requested.

Whole-response performance is primary. Secondary analyses cover unsupported detection by burden,
local exact-set and sentence-unit precision/recall, and whether local judging catches a reference
unsupported unit when whole judging misses the response.

The 60-record QA pilot and single judge limit generalization. Sentence units are analysis units,
not atomic semantic claims.

## Deterministic preprocessing lesson

The original `sentence-v1` audit passed 14/20 records and failed 6/20. Standalone numbered and list
markers could become separate units and inherit `UNSUPPORTED` through span overlap, manufacturing
apparent local judge misses. Corpus diagnostics counted 10,309 suspicious units before repair and
4 after it; formatting-marker-only units fell from 10,305 to zero.

`sentence-v2` deterministically merges each formatting-only marker into the immediately following
textual unit without rewriting response text or source offsets. A fresh audit passed 20/20 records,
while all 2,927 RAGTruth annotation spans still matched exactly. This is a core evaluation lesson:
preprocessing can create evaluator errors even when both the benchmark annotation and judge output
are faithfully recorded.

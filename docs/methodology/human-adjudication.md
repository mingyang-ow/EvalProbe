# Human adjudication

Human review diagnoses evaluation behavior without changing official RAGTruth labels or primary
metrics. Persisted decisions contain identifiers, classification, concise notes, reviewer metadata,
and timestamps—not questions, passages, answers, prompts, or annotation text.

The hierarchy is explicit: RAGTruth supplies the official benchmark reference, the judge supplies
the prediction, and the human reviewer supplies an explanatory diagnosis. None is silently promoted
to corrected ground truth. All reported accuracy, precision, recall, and confusion counts remain
judge versus RAGTruth.

## Classification taxonomy

- `JUDGE_ERROR`: the unit and reference are sensible, evidence supports the content, and the judge
  incorrectly flags it.
- `SEGMENTATION_DEFECT`: the deterministic unit itself is inappropriate.
- `REFERENCE_MAPPING_ARTIFACT`: the unit is sensible but span overlap gives a misleading reference.
- `BENCHMARK_AMBIGUITY`: the judge's unsupported reading is defensible but RAGTruth has no matching
  annotation.
- `RUBRIC_AMBIGUITY`: multiple readings arise specifically from unclear judge instructions.

Sentence-v1 review found six segmentation failures. An independent sentence-v2 review passed all
20 items. Historical TRAIN judge disagreements contained 3 segmentation defects and 8 benchmark
ambiguities. After repair, all 7 current sentence-v2 disagreements were benchmark ambiguities;
none revealed a new segmentation, mapping, or rubric defect.

TEST metrics must be generated before qualitative TEST review. Later review should focus on
informative errors and remain a separate explanatory layer.

The bounded TEST review completed 44/44 items. Whole false positives comprised 10 benchmark
ambiguities and 1 judge error; the sole whole false negative was a judge error. Local
reference-only misses comprised 8 judge errors and 4 benchmark ambiguities. All 20 deterministically
sampled local judge-only flags were benchmark ambiguities. Because only 20/71 judge-only units were
reviewed through a coverage sample, that result is descriptive of the sample and is not extrapolated
to the full population.

See [Review Console operations](../REVIEW_CONSOLE.md) and the safe summary at
`reports/review/review_summary.json`.

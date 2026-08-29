# EvalProbe interview guide

## 20-second explanation

EvalProbe tests whether an LLM judge can reliably detect unsupported claims in grounded RAG
answers. I froze a 60-response RAGTruth TEST pilot, evaluated whole responses and deterministic
sentence units with GPT-5.6 Sol, and then human-reviewed bounded disagreement populations. The
judge reached 96.7% unsupported recall, but error analysis showed the more important lesson:
benchmark disagreement is not automatically judge error, and evaluator preprocessing can create
fake failures.

## Two-minute explanation

RAG evaluation often treats either a benchmark or an LLM judge as ground truth. I wanted to test a
specific hypothesis: whether whole-response judging misses a small unsupported claim that local
sentence judging can recover. I used RAGTruth because it provides naturally occurring RAG answers
with human hallucination spans, then built independent whole-response and local reference views.

The first local method, `sentence-v1`, exposed an evaluation bug: standalone list markers became
separate units, so span overlap could label punctuation-like fragments as unsupported. I repaired
the deterministic splitter as `sentence-v2`, preserving all 2,927 exact span matches while
eliminating 10,305 marker-only units. Its fresh human audit improved from 14/20 PASS to 20/20 PASS.
I then froze the methodology before TEST.

On the 60-response frozen TEST pilot, GPT-5.6 Sol achieved 80.0% whole-response accuracy and 96.7%
unsupported recall. Local judging had 83.3% unsupported-unit recall but only 45.8% official
precision and 40.0% exact sentence-set agreement. Human review made those numbers interpretable:
10 of 11 whole false positives were plausible benchmark ambiguities, while 8 of 12 local
reference-only misses were genuine judge errors. All 20 sampled local judge-only units were
benchmark ambiguities, but that was a purposive 20/71 sample and is not a population estimate.

The original burden effect was weak: detection was 9/10, 10/10, and 10/10 across low, medium, and
high strata. There was only one whole unsupported miss, and local judging recovered 0/1, so the
planned granularity comparison was underpowered. I kept those null and inconclusive findings. The
portfolio contribution is therefore the evaluation system and its disciplined distinction among
model error, benchmark ambiguity, and methodology defects—not a claim that local judging won.

## Likely questions

### Why RAGTruth?

It offers real RAG outputs, source evidence, response-level labels, and exact human hallucination
spans. That combination supports both whole and local evaluation without creating synthetic
errors. I still treat its annotations as references rather than unquestionable truth.

### Why only one judge?

The project isolates methodology and failure analysis with a tightly controlled pilot. One frozen
judge avoids turning a small study into an underpowered model leaderboard. A multi-model,
multi-provider replication is future work.

### Why GPT-5.6 Sol?

It was the predeclared judge for this run, used with low reasoning effort and versioned prompts.
The important design choice is that the model and settings were frozen before TEST, not that this
single model represents all LLM judges.

### Why sentences instead of atomic claims?

Sentences are deterministic, auditable, and can be mapped exactly to character-span references.
They are analysis units, not guaranteed atomic semantic claims. Claim decomposition would add
another model-dependent stage and its own error surface.

### Why not correct RAGTruth after human review?

That would silently change the target after seeing model outputs. Official metrics remain Sol
versus unchanged RAGTruth; human classifications explain disagreements separately.

### Why only 60 TEST responses?

This is a bounded portfolio pilot with 30 supported and 30 unsupported examples, chosen before
execution and protected by a cost cap. It is large enough to exercise the system, but uncertainty
and rare-error denominators remain important limitations.

### Why no second provider?

Provider comparison was outside the frozen scope and budget. Adding one after seeing TEST results
would also change the study. Cross-provider replication is an explicit next step.

### Could the model have seen RAGTruth during training?

Yes. Training-data exposure cannot be ruled out, so the results should not be interpreted as a
clean generalization benchmark.

### How did you prevent evaluation leakage?

The judge received purpose-built DTOs containing only the evidence, answer, and unit needed for
the assigned view. It never received RAGTruth labels, hallucination spans, burden strata, expected
answers, or human adjudications. Tests enforce those field boundaries, and whole and local calls
were independent.

### What failed in sentence-v1?

The splitter detached standalone list markers from their following text. Because local references
were assigned by span overlap, those artifacts could receive unsupported labels and manufacture
apparent disagreements. Sentence-v2 deterministically merges those markers.

### Was sentence-v2 tuned to GPT-5.6 Sol's mistakes?

No. The repair targeted a structurally invalid unit pattern found in human segmentation audit,
not a semantic prediction pattern. It was validated on TRAIN and frozen before the paid TEST run.

### What does 45.8% local precision mean?

Against unchanged RAGTruth sentence references, 60 of the 131 units the judge called unsupported
were reference positives. It is the official benchmark precision, not proof that all other 71
calls were wrong; the bounded human sample found plausible benchmark ambiguity in 20/20 sampled
judge-only units.

### Can you claim the benchmark is incomplete?

Only cautiously. Human review found substantial plausible ambiguity in the inspected disagreement
sets. The 20 local judge-only cases were purposively sampled, so I do not extrapolate a corrected
precision or a population-wide incompleteness rate.

### What is unsupported burden?

It is a frozen proxy for how much of an answer is covered by annotated unsupported spans, used to
stratify the 30 unsupported responses into low, medium, and high groups. It is descriptive, not a
claim about reader-perceived difficulty.

### Why wasn't local judging better?

Whole judging already missed only one unsupported response, leaving a denominator of one for
recovery. Local judging recovered 0/1 and created many sentence-level disagreements, so the
planned advantage was neither demonstrated nor meaningfully testable in this pilot.

### What would you do next?

I would follow the preregistered replication roadmap below while retaining the separation between
frozen official metrics and post hoc explanatory analysis.

## Future work

Preregister a larger, multi-domain sample designed to yield more whole false negatives; add
blinded duplicate adjudication and inter-rater agreement; compare sentence units with a validated
claim-decomposition method; and replicate across judges and providers. These are proposed
extensions, not experiments completed in the current project.

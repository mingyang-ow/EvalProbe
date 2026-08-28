# EvalProbe project story

EvalProbe asks whether an LLM judge misses small unsupported claims inside otherwise grounded RAG
answers. The project deliberately stays narrow: RAGTruth QA, one judge, one frozen TEST pilot, and
whole-versus-local grounding judgments.

## From question to frozen pilot

**Observation:** RAGTruth provides human character-span hallucination annotations on natural RAG
responses. **Decision:** use its `good` QA responses and treat any annotation, including
`implicit_true=true`, as unsupported against the supplied evidence. **Reason:** exact spans support
both benchmark references and a deterministic hallucination-burden measure without an LLM claim
extractor. **Consequence:** Phase 0 audited schema, provenance, licensing, offsets, and quality
values before selecting data.

**Observation:** all 2,927 eligible QA annotations matched their recorded text offsets, with no
malformed or out-of-range spans. **Decision:** freeze a source-unique 60-response TEST pilot: 30
supported and 30 unsupported, with 10 unsupported responses in each TRAIN-derived burden tertile.
**Reason:** the TEST set must not choose thresholds or samples after judge behavior is known.
**Consequence:** `reports/phase0/pilot_manifest.jsonl` became immutable input to the final run.

## Judge contract and unexpected TRAIN behavior

**Observation:** whole and local judging require different semantic outputs but the same grounding
rubric. **Decision:** pin `gpt-5.6-sol` at low reasoning with independent Responses API requests,
`whole-grounding-v1`, `local-grounding-v1`, strict structured outputs, no tools, no fallback, and
no automatic retries. **Reason:** the experiment should isolate granularity, not model or prompt
variation. **Consequence:** a six-record TRAIN canary tested contracts and operations before TEST.

**Observation:** sentence-v1 local exact agreement was only 2/6, with eight false-positive and two
false-negative sentence IDs. **Decision:** build a local human Review Console rather than tune the
judge. **Reason:** poor agreement could reflect preprocessing, reference mapping, rubric ambiguity,
benchmark ambiguity, or judge error. **Consequence:** human review became an explanatory layer
while official RAGTruth labels remained unchanged.

## Repairing the evaluation pipeline

**Observation:** standalone list markers such as `1.` could inherit `UNSUPPORTED` labels even
though they were not independently judgeable. **Decision:** merge deterministic formatting-only
markers into the immediately following textual unit. **Reason:** detached markers could manufacture
false-negative evaluator errors. **Consequence:** sentence-v2 replaced sentence-v1 before TEST;
the change removed a demonstrated pipeline defect rather than optimizing a score.

**Observation:** corpus-wide units changed from 50,541 to 40,236 and formatting-marker-only units
from 10,305 to zero, while all 2,927 span matches remained exact. **Decision:** independently review
the same 20 Phase 0 examples under sentence-v2. **Reason:** a deterministic repair still required
human validation. **Consequence:** sentence-v2 passed 20/20 reviews, versus 14 PASS and 6 FAIL for
sentence-v1.

**Observation:** rerunning the same six TRAIN records produced 2/6 exact agreement, seven false
positives, and zero false negatives. The two old record-12839 misses disappeared. **Decision:**
adjudicate only the seven current sentence-v2 disagreements. **Reason:** historical defects that
mapped to `BOTH` were no longer current disagreements. **Consequence:** all seven current items
were classified `BENCHMARK_AMBIGUITY`; no segmentation, reference-mapping, or rubric defect
remained.

## Methodology freeze and TEST authorization

**Observation:** further TRAIN tuning had no identified methodological target. **Decision:** freeze
dataset, references, sample, burden thresholds, sentence-v2, span overlap, judge, reasoning effort,
prompts, schemas, and operational behavior. **Reason:** TEST performance must not change the
experiment that produced it. **Consequence:** the frozen 120-call TEST run initially required every
pre-execution gate—including the $1.50 conservative cost cap—to pass. If a gate failed, no paid call
was made and the blocked state was preserved as project history.

**Observation:** the complete zero-network planner estimated a conservative maximum cost of
$2.108292 for the unchanged 120 requests, above the $1.50 cap. **Decision:** stop before the first
TEST request. **Reason:** increasing the cap or shrinking the existing output allowances would
change a frozen operational constraint merely to force execution. **Consequence:** Phase 2 is
`BLOCKED BEFORE TEST`; the manifest and methodology remain frozen and no TEST result exists.

**Observation:** the human operator subsequently authorized a $3.00 cap as a budget-only change.
**Decision:** retain the frozen methodology and rerun the complete preflight unchanged. **Reason:**
the $2.108292 conservative maximum now fits within explicit authorization. **Consequence:** every
zero-call gate passed, but the execution environment rejected the paid data-export action before
the first provider request. TEST remains `BLOCKED BEFORE TEST`; no semantics, usage, or cost were
fabricated.

Navigation and stable definitions live in [docs/INDEX.md](docs/INDEX.md).

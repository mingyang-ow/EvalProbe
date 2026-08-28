# Human review console

EvalProbe treats both judge predictions and benchmark references as evidence to inspect rather
than unquestioned truth. The local review console separates experiment execution from human
adjudication and supports a repeatable workflow for diagnosing evaluator errors, segmentation
defects, reference-mapping artifacts, benchmark ambiguity, and rubric ambiguity.

## Start the console

From the repository root:

```bash
uv run streamlit run src/evalprobe/review/app.py
```

For Phase 3, choose `Judge disagreement review`, `sentence-v2`, and `Frozen TEST`. Then work through
the `Phase 3 review group` selector in this order: `WHOLE_DISAGREEMENTS`,
`LOCAL_REFERENCE_ONLY`, and `LOCAL_JUDGE_ONLY_SAMPLE`. Progress is shown separately for each group;
the console never calls a judge.

The app does not accept an API key and contains no judge-execution action.

## Review modes

- **Sentence audit** loads the existing 20-example Phase 0 manual audit. Reviewers inspect exact
  spans, deterministic sentence units, reference mapping, locality, and burden before choosing
  `PASS`, `PASS_WITH_LIMITATION`, or `FAIL`.
- **Judge disagreement review** loads the Phase 1A TRAIN canary. Its queue contains the whole
  disagreement and each individual local false-positive or false-negative sentence ID. Reviewers
  must select exactly one human classification.
- **Record inspector** displays any canary record/view, including agreements, using run, source,
  record, and view identifiers. It is inspection-only.

Sentence units that are number-only, punctuation-only, list-marker-only, alphabetic-free, or very
short formatting tokens receive a deterministic warning. Warnings are diagnostic signals, not
automatic classifications, and preprocessing is not changed.

## Human classification taxonomy

- `JUDGE_ERROR`: the reference/unit is sensible and the judge prediction is incorrect.
- `SEGMENTATION_DEFECT`: the splitter created an inappropriate local semantic unit.
- `REFERENCE_MAPPING_ARTIFACT`: segmentation is reasonable, but span overlap yields a misleading
  sentence reference.
- `BENCHMARK_AMBIGUITY`: the official annotation is debatable, incomplete, or boundary-sensitive.
- `RUBRIC_AMBIGUITY`: the current rubric reasonably permits the model interpretation.

The console records a concise optional note but asks reviewers not to copy large source passages.
It never relabels RAGTruth or changes prompts automatically.

## Persistence and licensing

Questions, passages, answers, annotations, and span text are loaded at runtime from local,
gitignored artifacts. They are never included in the adjudication schema.

Human decisions are stored idempotently in:

```text
reports/review/adjudications.jsonl
```

The stable identity is derived from `run_id`, `record_id`, `view`, and optional `sentence_id`.
Saving the same item replaces its existing decision. The tracked-safe schema contains only IDs,
review type/status, human classification, a concise note, reviewer marker, and timestamp.

Generate a safe classification summary after reviewing:

```bash
uv run evalprobe review summary
```

Recompute text-free suspicious-unit counts across eligible QA TRAIN and TEST responses with:

```bash
uv run evalprobe review diagnostics
```

Neither command uses judge results to change preprocessing or makes a methodological decision.

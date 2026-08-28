# Phase 0 — RAGTruth audit and frozen pilot

## Dataset

- Source records: 2965; response records: 17790.
- Task types (sources): `{'Data2txt': 1033, 'QA': 989, 'Summary': 943}`.
- Split values (all responses): `{'test': 2700, 'train': 15090}`.
- QA sources: 989; QA responses: 5934; raw split counts: `{'test': 900, 'train': 5034}`.
- QA upstream source values: `{'MARCO': 989}`.
- QA quality values: `{'good': 5767, 'incorrect_refusal': 143, 'truncated': 24}`. Only `good` is eligible; `incorrect_refusal` and `truncated` are excluded.
- Eligible QA responses: 5767; split counts: `{'test': 875, 'train': 4892}`; reference counts: `{'SUPPORTED': 4061, 'UNSUPPORTED': 1706}`; split/reference counts: `{'test:SUPPORTED': 715, 'test:UNSUPPORTED': 160, 'train:SUPPORTED': 3346, 'train:UNSUPPORTED': 1546}`.
- Raw QA generator distribution: `{'gpt-3.5-turbo-0613': 989, 'gpt-4-0613': 989, 'llama-2-13b-chat': 989, 'llama-2-70b-chat': 989, 'llama-2-7b-chat': 989, 'mistral-7B-instruct': 989}`.
- Eligible generator distribution: `{'gpt-3.5-turbo-0613': 930, 'gpt-4-0613': 961, 'llama-2-13b-chat': 984, 'llama-2-70b-chat': 982, 'llama-2-7b-chat': 971, 'mistral-7B-instruct': 939}`.
- Raw QA responses-per-source distribution: `{'6': 989}`.
- Eligible QA responses-per-source distribution: `{'1': 1, '2': 1, '3': 4, '4': 27, '5': 92, '6': 864}`.

Whole-response references are constructed only after quality filtering: no annotations means `SUPPORTED`; one or more annotations means `UNSUPPORTED`. An annotation with `implicit_true=true` still means unsupported because this experiment measures grounding in supplied evidence, not truth under outside knowledge.

## Annotation audit

- QA annotations checked: 2927.
- Span validation: `{'malformed': 0, 'matched': 2927, 'mismatch': 0, 'out_of_range': 0}`; total failures: 0.
- Hallucination label types: `{'Evident Baseless Info': 1562, 'Evident Conflict': 423, 'Subtle Baseless Info': 893, 'Subtle Conflict': 49}`.
- `implicit_true` values: `{'False': 2081, 'True': 846}`.
- Malformed/unexpected record issues: 0; duplicate source IDs: 0; duplicate response IDs: 0.

Any mismatch, malformed span, or out-of-range span is retained as a visible failure and blocks readiness; offsets are never repaired.

## Derived characteristics

- Unsupported response burden (union span characters / response characters): n=1706, min=0.000893, p25=0.081230, median=0.170433, mean=0.218740, p75=0.296534, max=1.000000.
- Locality: `{'DISTRIBUTED': 960, 'LOCALIZED': 746, 'NONE': 4061}`.
- Sentence-count distribution: n=5767, min=1.000000, p25=4.000000, median=7.000000, mean=8.763829, p75=13.000000, max=50.000000.
- Affected-sentence distribution among unsupported responses: n=1706, min=1.000000, p25=1.000000, median=2.000000, mean=2.794256, p75=3.000000, max=29.000000.
- Sentences use the deterministic `evalprobe_rule_v1` punctuation/newline segmenter with exact source offsets. Any overlapping hallucination span makes a sentence unsupported; spans touching two or more sentences are distributed.

## Pilot

- Tertile source: eligible QA train unsupported responses only.
- Train-derived thresholds: low ≤ 0.1059459459; medium ≤ 0.2495429616; high above that.
- Selected: 60; reference counts: `{'SUPPORTED': 30, 'UNSUPPORTED': 30}`; unsupported strata: `{'high': 10, 'low': 10, 'medium': 10}`.
- Unique source IDs: 60; maximum responses per source: 1.

## Risks and limitations

- RAGTruth predates the future judge, but benchmark exposure or contamination cannot be ruled out.
- Deterministic sentence boundaries are analysis units, not semantic claims.
- Human exact-span annotations are benchmark references, not infallible truth.
- QA material includes upstream MS MARCO passages. Raw and text-bearing artifacts remain local and are not redistributed here.

## Phase 0 decision

**READY FOR PHASE 1.**

The QA audit has no unresolved integrity failures and the 60-response pilot satisfies the frozen balance, train-only threshold, and source-uniqueness rules.

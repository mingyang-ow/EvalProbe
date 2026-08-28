# Judge-input leakage controls

Phase 1A constructs judge inputs through an explicit allowlist containing only:

- question;
- retrieved evidence/passages;
- answer.

Never include human hallucination labels, span offsets or text, annotation comments, `implicit_true`, reference verdicts, generator/model identity, temperature, response quality, burden stratum, locality, split, or other benchmark metadata.

Reference and judge-input construction use separate dataclasses. The safe DTOs cannot hold record
IDs, reference labels, spans, burden, split, generator metadata, or other evaluation fields. Whole
and local payloads are created independently, and neither receives a previous response ID.

Manifests and result envelopes are evaluation metadata and are never interpolated into judge
prompts. Tests inspect the allowlist, forbidden keys, independent payloads, and zero-network dry run.

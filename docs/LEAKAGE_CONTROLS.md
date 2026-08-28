# Judge-input leakage controls

Phase 0 does not implement or call an LLM judge. For the future judge harness, construct inputs through an explicit allowlist containing only:

- question;
- retrieved evidence/passages;
- answer.

Never include human hallucination labels, span offsets or text, annotation comments, `implicit_true`, reference verdicts, generator/model identity, temperature, response quality, burden stratum, locality, split, or other benchmark metadata.

Reference construction and judge-input construction must remain separate code paths. The pilot manifest is evaluation metadata and must never be interpolated into a judge prompt.

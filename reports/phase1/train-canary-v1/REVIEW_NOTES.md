# Phase 1A canary review notes

This is a six-record TRAIN development diagnostic, not a performance estimate.

## Contract outcome

- All 12 final whole/local contracts completed with valid Structured Outputs.
- The first `local-output-v1` attempt was rejected because its schema used an unsupported
  `uniqueItems` keyword. The failure is preserved in the ledger with no reported usage.
- Uniqueness was already application-validated. The schema was corrected and versioned as
  `local-output-v2`; prompt text was not changed.
- Resume skipped the previously completed whole request.

## Methodological review

- Record `12839` exposes standalone list-number sentence units. Human annotation spans overlap
  two of these formatting-only units, which makes exact local agreement methodologically
  questionable even though those units contain no substantive claim.
- Additional judge/reference disagreements on records `12009`, `13899`, and `16875` appear
  compatible with annotation-boundary or annotation-completeness risk rather than an obvious
  grounding-rubric defect.
- No unsupported-reference record produced the target granularity pattern in this canary.
- The prompt was not tuned or rerun to force agreement.

The frozen Phase 0 methodology has not been changed. Its manual sentence-conversion review remains
unapproved, and the sentence-unit issue should be resolved by a human methodological decision
before the TEST pilot.

## Phase status

**BLOCKED** — the final contract is operationally valid, but prompt freeze and TEST authorization
require recorded human review of the canary and the unresolved Phase 0 sentence-unit risk.

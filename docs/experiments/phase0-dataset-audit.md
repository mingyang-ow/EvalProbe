# Phase 0: dataset audit and frozen pilot

Phase 0 audited the locally obtained RAGTruth files rather than asserting published totals. It
found 2,965 source records, 17,790 responses, 5,934 QA responses, and 5,767 eligible `good` QA
responses. The audit recorded source and response file SHA-256 identities in
[`audit_summary.json`](../../reports/phase0/audit_summary.json).

All 2,927 QA annotations checked against response substrings exactly; malformed, mismatched, and
out-of-range counts were zero. Dataset handling and upstream licensing boundaries are documented in
[THIRD_PARTY_DATA.md](../../THIRD_PARTY_DATA.md); raw corpus files and text-bearing manual audits
remain local.

The frozen pilot uses seed `20260828`, TRAIN-only burden tertiles, and a maximum of one response per
source. Its 60 rows contain 30 supported and 30 unsupported responses, with 10 unsupported rows in
each burden stratum. See the safe [Phase 0 report](../../reports/phase0/report.md) and frozen
manifest identity in the [methodology freeze](../decisions/methodology-freeze.md).

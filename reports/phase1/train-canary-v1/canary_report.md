# Phase 1A TRAIN canary diagnostic

This six-record TRAIN canary validates the judge contract. It is not a scientific result.

## Completion

- Expected calls: 12
- Completed calls: 12
- Provider attempts: 13
- Operational statuses: `{'completed': 12}`
- All-attempt statuses: `{'completed': 12, 'provider_error': 1}`
- Historical failures: `[{'call_key': 'train-canary-v1:12009:local:local-grounding-v1', 'schema_version': 'local-output-v1', 'status': 'provider_error', 'error_type': 'BadRequestError'}]`

## Whole-response

- Agreement count: 5 / 6
- Records: `[{'record_id': '12009', 'reference_verdict': 'UNSUPPORTED', 'judge_verdict': 'UNSUPPORTED', 'agreement': True}, {'record_id': '12839', 'reference_verdict': 'UNSUPPORTED', 'judge_verdict': 'UNSUPPORTED', 'agreement': True}, {'record_id': '13899', 'reference_verdict': 'SUPPORTED', 'judge_verdict': 'UNSUPPORTED', 'agreement': False}, {'record_id': '15074', 'reference_verdict': 'SUPPORTED', 'judge_verdict': 'SUPPORTED', 'agreement': True}, {'record_id': '15288', 'reference_verdict': 'SUPPORTED', 'judge_verdict': 'SUPPORTED', 'agreement': True}, {'record_id': '16875', 'reference_verdict': 'UNSUPPORTED', 'judge_verdict': 'UNSUPPORTED', 'agreement': True}]`

## Local

- Exact sentence-set agreement: 2 / 6
- False-positive sentence IDs: 8
- False-negative sentence IDs: 2
- Records: `[{'record_id': '12009', 'reference_unsupported_sentence_ids': [5], 'judge_unsupported_sentence_ids': [2, 5, 6], 'agreement': False, 'false_positive_sentence_ids': [2, 6], 'false_negative_sentence_ids': []}, {'record_id': '12839', 'reference_unsupported_sentence_ids': [4, 5, 6, 7, 12], 'judge_unsupported_sentence_ids': [5, 7, 12, 13], 'agreement': False, 'false_positive_sentence_ids': [13], 'false_negative_sentence_ids': [4, 6]}, {'record_id': '13899', 'reference_unsupported_sentence_ids': [], 'judge_unsupported_sentence_ids': [12, 19], 'agreement': False, 'false_positive_sentence_ids': [12, 19], 'false_negative_sentence_ids': []}, {'record_id': '15074', 'reference_unsupported_sentence_ids': [], 'judge_unsupported_sentence_ids': [], 'agreement': True, 'false_positive_sentence_ids': [], 'false_negative_sentence_ids': []}, {'record_id': '15288', 'reference_unsupported_sentence_ids': [], 'judge_unsupported_sentence_ids': [], 'agreement': True, 'false_positive_sentence_ids': [], 'false_negative_sentence_ids': []}, {'record_id': '16875', 'reference_unsupported_sentence_ids': [5], 'judge_unsupported_sentence_ids': [5, 8, 11, 12], 'agreement': False, 'false_positive_sentence_ids': [8, 11, 12], 'false_negative_sentence_ids': []}]`

## Granularity examples

- Record IDs: `[]`

## Usage and estimated cost

- Input tokens: 8771
- Cached input tokens: 0
- Cache-write tokens: 1043
- Output tokens: 1470
- Reasoning tokens: 1238
- Total tokens: 10241
- Estimated API cost: $0.065527
- Average per completed call: $0.005461
- Configured cap: $0.50

## Freeze gate

Automated gates: **FAIL**. Human prompt review is still required.

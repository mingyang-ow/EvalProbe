# Phase 1C TRAIN segmentation comparison

This six-record TRAIN rerun validates methodology; it is not a performance estimate.

## Execution

- Whole predictions reused and hash-validated: 6
- Local calls completed: 6 / 6
- Prompt versions unchanged: whole-grounding-v1, local-grounding-v1
- Local units: sentence-v1 → sentence-v2

## Aggregate local comparison

- Exact agreement: 2 → 2 / 6
- False positives: 8 → 7
- False negatives: 2 → 0

## Record 12839

- Detached old marker units 4 and 6 removed: True
- Before: reference `[4, 5, 6, 7, 12]`, judge `[5, 7, 12, 13]`
- After: reference `[3, 4, 7]`, judge `[3, 4, 7, 8]`
- Context: Old units 4 and 6 were segmentation defects; the separate old-unit-13 disagreement was benchmark ambiguity and is not expected to disappear.

## Record-level safe metadata

`[{'record_id': '12009', 'old_to_new_unit_ids': [1, 2, 3, 4, 5, 6], 'merged_marker_old_unit_ids': [], 'reference_unsupported_before': [5], 'reference_unsupported_after': [5], 'judge_unsupported_before': [2, 5, 6], 'judge_unsupported_after': [2, 5, 6], 'false_positives_before': [2, 6], 'false_positives_after': [2, 6], 'false_negatives_before': [], 'false_negatives_after': [], 'exact_match_before': False, 'exact_match_after': False}, {'record_id': '12839', 'old_to_new_unit_ids': [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8], 'merged_marker_old_unit_ids': [2, 4, 6, 8, 10], 'reference_unsupported_before': [4, 5, 6, 7, 12], 'reference_unsupported_after': [3, 4, 7], 'judge_unsupported_before': [5, 7, 12, 13], 'judge_unsupported_after': [3, 4, 7, 8], 'false_positives_before': [13], 'false_positives_after': [8], 'false_negatives_before': [4, 6], 'false_negatives_after': [], 'exact_match_before': False, 'exact_match_after': False}, {'record_id': '13899', 'old_to_new_unit_ids': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 'merged_marker_old_unit_ids': [], 'reference_unsupported_before': [], 'reference_unsupported_after': [], 'judge_unsupported_before': [12, 19], 'judge_unsupported_after': [19], 'false_positives_before': [12, 19], 'false_positives_after': [19], 'false_negatives_before': [], 'false_negatives_after': [], 'exact_match_before': False, 'exact_match_after': False}, {'record_id': '15074', 'old_to_new_unit_ids': [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8], 'merged_marker_old_unit_ids': [2, 4, 6, 8, 10, 12], 'reference_unsupported_before': [], 'reference_unsupported_after': [], 'judge_unsupported_before': [], 'judge_unsupported_after': [], 'false_positives_before': [], 'false_positives_after': [], 'false_negatives_before': [], 'false_negatives_after': [], 'exact_match_before': True, 'exact_match_after': True}, {'record_id': '15288', 'old_to_new_unit_ids': [1, 2, 3, 4], 'merged_marker_old_unit_ids': [], 'reference_unsupported_before': [], 'reference_unsupported_after': [], 'judge_unsupported_before': [], 'judge_unsupported_after': [], 'false_positives_before': [], 'false_positives_after': [], 'false_negatives_before': [], 'false_negatives_after': [], 'exact_match_before': True, 'exact_match_after': True}, {'record_id': '16875', 'old_to_new_unit_ids': [1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 8], 'merged_marker_old_unit_ids': [2, 4, 7, 9], 'reference_unsupported_before': [5], 'reference_unsupported_after': [3], 'judge_unsupported_before': [5, 8, 11, 12], 'judge_unsupported_after': [3, 5, 7, 8], 'false_positives_before': [8, 11, 12], 'false_positives_after': [5, 7, 8], 'false_negatives_before': [], 'false_negatives_after': [], 'exact_match_before': False, 'exact_match_after': False}]`

## Status

**READY FOR HUMAN RE-REVIEW**

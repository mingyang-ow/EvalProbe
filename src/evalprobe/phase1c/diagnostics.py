from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from evalprobe.data.ragtruth import load_dataset, matched_spans, validate_annotation
from evalprobe.phase0.sentences import (
    is_formatting_list_marker,
    label_sentences,
    segment_local_units,
)
from evalprobe.phase1c.workflow import read_jsonl
from evalprobe.review.diagnostics import SUSPICIOUS_REASON_NAMES, suspicious_unit_reasons
from evalprobe.review.storage import load_adjudications, write_safe_json


def _unit_metrics(answer: str, units: tuple[Any, ...]) -> tuple[int, int]:
    suspicious = 0
    markers = 0
    for unit in units:
        text = answer[unit.start : unit.end]
        suspicious += bool(suspicious_unit_reasons(text))
        markers += is_formatting_list_marker(text)
    return suspicious, markers


def regenerate_phase0_v2(
    manual_v1_path: Path,
    manual_v2_path: Path,
    adjudications_path: Path,
    safe_summary_path: Path,
) -> dict[str, Any]:
    rows = read_jsonl(manual_v1_path)
    decisions = load_adjudications(adjudications_path)
    failed_ids = {
        decision.record_id
        for decision in decisions
        if decision.run_id == "phase0-sentence-audit-v1"
        and decision.sentence_audit_status == "FAIL"
        and decision.failure_type == "SEGMENTATION_DEFECT"
    }
    v2_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    changed_failed_ids: list[str] = []
    for row in rows:
        record_id = str(row["response_id"])
        answer = row.get("response_text")
        annotations = row.get("annotations")
        if not isinstance(answer, str) or not isinstance(annotations, list):
            raise ValueError(f"Invalid Phase 0 source-bearing record: {record_id}")
        if any(
            validate_annotation(answer, annotation).status != "matched"
            for annotation in annotations
        ):
            raise ValueError(f"Phase 0 annotation no longer matches: {record_id}")
        before = segment_local_units(answer, "sentence-v1")
        after = segment_local_units(answer, "sentence-v2")
        spans = matched_spans(answer, annotations)
        labelled = label_sentences(list(after.units), spans)
        before_suspicious, before_markers = _unit_metrics(answer, before.units)
        after_suspicious, after_markers = _unit_metrics(answer, after.units)
        changed = before.units != after.units
        if record_id in failed_ids and changed:
            changed_failed_ids.append(record_id)
        v2_rows.append(
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"sentences", "human_review"}
                },
                "run_id": "phase0-segmentation-v2",
                "local_units_version": "sentence-v2",
                "previous_local_units_version": "sentence-v1",
                "old_to_new_unit_ids": list(after.old_to_new),
                "sentences": [
                    {
                        "start": sentence.start,
                        "end": sentence.end,
                        "reference_label": sentence.label,
                        "text": answer[sentence.start : sentence.end],
                    }
                    for sentence in labelled
                ],
                "human_review": {
                    "offsets_sensible": None,
                    "sentence_mapping_sensible": None,
                    "notes": "",
                },
            }
        )
        records.append(
            {
                "record_id": record_id,
                "units_before": len(before.units),
                "units_after": len(after.units),
                "suspicious_units_before": before_suspicious,
                "suspicious_units_after": after_suspicious,
                "list_marker_only_before": before_markers,
                "list_marker_only_after": after_markers,
                "merged_marker_old_unit_ids": list(after.merged_marker_old_ids),
                "unmerged_marker_old_unit_ids": list(after.unmerged_marker_old_ids),
                "old_to_new_unit_ids": list(after.old_to_new),
                "previously_failed_segmentation": record_id in failed_ids,
                "structure_changed": changed,
            }
        )
    manual_v2_path.parent.mkdir(parents=True, exist_ok=True)
    with manual_v2_path.open("w", encoding="utf-8") as handle:
        for row in v2_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "run_id": "phase0-segmentation-v2",
        "record_count": len(rows),
        "same_record_ids_as_v1": len({str(row["response_id"]) for row in rows}) == len(rows),
        "previous_segmentation_failure_ids": sorted(failed_ids),
        "changed_previous_failure_ids": sorted(changed_failed_ids),
        "previous_failures_changed_count": len(changed_failed_ids),
        "units_before": sum(record["units_before"] for record in records),
        "units_after": sum(record["units_after"] for record in records),
        "suspicious_units_before": sum(record["suspicious_units_before"] for record in records),
        "suspicious_units_after": sum(record["suspicious_units_after"] for record in records),
        "list_marker_only_before": sum(record["list_marker_only_before"] for record in records),
        "list_marker_only_after": sum(record["list_marker_only_after"] for record in records),
        "detached_unsupported_list_markers_after": sum(
            record["list_marker_only_after"] for record in records
        ),
        "historical_adjudications_modified": False,
        "new_review_round": "phase0-segmentation-v2",
        "records": records,
    }
    write_safe_json(safe_summary_path, summary)
    return summary


def corpus_segmentation_diagnostics(data_dir: Path, output_path: Path) -> dict[str, Any]:
    sources, responses, issues = load_dataset(data_dir)
    if issues:
        raise ValueError(f"Invalid RAGTruth data: {issues}")
    qa_source_ids = {
        str(source["source_id"]) for source in sources if source.get("task_type") == "QA"
    }
    span_statuses: Counter[str] = Counter()
    for response in responses:
        if str(response.get("source_id")) not in qa_source_ids:
            continue
        answer = response.get("response")
        labels = response.get("labels")
        if not isinstance(answer, str) or not isinstance(labels, list):
            span_statuses["malformed"] += 1
            continue
        for annotation in labels:
            span_statuses[validate_annotation(answer, annotation).status] += 1

    splits: dict[str, Any] = {}
    totals: Counter[str] = Counter()
    warning_ids: list[str] = []
    remaining_suspicious_ids: list[str] = []
    reason_before: Counter[str] = Counter()
    reason_after: Counter[str] = Counter()
    for split in ("train", "test"):
        eligible = [
            response
            for response in responses
            if str(response.get("source_id")) in qa_source_ids
            and response.get("quality") == "good"
            and response.get("split") == split
        ]
        counts: Counter[str] = Counter()
        for response in eligible:
            answer = response.get("response")
            if not isinstance(answer, str):
                raise ValueError(f"Eligible response {response.get('id')} has invalid text")
            before = segment_local_units(answer, "sentence-v1")
            after = segment_local_units(answer, "sentence-v2")
            before_suspicious, before_markers = _unit_metrics(answer, before.units)
            after_suspicious, after_markers = _unit_metrics(answer, after.units)
            counts.update(
                units_before=len(before.units),
                units_after=len(after.units),
                suspicious_before=before_suspicious,
                suspicious_after=after_suspicious,
                list_marker_only_before=before_markers,
                list_marker_only_after=after_markers,
                merged_markers=len(after.merged_marker_old_ids),
                unmerged_marker_warnings=len(after.unmerged_marker_old_ids),
            )
            for unit in before.units:
                reason_before.update(suspicious_unit_reasons(answer[unit.start : unit.end]))
            for unit in after.units:
                reason_after.update(suspicious_unit_reasons(answer[unit.start : unit.end]))
            if after.unmerged_marker_old_ids:
                warning_ids.append(str(response["id"]))
            if after_suspicious:
                remaining_suspicious_ids.append(str(response["id"]))
        split_row = {"eligible_response_count": len(eligible), **dict(counts)}
        splits[split] = split_row
        totals.update(counts)
        totals["eligible_response_count"] += len(eligible)

    summary = {
        "scope": "eligible good QA TRAIN/TEST responses; read-only segmentation diagnostics",
        "versions": {"before": "sentence-v1", "after": "sentence-v2"},
        "splits": splits,
        "overall": dict(totals),
        "suspicious_reason_counts_before": {
            reason: reason_before[reason] for reason in SUSPICIOUS_REASON_NAMES
        },
        "suspicious_reason_counts_after": {
            reason: reason_after[reason] for reason in SUSPICIOUS_REASON_NAMES
        },
        "new_diagnostic_warning_record_ids": sorted(set(warning_ids)),
        "remaining_suspicious_record_ids": sorted(set(remaining_suspicious_ids)),
        "span_integrity": {
            status: span_statuses[status]
            for status in ("matched", "mismatch", "malformed", "out_of_range")
        },
        "raw_response_text_changed": False,
        "judge_calls_made": 0,
        "test_data_access": "read_only_diagnostics",
    }
    write_safe_json(output_path, summary)
    return summary

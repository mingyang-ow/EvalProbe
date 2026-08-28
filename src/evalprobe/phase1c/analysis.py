from __future__ import annotations

from pathlib import Path
from typing import Any

from evalprobe.phase0.sentences import segment_local_units
from evalprobe.phase1.persistence import write_json
from evalprobe.phase1.runner import CanaryPlan, manifest_rows
from evalprobe.phase1c.workflow import read_jsonl


def _latest_completed(results: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    latest = {str(result["call_key"]): result for result in results}
    return {
        (str(result["record_id"]), str(result["view"])): result
        for result in latest.values()
        if result.get("status") == "completed"
    }


def _aggregate_optional(rows: list[dict[str, Any]], field: str) -> int | float | None:
    values = [row.get(field) for row in rows]
    if any(value is None for value in values):
        return None
    return sum(values)


def analyze_phase1c(
    repaired_plan: CanaryPlan,
    base_plan: CanaryPlan,
    phase1a_manifest_path: Path,
    phase1a_results_path: Path,
    phase1c_results: list[dict[str, Any]],
    reusable_whole: dict[str, dict[str, Any]],
    output_dir: Path,
    configured_cap_usd: float,
) -> dict[str, Any]:
    old_manifest = {str(row["record_id"]): row for row in read_jsonl(phase1a_manifest_path)}
    new_manifest = {str(row["record_id"]): row for row in manifest_rows(repaired_plan)}
    old_results = _latest_completed(read_jsonl(phase1a_results_path))
    new_results = _latest_completed(phase1c_results)
    records: list[dict[str, Any]] = []
    base_records = {record.reference.record_id: record for record in base_plan.records}
    for record_id in sorted(old_manifest):
        old_local = old_results.get((record_id, "local"))
        new_local = new_results.get((record_id, "local"))
        if old_local is None or new_local is None:
            raise ValueError(f"Missing completed local result for comparison: {record_id}")
        if new_local.get("local_units_version") != "sentence-v2":
            raise ValueError(f"Missing sentence-v2 result metadata: {record_id}")
        old_reference = set(
            int(value) for value in old_manifest[record_id]["reference_unsupported_sentence_ids"]
        )
        new_reference = set(
            int(value) for value in new_manifest[record_id]["reference_unsupported_sentence_ids"]
        )
        old_judge = set(int(value) for value in old_local["semantic_prediction"])
        new_judge = set(int(value) for value in new_local["semantic_prediction"])
        source = base_records[record_id].source
        mapping = segment_local_units(source.answer, "sentence-v2")
        records.append(
            {
                "record_id": record_id,
                "old_to_new_unit_ids": list(mapping.old_to_new),
                "merged_marker_old_unit_ids": list(mapping.merged_marker_old_ids),
                "reference_unsupported_before": sorted(old_reference),
                "reference_unsupported_after": sorted(new_reference),
                "judge_unsupported_before": sorted(old_judge),
                "judge_unsupported_after": sorted(new_judge),
                "false_positives_before": sorted(old_judge - old_reference),
                "false_positives_after": sorted(new_judge - new_reference),
                "false_negatives_before": sorted(old_reference - old_judge),
                "false_negatives_after": sorted(new_reference - new_judge),
                "exact_match_before": old_reference == old_judge,
                "exact_match_after": new_reference == new_judge,
            }
        )
    completed_local = [
        result
        for (record_id, view), result in new_results.items()
        if view == "local" and record_id in new_manifest
    ]
    record_12839 = next(row for row in records if row["record_id"] == "12839")
    analysis = {
        "run_id": repaired_plan.config["canary"]["run_id"],
        "interpretation": (
            "TRAIN-only six-record methodology validation; not a performance estimate"
        ),
        "prompts_unchanged": True,
        "prompt_versions": {"whole": "whole-grounding-v1", "local": "local-grounding-v1"},
        "local_units_versions": {"before": "sentence-v1", "after": "sentence-v2"},
        "official_benchmark_labels_modified": False,
        "whole_results": {
            "strategy": "reused_phase1a_without_provider_calls",
            "validated_reusable_count": len(reusable_whole),
        },
        "local_calls": {
            "planned": len(repaired_plan.calls),
            "completed": len(completed_local),
        },
        "aggregate": {
            "exact_agreement_before": sum(row["exact_match_before"] for row in records),
            "exact_agreement_after": sum(row["exact_match_after"] for row in records),
            "false_positive_count_before": sum(
                len(row["false_positives_before"]) for row in records
            ),
            "false_positive_count_after": sum(len(row["false_positives_after"]) for row in records),
            "false_negative_count_before": sum(
                len(row["false_negatives_before"]) for row in records
            ),
            "false_negative_count_after": sum(len(row["false_negatives_after"]) for row in records),
        },
        "record_12839": {
            **record_12839,
            "detached_markers_removed": all(
                old_id in record_12839["merged_marker_old_unit_ids"] for old_id in (4, 6)
            ),
            "human_context": (
                "Old units 4 and 6 were segmentation defects; the separate old-unit-13 "
                "disagreement was benchmark ambiguity and is not expected to disappear."
            ),
        },
        "records": records,
        "usage": {
            field: _aggregate_optional(completed_local, field)
            for field in (
                "input_tokens",
                "cached_input_tokens",
                "cache_write_tokens",
                "output_tokens",
                "reasoning_tokens",
                "total_tokens",
            )
        },
        "total_estimated_cost_usd": _aggregate_optional(completed_local, "estimated_cost_usd"),
        "configured_cap_usd": configured_cap_usd,
        "phase_status": "READY FOR HUMAN RE-REVIEW",
    }
    write_json(output_dir / "comparison.json", analysis)
    _write_report(analysis, output_dir / "comparison.md")
    return analysis


def _write_report(analysis: dict[str, Any], path: Path) -> None:
    aggregate = analysis["aggregate"]
    diagnostic = analysis["record_12839"]
    reused_count = analysis["whole_results"]["validated_reusable_count"]
    completed_count = analysis["local_calls"]["completed"]
    planned_count = analysis["local_calls"]["planned"]
    lines = [
        "# Phase 1C TRAIN segmentation comparison",
        "",
        "This six-record TRAIN rerun validates methodology; it is not a performance estimate.",
        "",
        "## Execution",
        "",
        f"- Whole predictions reused and hash-validated: {reused_count}",
        f"- Local calls completed: {completed_count} / {planned_count}",
        "- Prompt versions unchanged: whole-grounding-v1, local-grounding-v1",
        "- Local units: sentence-v1 → sentence-v2",
        "",
        "## Aggregate local comparison",
        "",
        (
            f"- Exact agreement: {aggregate['exact_agreement_before']} → "
            f"{aggregate['exact_agreement_after']} / 6"
        ),
        (
            f"- False positives: {aggregate['false_positive_count_before']} → "
            f"{aggregate['false_positive_count_after']}"
        ),
        (
            f"- False negatives: {aggregate['false_negative_count_before']} → "
            f"{aggregate['false_negative_count_after']}"
        ),
        "",
        "## Record 12839",
        "",
        f"- Detached old marker units 4 and 6 removed: {diagnostic['detached_markers_removed']}",
        (
            f"- Before: reference `{diagnostic['reference_unsupported_before']}`, "
            f"judge `{diagnostic['judge_unsupported_before']}`"
        ),
        (
            f"- After: reference `{diagnostic['reference_unsupported_after']}`, "
            f"judge `{diagnostic['judge_unsupported_after']}`"
        ),
        f"- Context: {diagnostic['human_context']}",
        "",
        "## Record-level safe metadata",
        "",
        f"`{analysis['records']}`",
        "",
        "## Status",
        "",
        f"**{analysis['phase_status']}**",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

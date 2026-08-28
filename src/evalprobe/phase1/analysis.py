from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from evalprobe.phase1.persistence import write_json


def _aggregate_optional(results: list[dict[str, Any]], field: str) -> int | float | None:
    values = [result.get(field) for result in results]
    if any(value is None for value in values):
        return None
    return sum(values)


def analyze_canary(
    manifest: list[dict[str, Any]],
    results: list[dict[str, Any]],
    output_dir: Path,
    configured_cap_usd: float,
) -> dict[str, Any]:
    latest = {str(result["call_key"]): result for result in results}
    completed = [result for result in latest.values() if result.get("status") == "completed"]
    by_record_view = {
        (str(result["record_id"]), str(result["view"])): result for result in completed
    }
    whole_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    granularity_examples: list[str] = []
    for reference in manifest:
        record_id = str(reference["record_id"])
        whole = by_record_view.get((record_id, "whole"))
        local = by_record_view.get((record_id, "local"))
        if whole:
            whole_rows.append(
                {
                    "record_id": record_id,
                    "reference_verdict": reference["reference_label"],
                    "judge_verdict": whole["semantic_prediction"],
                    "agreement": whole["semantic_prediction"] == reference["reference_label"],
                }
            )
        if local:
            expected = set(reference["reference_unsupported_sentence_ids"])
            predicted = set(local["semantic_prediction"])
            local_rows.append(
                {
                    "record_id": record_id,
                    "reference_unsupported_sentence_ids": sorted(expected),
                    "judge_unsupported_sentence_ids": sorted(predicted),
                    "agreement": expected == predicted,
                    "false_positive_sentence_ids": sorted(predicted - expected),
                    "false_negative_sentence_ids": sorted(expected - predicted),
                }
            )
            if (
                reference["reference_label"] == "UNSUPPORTED"
                and whole
                and whole["semantic_prediction"] == "SUPPORTED"
                and predicted.intersection(expected)
            ):
                granularity_examples.append(record_id)

    usage = {
        field: _aggregate_optional(completed, field)
        for field in (
            "input_tokens",
            "cached_input_tokens",
            "cache_write_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
        )
    }
    total_cost = _aggregate_optional(completed, "estimated_cost_usd")
    analysis = {
        "attempt_count": len(results),
        "expected_calls": len(manifest) * 2,
        "completed_calls": len(completed),
        "operational_status_counts": dict(
            sorted(Counter(str(result.get("status")) for result in latest.values()).items())
        ),
        "operational_status_counts_all_attempts": dict(
            sorted(Counter(str(result.get("status")) for result in results).items())
        ),
        "historical_operational_failures": [
            {
                "call_key": result["call_key"],
                "schema_version": result["schema_version"],
                "status": result["status"],
                "error_type": result["error_type"],
            }
            for result in results
            if result.get("status") != "completed"
        ],
        "whole_response": {
            "records": whole_rows,
            "agreement_count": sum(row["agreement"] for row in whole_rows),
        },
        "local": {
            "records": local_rows,
            "exact_agreement_count": sum(row["agreement"] for row in local_rows),
            "false_positive_sentence_count": sum(
                len(row["false_positive_sentence_ids"]) for row in local_rows
            ),
            "false_negative_sentence_count": sum(
                len(row["false_negative_sentence_ids"]) for row in local_rows
            ),
        },
        "granularity_example_record_ids": granularity_examples,
        "usage": usage,
        "total_estimated_cost_usd": total_cost,
        "average_estimated_cost_per_completed_call_usd": (
            total_cost / len(completed) if total_cost is not None and completed else None
        ),
        "attempts_with_missing_usage": sum(
            result.get("accounting_status") == "missing_usage" for result in results
        ),
        "configured_cap_usd": configured_cap_usd,
        "interpretation": (
            "TRAIN-only six-record contract diagnostic; do not interpret as a performance estimate"
        ),
        "automated_freeze_gates": (
            "pass"
            if len(completed) == len(manifest) * 2
            and all(result.get("status") == "completed" for result in results)
            else "fail"
        ),
        "human_prompt_review_required": True,
    }
    write_json(output_dir / "canary_analysis.json", analysis)
    _write_markdown(analysis, output_dir / "canary_report.md")
    return analysis


def _write_markdown(analysis: dict[str, Any], path: Path) -> None:
    whole = analysis["whole_response"]
    local = analysis["local"]
    usage = analysis["usage"]
    lines = [
        "# Phase 1A TRAIN canary diagnostic",
        "",
        "This six-record TRAIN canary validates the judge contract. It is not a scientific result.",
        "",
        "## Completion",
        "",
        f"- Expected calls: {analysis['expected_calls']}",
        f"- Completed calls: {analysis['completed_calls']}",
        f"- Provider attempts: {analysis['attempt_count']}",
        f"- Operational statuses: `{analysis['operational_status_counts']}`",
        (f"- All-attempt statuses: `{analysis['operational_status_counts_all_attempts']}`"),
        f"- Historical failures: `{analysis['historical_operational_failures']}`",
        "",
        "## Whole-response",
        "",
        f"- Agreement count: {whole['agreement_count']} / {len(whole['records'])}",
        f"- Records: `{whole['records']}`",
        "",
        "## Local",
        "",
        (
            f"- Exact sentence-set agreement: {local['exact_agreement_count']} / "
            f"{len(local['records'])}"
        ),
        f"- False-positive sentence IDs: {local['false_positive_sentence_count']}",
        f"- False-negative sentence IDs: {local['false_negative_sentence_count']}",
        f"- Records: `{local['records']}`",
        "",
        "## Granularity examples",
        "",
        f"- Record IDs: `{analysis['granularity_example_record_ids']}`",
        "",
        "## Usage and estimated cost",
        "",
        f"- Input tokens: {usage['input_tokens']}",
        f"- Cached input tokens: {usage['cached_input_tokens']}",
        f"- Cache-write tokens: {usage['cache_write_tokens']}",
        f"- Output tokens: {usage['output_tokens']}",
        f"- Reasoning tokens: {usage['reasoning_tokens']}",
        f"- Total tokens: {usage['total_tokens']}",
        f"- Estimated API cost: ${analysis['total_estimated_cost_usd']:.6f}",
        (
            "- Average per completed call: "
            f"${analysis['average_estimated_cost_per_completed_call_usd']:.6f}"
        ),
        f"- Configured cap: ${analysis['configured_cap_usd']:.2f}",
        "",
        "## Freeze gate",
        "",
        (
            f"Automated gates: **{analysis['automated_freeze_gates'].upper()}**. "
            "Human prompt review is still required."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

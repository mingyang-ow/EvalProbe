from __future__ import annotations

import html
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from evalprobe.phase1.persistence import write_json, write_jsonl
from evalprobe.phase1.runner import CanaryPlan


def _ratio(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def _wilson(successes: int, total: int) -> dict[str, float | None]:
    if not total:
        return {"low": None, "high": None}
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return {"low": centre - margin, "high": centre + margin}


def _metric(numerator: int, denominator: int) -> dict[str, Any]:
    value = _ratio(numerator, denominator)
    value["wilson_95"] = _wilson(numerator, denominator)
    return value


def _latest(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(result["call_key"]): result for result in results}


def _svg_bars(title: str, rows: list[tuple[str, int, int]], path: Path) -> None:
    width, height = 760, 100 + 70 * len(rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="36" font-family="sans-serif" font-size="20" '
        f'font-weight="600">{html.escape(title)}</text>',
    ]
    for index, (label, numerator, denominator) in enumerate(rows):
        y = 68 + index * 70
        fraction = numerator / denominator if denominator else 0
        parts.extend(
            [
                f'<text x="24" y="{y + 18}" font-family="sans-serif" font-size="14">'
                f"{html.escape(label)}</text>",
                f'<rect x="170" y="{y}" width="500" height="24" rx="4" fill="#e5e7eb"/>',
                f'<rect x="170" y="{y}" width="{500 * fraction:.1f}" height="24" '
                'rx="4" fill="#2563eb"/>',
                f'<text x="680" y="{y + 18}" font-family="sans-serif" font-size="14">'
                f"{numerator}/{denominator} ({fraction:.1%})</text>",
            ]
        )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def analyze_frozen_test(
    plan: CanaryPlan,
    results: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    latest = _latest(results)
    expected_keys = {call.call_key for call in plan.calls}
    current = [latest[key] for key in expected_keys if key in latest]
    completed = [row for row in current if row.get("status") == "completed"]
    failures = [row for row in current if row.get("status") != "completed"]
    completed_by_record_view = {(str(row["record_id"]), str(row["view"])): row for row in completed}
    status_counts = Counter(str(row.get("status")) for row in current)
    all_status_counts = Counter(str(row.get("status")) for row in results)

    usage_fields = (
        "input_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
    )
    usage = {field: sum(int(row.get(field) or 0) for row in results) for field in usage_fields}
    costs = [
        float(row["estimated_cost_usd"])
        for row in results
        if row.get("estimated_cost_usd") is not None
    ]
    latencies = [int(row["latency_ms"]) for row in results]
    operational = {
        "attempt_rows": len(results),
        "expected_calls": len(expected_keys),
        "completed_calls": len(completed),
        "operational_failure_count": len(failures),
        "current_status_counts": dict(sorted(status_counts.items())),
        "all_attempt_status_counts": dict(sorted(all_status_counts.items())),
        "failures": [
            {
                "record_id": row.get("record_id"),
                "view": row.get("view"),
                "status": row.get("status"),
                "error_type": row.get("error_type"),
            }
            for row in failures
        ],
        "usage": usage,
        "cost_usd": sum(costs),
        "average_cost_usd": sum(costs) / len(costs) if costs else None,
        "latency_ms": {
            "min": min(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "mean": statistics.mean(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
    }

    confusion = Counter()
    burden = {name: Counter() for name in ("low", "medium", "high")}
    locality: dict[str, Counter[str]] = {}
    whole_queue: list[dict[str, Any]] = []
    local_queue: list[dict[str, Any]] = []
    exact = tp = fp = fn = 0
    granularity_misses = granularity_recovered = 0
    for record in plan.records:
        reference = record.reference
        record_id = reference.record_id
        whole = completed_by_record_view.get((record_id, "whole"))
        local = completed_by_record_view.get((record_id, "local"))
        reference_ids = set(reference.reference_unsupported_sentence_ids)
        predicted_ids: set[int] | None = None
        if local is not None and isinstance(local.get("semantic_prediction"), list):
            predicted_ids = {int(value) for value in local["semantic_prediction"]}
            exact += predicted_ids == reference_ids
            tp += len(predicted_ids & reference_ids)
            fp += len(predicted_ids - reference_ids)
            fn += len(reference_ids - predicted_ids)
            for sentence_id in sorted(reference_ids - predicted_ids):
                local_queue.append(_queue_row(record, "local", "REFERENCE_ONLY", sentence_id))
            for sentence_id in sorted(predicted_ids - reference_ids):
                local_queue.append(_queue_row(record, "local", "JUDGE_ONLY", sentence_id))
        if whole is None:
            continue
        prediction = str(whole["semantic_prediction"])
        confusion[(reference.reference_label, prediction)] += 1
        if prediction != reference.reference_label:
            mismatch = (
                "FALSE_NEGATIVE" if reference.reference_label == "UNSUPPORTED" else "FALSE_POSITIVE"
            )
            whole_queue.append(_queue_row(record, "whole", mismatch, None))
        if reference.reference_label == "UNSUPPORTED":
            stratum = reference.burden_stratum
            burden[stratum]["total"] += 1
            burden[stratum]["caught" if prediction == "UNSUPPORTED" else "missed"] += 1
            group = reference.locality or "UNKNOWN"
            locality.setdefault(group, Counter())["total"] += 1
            locality[group]["caught" if prediction == "UNSUPPORTED" else "missed"] += 1
            if prediction == "SUPPORTED":
                granularity_misses += 1
                if predicted_ids is not None and predicted_ids & reference_ids:
                    granularity_recovered += 1

    tn = confusion[("SUPPORTED", "SUPPORTED")]
    whole_fp = confusion[("SUPPORTED", "UNSUPPORTED")]
    whole_fn = confusion[("UNSUPPORTED", "SUPPORTED")]
    whole_tp = confusion[("UNSUPPORTED", "UNSUPPORTED")]
    whole_total = sum(confusion.values())
    supported_recall = _ratio(tn, tn + whole_fp)
    unsupported_recall = _ratio(whole_tp, whole_tp + whole_fn)
    balanced_values = [
        value["value"]
        for value in (supported_recall, unsupported_recall)
        if value["value"] is not None
    ]
    whole_metrics = {
        "confusion_matrix": {
            "reference_supported_predicted_supported": tn,
            "reference_supported_predicted_unsupported": whole_fp,
            "reference_unsupported_predicted_supported": whole_fn,
            "reference_unsupported_predicted_unsupported": whole_tp,
        },
        "accuracy": _metric(tn + whole_tp, whole_total),
        "balanced_accuracy": sum(balanced_values) / len(balanced_values)
        if balanced_values
        else None,
        "supported": {
            "precision": _ratio(tn, tn + whole_fn),
            "recall": supported_recall,
        },
        "unsupported": {
            "precision": _ratio(whole_tp, whole_tp + whole_fp),
            "recall": unsupported_recall,
            "false_negative_count": whole_fn,
            "false_negative_rate": _ratio(whole_fn, whole_tp + whole_fn),
        },
    }
    burden_metrics = {
        name: {
            "total": counts["total"],
            "caught": counts["caught"],
            "missed": counts["missed"],
            "detection_rate": _metric(counts["caught"], counts["total"]),
            "false_negative_rate": _ratio(counts["missed"], counts["total"]),
        }
        for name, counts in burden.items()
    }
    local_metrics = {
        "response_exact_set_agreement": _ratio(
            exact,
            sum(
                (record.reference.record_id, "local") in completed_by_record_view
                for record in plan.records
            ),
        ),
        "unsupported_unit_precision": _ratio(tp, tp + fp),
        "unsupported_unit_recall": _ratio(tp, tp + fn),
        "true_positive_units": tp,
        "false_positive_units": fp,
        "false_negative_units": fn,
        "reference_unsupported_units": tp + fn,
        "predicted_unsupported_units": tp + fp,
    }
    granularity = {
        "whole_false_negative_records": granularity_misses,
        "recovered_by_local_overlap": granularity_recovered,
        "recovery_rate": _ratio(granularity_recovered, granularity_misses),
    }
    locality_metrics = {
        name: {
            "total": counts["total"],
            "caught": counts["caught"],
            "missed": counts["missed"],
            "detection_rate": _ratio(counts["caught"], counts["total"]),
        }
        for name, counts in sorted(locality.items())
    }
    review_queue = whole_queue + local_queue
    queue_counts = Counter(str(row["mismatch_type"]) for row in review_queue)
    analysis = {
        "run_id": plan.config["canary"]["run_id"],
        "phase_status": "complete"
        if len(completed) == len(expected_keys) and not failures
        else "partial",
        "operational": operational,
        "whole_response": whole_metrics,
        "unsupported_burden_strata": burden_metrics,
        "local_sentence_v2": local_metrics,
        "granularity_gap": granularity,
        "unsupported_locality": locality_metrics,
        "review_queue": {
            "item_count": len(review_queue),
            "counts_by_type": dict(sorted(queue_counts.items())),
            "human_adjudication_status": "not_started",
        },
        "methodology": {
            "model": plan.config["judge"]["model"],
            "reasoning_effort": plan.config["judge"]["reasoning_effort"],
            "prompts": plan.config["judge"]["prompts"],
            "local_units_version": plan.config["local_units"]["version"],
        },
    }
    write_json(output_dir / "analysis.json", analysis)
    write_jsonl(output_dir / "review_queue.jsonl", review_queue)
    _svg_bars(
        "Unsupported detection by burden stratum",
        [
            (name, burden[name]["caught"], burden[name]["total"])
            for name in ("low", "medium", "high")
        ],
        output_dir / "burden_detection.svg",
    )
    _svg_bars(
        "Granularity gap: whole misses recovered locally",
        [("Local overlap among whole false negatives", granularity_recovered, granularity_misses)],
        output_dir / "granularity_gap.svg",
    )
    (output_dir / "report.md").write_text(_report(analysis), encoding="utf-8")
    return analysis


def _queue_row(
    record: Any, view: str, mismatch_type: str, sentence_id: int | None
) -> dict[str, Any]:
    return {
        "run_id": "frozen-test-v1",
        "record_id": record.reference.record_id,
        "source_id": record.reference.source_id,
        "split": "test",
        "view": view,
        "sentence_id": sentence_id,
        "mismatch_type": mismatch_type,
    }


def _format_ratio(value: dict[str, Any]) -> str:
    score = value["value"]
    rendered = "n/a" if score is None else f"{score:.3f}"
    return f"{value['numerator']}/{value['denominator']} ({rendered})"


def _report(analysis: dict[str, Any]) -> str:
    operational = analysis["operational"]
    whole = analysis["whole_response"]
    local = analysis["local_sentence_v2"]
    granularity = analysis["granularity_gap"]
    burden = analysis["unsupported_burden_strata"]
    usage = operational["usage"]
    token_summary = (
        f"{usage['total_tokens']:,} total ({usage['input_tokens']:,} input; "
        f"{usage['cached_input_tokens']:,} cached input; {usage['output_tokens']:,} output; "
        f"{usage['reasoning_tokens']:,} reasoning)"
    )
    balanced = whole["balanced_accuracy"]
    balanced_text = "n/a" if balanced is None else f"{balanced:.3f}"
    burden_rows = "\n".join(
        f"| {name} | {burden[name]['caught']} | {burden[name]['missed']} | "
        f"{_format_ratio(burden[name]['detection_rate'])} |"
        for name in ("low", "medium", "high")
    )
    local_errors = f"{local['false_positive_units']} / {local['false_negative_units']}"
    return f"""# Frozen TEST results

Status: **{analysis["phase_status"].upper()}**. Human TEST adjudication: **not started**.

## Operational

- Completed calls: {operational["completed_calls"]} / {operational["expected_calls"]}
- Operational failures: {operational["operational_failure_count"]}
- Tokens: {token_summary}
- Estimated cost: ${operational["cost_usd"]:.6f}

## Whole-response benchmark metrics

- Accuracy: {_format_ratio(whole["accuracy"])}
- Balanced accuracy: {balanced_text}
- Unsupported precision: {_format_ratio(whole["unsupported"]["precision"])}
- Unsupported recall: {_format_ratio(whole["unsupported"]["recall"])}
- Unsupported false negatives: {whole["unsupported"]["false_negative_count"]}

## Unsupported burden strata

| Stratum | Detected | Missed | Detection rate |
|---|---:|---:|---:|
{burden_rows}

## Sentence-v2 local metrics

- Exact set agreement: {_format_ratio(local["response_exact_set_agreement"])}
- Unsupported-unit precision: {_format_ratio(local["unsupported_unit_precision"])}
- Unsupported-unit recall: {_format_ratio(local["unsupported_unit_recall"])}
- False positives / false negatives: {local_errors}
- Whole false negatives recovered by local overlap: {_format_ratio(granularity["recovery_rate"])}

## Review queue

{analysis["review_queue"]["item_count"]} benchmark disagreements are queued for optional
explanatory human review. No TEST item has been adjudicated and primary metrics remain unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from evalprobe.data.ragtruth import load_dataset, matched_spans
from evalprobe.phase0.sentences import affected_sentence_indices, segment_local_units
from evalprobe.phase1.runner import CanaryPlan, build_plan, build_plan_for_records
from evalprobe.phase1.selection import CanaryReference, SelectedCanaryRecord


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def load_phase1c_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Phase 1C config must be a YAML mapping")
    canary = value.get("canary", {})
    judge = value.get("judge", {})
    budget = value.get("budget", {})
    if canary.get("split") != "train" or canary.get("total") != 6:
        raise ValueError("Phase 1C requires the same six TRAIN records")
    if canary.get("expected_calls") != 6 or judge.get("views") != ["local"]:
        raise ValueError("Phase 1C must plan exactly six local-only calls")
    if value.get("local_units", {}).get("version") != "sentence-v2":
        raise ValueError("Phase 1C requires sentence-v2 local units")
    if judge.get("model") != "gpt-5.6-sol" or judge.get("reasoning_effort") != "low":
        raise ValueError("Phase 1C requires gpt-5.6-sol with low reasoning")
    if judge.get("automatic_retries") != 0 or judge.get("fallback_models") != 0:
        raise ValueError("Phase 1C prohibits retries and fallback models")
    if judge.get("prompts") != {
        "whole": "whole-grounding-v1",
        "local": "local-grounding-v1",
    }:
        raise ValueError("Phase 1C must preserve the Phase 1A prompts")
    if budget.get("maximum_paid_calls") != 6 or float(budget.get("hard_cap_usd", 0)) != 0.25:
        raise ValueError("Phase 1C requires a six-call, USD $0.25 hard cap")
    return value


def build_phase1c_plan(
    config: dict[str, Any], data_dir: Path, feature_path: Path, repository_root: Path
) -> tuple[CanaryPlan, CanaryPlan]:
    reuse = config["reuse"]
    phase1a_config = yaml.safe_load(
        (repository_root / str(reuse["phase1a_config"])).read_text(encoding="utf-8")
    )
    if not isinstance(phase1a_config, dict):
        raise ValueError("Invalid Phase 1A config")
    base_plan = build_plan(phase1a_config, data_dir, feature_path)
    base_manifest = read_jsonl(repository_root / str(reuse["phase1a_manifest"]))
    expected_ids = {str(value) for value in config["canary"]["expected_record_ids"]}
    manifest_ids = {str(row["record_id"]) for row in base_manifest}
    selected_ids = {record.reference.record_id for record in base_plan.records}
    if manifest_ids != expected_ids or selected_ids != expected_ids:
        raise ValueError("Phase 1C record set differs from the frozen Phase 1A TRAIN canary")

    _, responses, issues = load_dataset(data_dir)
    if issues:
        raise ValueError(f"Invalid RAGTruth files: {issues}")
    response_by_id = {str(response["id"]): response for response in responses}

    repaired: list[SelectedCanaryRecord] = []
    for record in base_plan.records:
        segmentation = segment_local_units(record.source.answer, "sentence-v2")
        response = response_by_id[record.reference.record_id]
        labels = response.get("labels")
        if not isinstance(labels, list):
            raise ValueError(f"Canary record has invalid labels: {record.reference.record_id}")
        spans = matched_spans(record.source.answer, labels)
        repaired_ids = tuple(
            sorted(
                index + 1 for index in affected_sentence_indices(list(segmentation.units), spans)
            )
        )
        mapped_old_ids = tuple(
            sorted(
                {
                    segmentation.old_to_new[old_id - 1]
                    for old_id in record.reference.reference_unsupported_sentence_ids
                }
            )
        )
        if repaired_ids != mapped_old_ids:
            raise ValueError(f"Reference remapping mismatch: {record.reference.record_id}")
        repaired.append(
            SelectedCanaryRecord(
                source=record.source,
                reference=CanaryReference(
                    record_id=record.reference.record_id,
                    source_id=record.reference.source_id,
                    reference_label=record.reference.reference_label,
                    burden_stratum=record.reference.burden_stratum,
                    reference_unsupported_sentence_ids=repaired_ids,
                ),
            )
        )
    return build_plan_for_records(config, repaired), base_plan


def reusable_whole_results(base_plan: CanaryPlan, results_path: Path) -> dict[str, dict[str, Any]]:
    results = read_jsonl(results_path)
    latest = {str(result["call_key"]): result for result in results}
    reusable: dict[str, dict[str, Any]] = {}
    for call in base_plan.calls:
        if call.view != "whole":
            continue
        result = latest.get(call.call_key)
        if (
            result is None
            or result.get("status") != "completed"
            or result.get("input_hash") != call.input_hash
            or result.get("record_id") != call.record_id
            or result.get("source_id") != call.source_id
            or result.get("view") != "whole"
            or result.get("prompt_version") != call.request.prompt_version
            or result.get("model_requested") != call.request.model
            or result.get("semantic_prediction") not in {"SUPPORTED", "UNSUPPORTED"}
        ):
            raise ValueError(f"Phase 1A whole result is not safely reusable: {call.record_id}")
        reusable[call.record_id] = result
    if len(reusable) != len(base_plan.records):
        raise ValueError("Not all Phase 1A whole results are reusable")
    return reusable

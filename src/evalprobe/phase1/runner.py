from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from evalprobe.phase0.sentences import segment_local_units
from evalprobe.phase1.contracts import (
    LOCAL_OUTPUT_SCHEMA,
    WHOLE_OUTPUT_SCHEMA,
    LocalJudgeInput,
    NumberedSentence,
    WholeJudgeInput,
    render_model_input,
)
from evalprobe.phase1.persistence import ResultStore, write_json, write_jsonl
from evalprobe.phase1.pricing import (
    Pricing,
    approximate_tokens,
    calculate_cost_usd,
    estimate_max_cost_usd,
)
from evalprobe.phase1.prompts import get_prompt
from evalprobe.phase1.provider import OpenAIResponsesJudge, ProviderRequest, View
from evalprobe.phase1.selection import CanaryReference, SelectedCanaryRecord, load_canary_records


class CanaryExecutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CanaryCall:
    run_id: str
    record_id: str
    source_id: str
    view: View
    schema_version: str
    request: ProviderRequest
    input_hash: str
    approximate_input_characters: int
    approximate_input_tokens: int
    estimated_max_cost_usd: float
    local_units_version: str | None = None

    @property
    def call_key(self) -> str:
        return f"{self.run_id}:{self.record_id}:{self.view}:{self.request.prompt_version}"


@dataclass(frozen=True, slots=True)
class CanaryPlan:
    records: tuple[SelectedCanaryRecord, ...]
    calls: tuple[CanaryCall, ...]
    pricing: Pricing
    config: dict[str, Any]


def _hash_request(request: ProviderRequest, local_units_version: str | None = None) -> str:
    payload = {
        "model": request.model,
        "reasoning_effort": request.reasoning_effort,
        "prompt_version": request.prompt_version,
        "prompt_text": request.prompt_text,
        "model_input": request.model_input,
        "schema": request.schema,
        "max_output_tokens": request.max_output_tokens,
    }
    if local_units_version is not None:
        payload["local_units_version"] = local_units_version
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _build_call(
    *,
    config: dict[str, Any],
    reference: CanaryReference,
    view: Literal["whole", "local"],
    model_input: str,
    valid_sentence_ids: frozenset[int],
    schema: dict[str, Any],
    schema_version: str,
    pricing: Pricing,
    local_units_version: str | None = None,
) -> CanaryCall:
    judge_config = config["judge"]
    prompt = get_prompt(str(judge_config["prompts"][view]))
    max_output_tokens = int(judge_config["max_output_tokens"][view])
    request = ProviderRequest(
        view=view,
        prompt_text=prompt.text,
        prompt_version=prompt.version,
        model_input=model_input,
        schema_name=schema_version.replace("-", "_"),
        schema=schema,
        model=str(judge_config["model"]),
        reasoning_effort=str(judge_config["reasoning_effort"]),
        max_output_tokens=max_output_tokens,
        valid_sentence_ids=valid_sentence_ids,
    )
    character_count = len(prompt.text) + len(model_input) + len(json.dumps(schema))
    approximate_input = approximate_tokens(
        character_count + 150,
        int(config["budget"]["approximate_characters_per_token"]),
    )
    return CanaryCall(
        run_id=str(config["canary"]["run_id"]),
        record_id=reference.record_id,
        source_id=reference.source_id,
        view=view,
        schema_version=schema_version,
        request=request,
        input_hash=_hash_request(request, local_units_version),
        approximate_input_characters=character_count,
        approximate_input_tokens=approximate_input,
        estimated_max_cost_usd=estimate_max_cost_usd(approximate_input, max_output_tokens, pricing),
        local_units_version=local_units_version,
    )


def build_plan(config: dict[str, Any], data_dir: Path, feature_path: Path) -> CanaryPlan:
    records = load_canary_records(data_dir, feature_path, int(config["canary"]["seed"]))
    return build_plan_for_records(config, records)


def build_plan_for_records(
    config: dict[str, Any],
    records: list[SelectedCanaryRecord],
) -> CanaryPlan:
    """Build a canary plan from preselected records without exposing reference data to judges."""
    pricing = Pricing.from_config(config["budget"]["pricing"])
    views = tuple(config["judge"].get("views", ("whole", "local")))
    if not views or any(view not in {"whole", "local"} for view in views):
        raise ValueError("judge.views must contain whole and/or local")
    local_units_version = str(config.get("local_units", {}).get("version", "sentence-v1"))
    if local_units_version not in {"sentence-v1", "sentence-v2"}:
        raise ValueError("Unknown local-units version")
    calls: list[CanaryCall] = []
    for record in records:
        answer_sentences = segment_local_units(
            record.source.answer,
            local_units_version,  # type: ignore[arg-type]
        ).units
        numbered = tuple(
            NumberedSentence(index, record.source.answer[sentence.start : sentence.end])
            for index, sentence in enumerate(answer_sentences, start=1)
        )
        local_input = LocalJudgeInput(
            question=record.source.question,
            evidence=record.source.evidence,
            numbered_sentences=numbered,
        )
        if "whole" in views:
            whole_input = WholeJudgeInput.from_source(record.source)
            calls.append(
                _build_call(
                    config=config,
                    reference=record.reference,
                    view="whole",
                    model_input=render_model_input(whole_input),
                    valid_sentence_ids=frozenset(),
                    schema=WHOLE_OUTPUT_SCHEMA,
                    schema_version="whole-output-v1",
                    pricing=pricing,
                )
            )
        if "local" in views:
            calls.append(
                _build_call(
                    config=config,
                    reference=record.reference,
                    view="local",
                    model_input=render_model_input(local_input),
                    valid_sentence_ids=local_input.sentence_ids,
                    schema=LOCAL_OUTPUT_SCHEMA,
                    schema_version="local-output-v2",
                    pricing=pricing,
                    local_units_version=local_units_version,
                )
            )
    plan = CanaryPlan(tuple(records), tuple(calls), pricing, config)
    validate_plan(plan)
    return plan


def validate_plan(plan: CanaryPlan) -> None:
    config = plan.config
    if len(plan.records) != int(config["canary"]["total"]):
        raise ValueError("Canary record count does not match config")
    if len(plan.calls) != int(config["canary"]["expected_calls"]):
        raise ValueError("Canary call count does not match config")
    if len({record.reference.source_id for record in plan.records}) != len(plan.records):
        raise ValueError("Canary source_id values are not unique")
    if any(
        record.reference.reference_label == "SUPPORTED"
        and record.reference.reference_unsupported_sentence_ids
        for record in plan.records
    ):
        raise ValueError("Supported canary record has unsupported reference sentence IDs")
    views_by_record: dict[str, set[str]] = {}
    for call in plan.calls:
        views_by_record.setdefault(call.record_id, set()).add(call.view)
        if call.view == "local" and not call.request.valid_sentence_ids:
            raise ValueError("Local request has no valid sentence IDs")
        if "previous_response_id" in call.request.api_kwargs():
            raise ValueError("Judge calls must not share conversation state")
    expected_views = set(config["judge"].get("views", ("whole", "local")))
    if any(views != expected_views for views in views_by_record.values()):
        raise ValueError("Every record requires exactly the configured views")


def manifest_rows(plan: CanaryPlan) -> list[dict[str, Any]]:
    sentence_counts = {
        call.record_id: len(call.request.valid_sentence_ids)
        for call in plan.calls
        if call.view == "local"
    }
    return [
        {
            "record_id": record.reference.record_id,
            "source_id": record.reference.source_id,
            "split": "train",
            "reference_label": record.reference.reference_label,
            "burden_stratum": record.reference.burden_stratum,
            "reference_unsupported_sentence_ids": list(
                record.reference.reference_unsupported_sentence_ids
            ),
            "sentence_count": sentence_counts[record.reference.record_id],
        }
        for record in plan.records
    ]


def preflight_summary(plan: CanaryPlan, max_cost_usd: float) -> dict[str, Any]:
    return {
        "run_id": plan.config["canary"]["run_id"],
        "mode": "dry_run",
        "model": plan.config["judge"]["model"],
        "reasoning_effort": plan.config["judge"]["reasoning_effort"],
        "local_units_version": plan.config.get("local_units", {}).get("version", "sentence-v1"),
        "selected_record_count": len(plan.records),
        "selected_split": "train",
        "whole_call_count": sum(call.view == "whole" for call in plan.calls),
        "local_call_count": sum(call.view == "local" for call in plan.calls),
        "expected_call_count": len(plan.calls),
        "approximate_input_characters": sum(
            call.approximate_input_characters for call in plan.calls
        ),
        "approximate_input_tokens": sum(call.approximate_input_tokens for call in plan.calls),
        "max_output_tokens": plan.config["judge"]["max_output_tokens"],
        "estimated_max_cost_usd": sum(call.estimated_max_cost_usd for call in plan.calls),
        "hard_spend_limit_usd": max_cost_usd,
        "source_ids_unique": len({record.reference.source_id for record in plan.records})
        == len(plan.records),
        "judge_input_allowlist": ["question", "evidence", "answer_or_numbered_sentences"],
        "reference_fields_in_judge_dtos": False,
        "independent_whole_local_requests": True,
        "structured_output_schemas_validated": True,
        "network_calls_made": 0,
        "sentence_conversion_human_review": plan.config["prerequisites"][
            "sentence_conversion_human_review"
        ],
        "calls": [
            {
                "record_id": call.record_id,
                "source_id": call.source_id,
                "view": call.view,
                "prompt_version": call.request.prompt_version,
                "schema_version": call.schema_version,
                "local_units_version": call.local_units_version,
                "input_hash": call.input_hash,
                "approximate_input_characters": call.approximate_input_characters,
                "approximate_input_tokens": call.approximate_input_tokens,
                "max_output_tokens": call.request.max_output_tokens,
                "estimated_max_cost_usd": call.estimated_max_cost_usd,
            }
            for call in plan.calls
        ],
    }


def write_dry_run(plan: CanaryPlan, output_dir: Path, max_cost_usd: float) -> dict[str, Any]:
    summary = preflight_summary(plan, max_cost_usd)
    write_jsonl(output_dir / "manifest.jsonl", manifest_rows(plan))
    write_json(output_dir / "dry_run.json", summary)
    return summary


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _result_envelope(
    call: CanaryCall,
    outcome: Any,
    latency_ms: int,
    pricing: Pricing,
    pricing_effective_date: str,
) -> dict[str, Any]:
    estimated_cost = calculate_cost_usd(outcome.usage, pricing)
    prediction = outcome.semantic_prediction
    if isinstance(prediction, tuple):
        prediction = list(prediction)
    return {
        "run_id": call.run_id,
        "call_key": call.call_key,
        "record_id": call.record_id,
        "source_id": call.source_id,
        "view": call.view,
        "prompt_version": call.request.prompt_version,
        "schema_version": call.schema_version,
        "local_units_version": call.local_units_version,
        "model_requested": call.request.model,
        "model_returned": outcome.model_returned,
        "reasoning_effort": call.request.reasoning_effort,
        "status": outcome.status,
        "semantic_prediction": prediction,
        "input_hash": call.input_hash,
        "provider_response_id": outcome.provider_response_id,
        "input_tokens": outcome.usage.input_tokens,
        "cached_input_tokens": outcome.usage.cached_input_tokens,
        "cache_write_tokens": outcome.usage.cache_write_tokens,
        "output_tokens": outcome.usage.output_tokens,
        "reasoning_tokens": outcome.usage.reasoning_tokens,
        "total_tokens": outcome.usage.total_tokens,
        "estimated_cost_usd": estimated_cost,
        "accounting_status": "complete" if estimated_cost is not None else "missing_usage",
        "pricing_effective_date": pricing_effective_date,
        "latency_ms": latency_ms,
        "timestamp": _timestamp(),
        "error_type": outcome.error_type,
    }


def _budget_guard_envelope(call: CanaryCall, reason: str) -> dict[str, Any]:
    return {
        "run_id": call.run_id,
        "call_key": call.call_key,
        "record_id": call.record_id,
        "source_id": call.source_id,
        "view": call.view,
        "prompt_version": call.request.prompt_version,
        "schema_version": call.schema_version,
        "local_units_version": call.local_units_version,
        "model_requested": call.request.model,
        "model_returned": None,
        "reasoning_effort": call.request.reasoning_effort,
        "status": "budget_guard",
        "semantic_prediction": None,
        "input_hash": call.input_hash,
        "provider_response_id": None,
        "input_tokens": None,
        "cached_input_tokens": None,
        "cache_write_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": None,
        "estimated_cost_usd": None,
        "accounting_status": "not_called",
        "pricing_effective_date": None,
        "latency_ms": 0,
        "timestamp": _timestamp(),
        "error_type": reason,
    }


def execute_plan(
    plan: CanaryPlan,
    judge: OpenAIResponsesJudge,
    output_dir: Path,
    max_cost_usd: float,
) -> list[dict[str, Any]]:
    hard_cap = float(plan.config["budget"]["hard_cap_usd"])
    if max_cost_usd <= 0 or max_cost_usd > hard_cap:
        raise ValueError(f"Execution cost limit must be in (0, {hard_cap}]")
    store = ResultStore(output_dir / "results.jsonl")
    write_jsonl(output_dir / "manifest.jsonl", manifest_rows(plan))
    existing = store.read_all()
    latest = {str(result["call_key"]): result for result in existing}
    spent = sum(
        float(result["estimated_cost_usd"])
        for result in existing
        if result.get("estimated_cost_usd") is not None
    )
    if any(
        result.get("status") == "completed" and result.get("estimated_cost_usd") is None
        for result in latest.values()
    ):
        raise CanaryExecutionError("Cannot resume: a completed call has missing usage accounting")
    remaining = [
        call
        for call in plan.calls
        if not (
            latest.get(call.call_key, {}).get("status") == "completed"
            and latest.get(call.call_key, {}).get("input_hash") == call.input_hash
        )
    ]
    projected_remaining = sum(call.estimated_max_cost_usd for call in remaining)
    if spent + projected_remaining > max_cost_usd and remaining:
        guard = _budget_guard_envelope(remaining[0], "projected_run_cost_exceeds_limit")
        store.append(guard)
        raise CanaryExecutionError("Budget guard blocked execution before any new call")

    maximum_paid_calls = int(plan.config["budget"]["maximum_paid_calls"])
    pricing_date = str(plan.config["budget"]["pricing"]["effective_date"])
    for paid_call_number, call in enumerate(remaining, start=1):
        if paid_call_number > maximum_paid_calls:
            store.append(_budget_guard_envelope(call, "maximum_paid_calls_reached"))
            raise CanaryExecutionError("Maximum paid call count reached")
        if spent + call.estimated_max_cost_usd > max_cost_usd:
            store.append(_budget_guard_envelope(call, "projected_call_cost_exceeds_limit"))
            raise CanaryExecutionError("Budget guard blocked the next call")
        started = time.perf_counter()
        outcome = judge.call(call.request)
        latency_ms = round((time.perf_counter() - started) * 1000)
        envelope = _result_envelope(call, outcome, latency_ms, plan.pricing, pricing_date)
        store.append(envelope)
        if envelope["estimated_cost_usd"] is not None:
            spent += float(envelope["estimated_cost_usd"])
        if outcome.status != "completed":
            raise CanaryExecutionError(f"Canary stopped after {call.call_key}: {outcome.status}")
        if envelope["estimated_cost_usd"] is None:
            raise CanaryExecutionError(
                f"Canary stopped after {call.call_key}: provider usage was incomplete"
            )
        if spent > max_cost_usd:
            raise CanaryExecutionError("Canary exceeded the configured spend limit")
    return store.read_all()

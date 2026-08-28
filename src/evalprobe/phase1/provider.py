from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from evalprobe.phase1.contracts import (
    ContractValidationError,
    StructuredOutputError,
    Verdict,
    parse_local_output,
    parse_whole_output,
)
from evalprobe.phase1.pricing import Usage

View = Literal["whole", "local"]


@dataclass(frozen=True, slots=True)
class ProviderOutcome:
    status: str
    semantic_prediction: Verdict | tuple[int, ...] | None
    provider_response_id: str | None
    model_returned: str | None
    usage: Usage
    error_type: str | None


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    view: View
    prompt_text: str
    prompt_version: str
    model_input: str
    schema_name: str
    schema: dict[str, Any]
    model: str
    reasoning_effort: str
    max_output_tokens: int
    valid_sentence_ids: frozenset[int]

    def api_kwargs(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "instructions": self.prompt_text,
            "input": [{"role": "user", "content": self.model_input}],
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": self.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": self.schema_name,
                    "strict": True,
                    "schema": self.schema,
                }
            },
            "tools": [],
            "tool_choice": "none",
            "store": False,
        }


def _get(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def extract_usage(response: object) -> Usage:
    usage = _get(response, "usage")
    if usage is None:
        return Usage(None, None, None, None, None, None)
    input_details = _get(usage, "input_tokens_details")
    output_details = _get(usage, "output_tokens_details")
    return Usage(
        input_tokens=_get(usage, "input_tokens"),
        cached_input_tokens=_get(input_details, "cached_tokens"),
        cache_write_tokens=_get(input_details, "cache_write_tokens"),
        output_tokens=_get(usage, "output_tokens"),
        reasoning_tokens=_get(output_details, "reasoning_tokens"),
        total_tokens=_get(usage, "total_tokens"),
    )


def _find_refusal(response: object) -> bool:
    for item in _get(response, "output", []) or []:
        if _get(item, "type") != "message":
            continue
        for content in _get(item, "content", []) or []:
            if _get(content, "type") == "refusal":
                return True
    return False


def _error_outcome(response: object, status: str, error_type: str) -> ProviderOutcome:
    return ProviderOutcome(
        status=status,
        semantic_prediction=None,
        provider_response_id=_get(response, "id"),
        model_returned=_get(response, "model"),
        usage=extract_usage(response),
        error_type=error_type,
    )


class OpenAIResponsesJudge:
    """Single-provider boundary for one independent Responses API call."""

    def __init__(self, client: object) -> None:
        self._client = client

    def call(self, request: ProviderRequest) -> ProviderOutcome:
        try:
            response = self._client.responses.create(**request.api_kwargs())  # type: ignore[attr-defined]
        except Exception as error:  # Provider exception classes vary by SDK version.
            provider_code = getattr(error, "code", None)
            error_type = type(error).__name__
            if isinstance(provider_code, str) and provider_code:
                error_type = f"{error_type}:{provider_code}"
            return ProviderOutcome(
                status="provider_error",
                semantic_prediction=None,
                provider_response_id=None,
                model_returned=None,
                usage=Usage(None, None, None, None, None, None),
                error_type=error_type,
            )

        response_status = _get(response, "status")
        if response_status == "incomplete":
            details = _get(response, "incomplete_details")
            reason = _get(details, "reason", "incomplete")
            return _error_outcome(response, "provider_incomplete", str(reason))
        if response_status != "completed":
            return _error_outcome(
                response, "provider_error", f"provider_status_{response_status or 'missing'}"
            )
        if _find_refusal(response):
            return _error_outcome(response, "provider_refusal", "refusal")
        output_text = _get(response, "output_text")
        if not isinstance(output_text, str) or not output_text:
            return _error_outcome(response, "structured_output_error", "missing_output_text")
        try:
            if request.view == "whole":
                prediction: Verdict | tuple[int, ...] = parse_whole_output(output_text)
            else:
                prediction = parse_local_output(output_text, request.valid_sentence_ids)
        except StructuredOutputError as error:
            return _error_outcome(response, "structured_output_error", type(error).__name__)
        except ContractValidationError as error:
            return _error_outcome(response, "contract_validation_error", type(error).__name__)
        return ProviderOutcome(
            status="completed",
            semantic_prediction=prediction,
            provider_response_id=_get(response, "id"),
            model_returned=_get(response, "model"),
            usage=extract_usage(response),
            error_type=None,
        )

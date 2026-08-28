import json
from pathlib import Path

import pytest

from evalprobe.phase1.contracts import SafeJudgeSource
from evalprobe.phase1.pricing import Pricing
from evalprobe.phase1.prompts import WHOLE_GROUNDING_V1
from evalprobe.phase1.provider import ProviderRequest
from evalprobe.phase1.runner import CanaryCall, CanaryPlan, build_plan_for_records
from evalprobe.phase1.selection import CanaryReference, SelectedCanaryRecord
from evalprobe.phase1c.workflow import reusable_whole_results


def _record() -> SelectedCanaryRecord:
    return SelectedCanaryRecord(
        SafeJudgeSource("question", "evidence", "1. Claim."),
        CanaryReference("r1", "s1", "UNSUPPORTED", "high", (1,)),
    )


def _config() -> dict[str, object]:
    return {
        "canary": {"run_id": "phase1c", "total": 1, "expected_calls": 1},
        "judge": {
            "model": "gpt-5.6-sol",
            "reasoning_effort": "low",
            "views": ["local"],
            "max_output_tokens": {"whole": 10, "local": 10},
            "prompts": {"whole": "whole-grounding-v1", "local": "local-grounding-v1"},
        },
        "local_units": {"version": "sentence-v2"},
        "budget": {
            "hard_cap_usd": 0.25,
            "maximum_paid_calls": 1,
            "approximate_characters_per_token": 3,
            "pricing": {
                "input_per_million_usd": 4,
                "cached_input_per_million_usd": 0.4,
                "cache_write_per_million_usd": 5,
                "output_per_million_usd": 20,
                "long_context_threshold_tokens": 272000,
                "long_context_input_multiplier": 2,
                "long_context_output_multiplier": 1.5,
                "effective_date": "2026-08-28",
            },
        },
        "prerequisites": {"sentence_conversion_human_review": "complete"},
    }


def test_phase1c_plan_is_local_only_and_versions_units() -> None:
    plan = build_plan_for_records(_config(), [_record()])  # type: ignore[arg-type]
    assert len(plan.calls) == 1
    assert plan.calls[0].view == "local"
    assert plan.calls[0].local_units_version == "sentence-v2"
    assert "1. Claim." in plan.calls[0].request.model_input


def _whole_plan() -> CanaryPlan:
    request = ProviderRequest(
        view="whole",
        prompt_text=WHOLE_GROUNDING_V1.text,
        prompt_version=WHOLE_GROUNDING_V1.version,
        model_input="safe",
        schema_name="whole",
        schema={},
        model="gpt-5.6-sol",
        reasoning_effort="low",
        max_output_tokens=10,
        valid_sentence_ids=frozenset(),
    )
    call = CanaryCall("base", "r1", "s1", "whole", "whole-output-v1", request, "hash", 1, 1, 0.01)
    return CanaryPlan(
        (_record(),),
        (call,),
        Pricing(4, 0.4, 5, 20, 272000, 2, 1.5),
        {"canary": {"total": 1, "expected_calls": 1}, "judge": {"views": ["whole"]}},
    )


def test_prior_whole_result_requires_exact_hash_and_contract(tmp_path: Path) -> None:
    result = {
        "call_key": "base:r1:whole:whole-grounding-v1",
        "record_id": "r1",
        "source_id": "s1",
        "view": "whole",
        "status": "completed",
        "input_hash": "hash",
        "prompt_version": "whole-grounding-v1",
        "model_requested": "gpt-5.6-sol",
        "semantic_prediction": "SUPPORTED",
    }
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    assert set(reusable_whole_results(_whole_plan(), path)) == {"r1"}

    result["input_hash"] = "different"
    path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not safely reusable"):
        reusable_whole_results(_whole_plan(), path)

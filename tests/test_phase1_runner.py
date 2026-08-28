from pathlib import Path

import pytest

from evalprobe.phase1.contracts import SafeJudgeSource
from evalprobe.phase1.persistence import ResultStore
from evalprobe.phase1.pricing import Pricing, Usage
from evalprobe.phase1.prompts import WHOLE_GROUNDING_V1
from evalprobe.phase1.provider import ProviderOutcome, ProviderRequest
from evalprobe.phase1.runner import (
    CanaryCall,
    CanaryExecutionError,
    CanaryPlan,
    execute_plan,
    write_dry_run,
)
from evalprobe.phase1.selection import CanaryReference, SelectedCanaryRecord


class FakeJudge:
    def __init__(self) -> None:
        self.calls = 0

    def call(self, request: ProviderRequest) -> ProviderOutcome:
        self.calls += 1
        prediction: str | tuple[int, ...] = "SUPPORTED" if request.view == "whole" else ()
        return ProviderOutcome(
            status="completed",
            semantic_prediction=prediction,  # type: ignore[arg-type]
            provider_response_id=f"resp_{self.calls}",
            model_returned="gpt-5.6-sol",
            usage=Usage(100, 0, 0, 10, 2, 110),
            error_type=None,
        )


def _plan() -> CanaryPlan:
    config = {
        "canary": {"run_id": "run", "total": 1, "expected_calls": 2},
        "judge": {
            "model": "gpt-5.6-sol",
            "reasoning_effort": "low",
            "max_output_tokens": {"whole": 10, "local": 10},
        },
        "budget": {
            "hard_cap_usd": 0.50,
            "maximum_paid_calls": 12,
            "pricing": {"effective_date": "2026-08-28"},
        },
        "prerequisites": {"sentence_conversion_human_review": "not_recorded"},
    }
    record = SelectedCanaryRecord(
        SafeJudgeSource("q", "e", "a"),
        CanaryReference("r1", "s1", "SUPPORTED", "none", ()),
    )
    calls = []
    for view in ("whole", "local"):
        request = ProviderRequest(
            view=view,  # type: ignore[arg-type]
            prompt_text=WHOLE_GROUNDING_V1.text,
            prompt_version=WHOLE_GROUNDING_V1.version,
            model_input="safe input",
            schema_name="schema",
            schema={},
            model="gpt-5.6-sol",
            reasoning_effort="low",
            max_output_tokens=10,
            valid_sentence_ids=frozenset({1}) if view == "local" else frozenset(),
        )
        calls.append(
            CanaryCall(
                "run",
                "r1",
                "s1",
                view,  # type: ignore[arg-type]
                "schema-v1",
                request,
                f"hash-{view}",
                10,
                10,
                0.001,
            )
        )
    return CanaryPlan((record,), tuple(calls), Pricing(4, 0.4, 5, 20, 272000, 2, 1.5), config)


def test_dry_run_makes_no_provider_calls(tmp_path: Path) -> None:
    judge = FakeJudge()
    summary = write_dry_run(_plan(), tmp_path, 0.50)
    assert judge.calls == 0
    assert summary["network_calls_made"] == 0
    assert "safe input" not in (tmp_path / "dry_run.json").read_text()


def test_result_metadata_and_resume_skip_completed_calls(tmp_path: Path) -> None:
    judge = FakeJudge()
    results = execute_plan(_plan(), judge, tmp_path, 0.50)
    assert judge.calls == 2
    assert all(result["run_id"] == "run" for result in results)
    assert all(result["model_requested"] == "gpt-5.6-sol" for result in results)
    assert all(result["input_hash"].startswith("hash-") for result in results)

    execute_plan(_plan(), judge, tmp_path, 0.50)
    assert judge.calls == 2
    assert len(ResultStore(tmp_path / "results.jsonl").read_all()) == 2


def test_budget_guard_blocks_before_provider_call(tmp_path: Path) -> None:
    judge = FakeJudge()
    with pytest.raises(CanaryExecutionError, match="Budget guard"):
        execute_plan(_plan(), judge, tmp_path, 0.0001)
    assert judge.calls == 0
    result = ResultStore(tmp_path / "results.jsonl").read_all()[0]
    assert result["status"] == "budget_guard"

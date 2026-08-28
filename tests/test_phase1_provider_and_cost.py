from types import SimpleNamespace

from evalprobe.phase1.contracts import WHOLE_OUTPUT_SCHEMA
from evalprobe.phase1.pricing import Pricing, Usage, calculate_cost_usd
from evalprobe.phase1.provider import OpenAIResponsesJudge, ProviderRequest, extract_usage


class FakeResponses:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.response


def _request() -> ProviderRequest:
    return ProviderRequest(
        view="whole",
        prompt_text="rubric",
        prompt_version="whole-grounding-v1",
        model_input="input",
        schema_name="whole_output_v1",
        schema=WHOLE_OUTPUT_SCHEMA,
        model="gpt-5.6-sol",
        reasoning_effort="low",
        max_output_tokens=512,
        valid_sentence_ids=frozenset(),
    )


def _usage() -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=100,
        input_tokens_details=SimpleNamespace(cached_tokens=20, cache_write_tokens=10),
        output_tokens=30,
        output_tokens_details=SimpleNamespace(reasoning_tokens=12),
        total_tokens=130,
    )


def test_completed_provider_response_attaches_semantics_and_usage() -> None:
    response = SimpleNamespace(
        status="completed",
        id="resp_1",
        model="gpt-5.6-sol",
        output_text='{"verdict":"UNSUPPORTED"}',
        output=[],
        usage=_usage(),
    )
    fake = FakeResponses(response)
    outcome = OpenAIResponsesJudge(SimpleNamespace(responses=fake)).call(_request())
    assert outcome.status == "completed"
    assert outcome.semantic_prediction == "UNSUPPORTED"
    assert outcome.usage.reasoning_tokens == 12
    assert fake.calls[0]["store"] is False
    assert fake.calls[0]["tools"] == []
    assert "previous_response_id" not in fake.calls[0]


def test_provider_incomplete_and_refusal_are_operational_failures() -> None:
    incomplete = SimpleNamespace(
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        id="resp_i",
        model="gpt-5.6-sol",
        usage=_usage(),
    )
    outcome = OpenAIResponsesJudge(SimpleNamespace(responses=FakeResponses(incomplete))).call(
        _request()
    )
    assert outcome.status == "provider_incomplete"
    assert outcome.error_type == "max_output_tokens"

    refusal = SimpleNamespace(
        status="completed",
        id="resp_r",
        model="gpt-5.6-sol",
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="refusal", refusal="no")],
            )
        ],
        output_text="",
        usage=_usage(),
    )
    outcome = OpenAIResponsesJudge(SimpleNamespace(responses=FakeResponses(refusal))).call(
        _request()
    )
    assert outcome.status == "provider_refusal"


def test_missing_usage_remains_missing() -> None:
    usage = extract_usage(SimpleNamespace(usage=None))
    assert usage == Usage(None, None, None, None, None, None)
    assert calculate_cost_usd(usage, _pricing()) is None


def _pricing() -> Pricing:
    return Pricing(4.0, 0.4, 5.0, 20.0, 272_000, 2.0, 1.5)


def test_cost_accounts_for_cache_and_does_not_double_count_reasoning() -> None:
    usage = Usage(100, 20, 10, 30, 12, 130)
    expected = ((70 * 4.0) + (20 * 0.4) + (10 * 5.0) + (30 * 20.0)) / 1_000_000
    assert calculate_cost_usd(usage, _pricing()) == expected

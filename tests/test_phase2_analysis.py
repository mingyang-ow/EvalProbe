from pathlib import Path
from types import SimpleNamespace

from evalprobe.phase2.analysis import analyze_frozen_test


def _record(record_id: str, label: str, ids: tuple[int, ...], stratum: str, locality: str):
    reference = SimpleNamespace(
        record_id=record_id,
        source_id=f"source-{record_id}",
        reference_label=label,
        reference_unsupported_sentence_ids=ids,
        burden_stratum=stratum,
        locality=locality,
    )
    return SimpleNamespace(reference=reference)


def _result(record_id: str, view: str, prediction: object) -> dict[str, object]:
    return {
        "call_key": f"frozen-test-v1:{record_id}:{view}:prompt",
        "record_id": record_id,
        "view": view,
        "status": "completed",
        "semantic_prediction": prediction,
        "input_tokens": 10,
        "cached_input_tokens": 2,
        "cache_write_tokens": 0,
        "output_tokens": 3,
        "reasoning_tokens": 1,
        "total_tokens": 13,
        "estimated_cost_usd": 0.001,
        "latency_ms": 100,
    }


def test_frozen_test_analysis_preserves_whole_and_local_disagreements(tmp_path: Path) -> None:
    records = (
        _record("a", "SUPPORTED", (), "none", "NONE"),
        _record("b", "UNSUPPORTED", (2,), "low", "LOCALIZED"),
        _record("c", "UNSUPPORTED", (1, 3), "high", "DISTRIBUTED"),
    )
    calls = tuple(
        SimpleNamespace(call_key=f"frozen-test-v1:{record.reference.record_id}:{view}:prompt")
        for record in records
        for view in ("whole", "local")
    )
    plan = SimpleNamespace(
        records=records,
        calls=calls,
        config={
            "canary": {"run_id": "frozen-test-v1"},
            "judge": {
                "model": "gpt-5.6-sol",
                "reasoning_effort": "low",
                "prompts": {"whole": "whole-grounding-v1", "local": "local-grounding-v1"},
            },
            "local_units": {"version": "sentence-v2"},
        },
    )
    results = [
        _result("a", "whole", "SUPPORTED"),
        _result("a", "local", []),
        _result("b", "whole", "SUPPORTED"),
        _result("b", "local", [2]),
        _result("c", "whole", "UNSUPPORTED"),
        _result("c", "local", [1, 2]),
    ]

    analysis = analyze_frozen_test(plan, results, tmp_path)

    assert analysis["whole_response"]["unsupported"]["false_negative_count"] == 1
    assert analysis["granularity_gap"]["recovery_rate"] == {
        "numerator": 1,
        "denominator": 1,
        "value": 1.0,
    }
    assert analysis["local_sentence_v2"]["false_positive_units"] == 1
    assert analysis["local_sentence_v2"]["false_negative_units"] == 1
    assert analysis["review_queue"]["counts_by_type"] == {
        "FALSE_NEGATIVE": 1,
        "JUDGE_ONLY": 1,
        "REFERENCE_ONLY": 1,
    }
    assert (tmp_path / "burden_detection.svg").is_file()
    assert (tmp_path / "granularity_gap.svg").is_file()
    queue = (tmp_path / "review_queue.jsonl").read_text()
    assert "answer" not in queue and "evidence" not in queue

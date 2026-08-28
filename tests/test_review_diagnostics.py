import json
from pathlib import Path

from evalprobe.review.diagnostics import aggregate_suspicious_units, suspicious_unit_reasons


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_suspicious_unit_diagnostic_flags_formatting_without_classifying() -> None:
    assert "LIST_MARKER_ONLY" in suspicious_unit_reasons("2.")
    assert "NUMBER_ONLY" in suspicious_unit_reasons("42")
    assert "PUNCTUATION_ONLY" in suspicious_unit_reasons("...")
    assert suspicious_unit_reasons("Substantive sentence.") == ()


def test_suspicious_unit_aggregate_counts_train_and_test_without_judge_data(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    _write_jsonl(
        data_dir / "source_info.jsonl",
        [
            {
                "source_id": "source",
                "task_type": "QA",
                "source": "MARCO",
                "source_info": {"question": "q", "passages": "e"},
                "prompt": "p",
            }
        ],
    )
    _write_jsonl(
        data_dir / "response.jsonl",
        [
            {
                "id": split,
                "source_id": "source",
                "model": "m",
                "temperature": 0,
                "labels": [],
                "split": split,
                "quality": "good",
                "response": "1. Substantive sentence.",
            }
            for split in ("train", "test")
        ],
    )
    summary = aggregate_suspicious_units(data_dir)
    assert summary["splits"]["train"]["suspicious_unit_count"] == 1
    assert summary["splits"]["test"]["suspicious_unit_count"] == 1
    assert summary["overall"]["reason_counts"]["NUMBER_ONLY"] == 0
    assert summary["judge_results_used"] is False
    assert summary["methodology_changed"] is False

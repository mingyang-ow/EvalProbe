import json
from pathlib import Path

from evalprobe.phase1c.diagnostics import corpus_segmentation_diagnostics


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_corpus_diagnostics_preserve_span_integrity_and_make_no_judge_calls(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "raw"
    _write_jsonl(
        data_dir / "source_info.jsonl",
        [
            {
                "source_id": "s1",
                "task_type": "QA",
                "source": "fixture",
                "source_info": {"question": "Q?", "passages": "Evidence."},
                "prompt": "fixture",
            }
        ],
    )
    base = {
        "source_id": "s1",
        "model": "fixture",
        "temperature": 0,
        "quality": "good",
    }
    _write_jsonl(
        data_dir / "response.jsonl",
        [
            {
                **base,
                "id": "train-row",
                "split": "train",
                "response": "1. Claim.",
                "labels": [
                    {
                        "start": 3,
                        "end": 9,
                        "text": "Claim.",
                        "label_type": "Baseless Info",
                    }
                ],
            },
            {
                **base,
                "id": "test-row",
                "split": "test",
                "response": "2. Supported.",
                "labels": [],
            },
        ],
    )

    summary = corpus_segmentation_diagnostics(data_dir, tmp_path / "summary.json")
    assert summary["overall"]["list_marker_only_before"] == 2
    assert summary["overall"]["list_marker_only_after"] == 0
    assert summary["span_integrity"] == {
        "matched": 1,
        "mismatch": 0,
        "malformed": 0,
        "out_of_range": 0,
    }
    assert summary["raw_response_text_changed"] is False
    assert summary["judge_calls_made"] == 0
    assert summary["test_data_access"] == "read_only_diagnostics"

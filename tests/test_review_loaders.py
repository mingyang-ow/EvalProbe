import json
from pathlib import Path

from evalprobe.review.loaders import (
    load_phase0_audit_records,
    load_phase1_canary_records,
    phase1_disagreement_targets,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_corpus(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    _write_jsonl(
        data_dir / "source_info.jsonl",
        [
            {
                "source_id": "s1",
                "task_type": "QA",
                "source": "MARCO",
                "source_info": {"question": "Question?", "passages": "Evidence."},
                "prompt": "source-bearing prompt",
            }
        ],
    )
    _write_jsonl(
        data_dir / "response.jsonl",
        [
            {
                "id": "r1",
                "source_id": "s1",
                "model": "generator",
                "temperature": 0.0,
                "labels": [
                    {
                        "start": 3,
                        "end": 12,
                        "text": "Do thing.",
                        "label_type": "Baseless Info",
                        "implicit_true": False,
                    }
                ],
                "split": "train",
                "quality": "good",
                "response": "1. Do thing.",
            }
        ],
    )
    return data_dir


def test_phase0_audit_adapter_loads_source_context_and_existing_units(tmp_path: Path) -> None:
    data_dir = _write_corpus(tmp_path)
    audit_path = tmp_path / "manual_audit.jsonl"
    _write_jsonl(
        audit_path,
        [
            {
                "response_id": "r1",
                "source_id": "s1",
                "split": "train",
                "response_text": "1. Do thing.",
                "annotations": [
                    {
                        "start": 3,
                        "end": 12,
                        "text": "Do thing.",
                        "label_type": "Baseless Info",
                        "implicit_true": False,
                    }
                ],
                "sentences": [
                    {"start": 0, "end": 2, "text": "1.", "reference_label": "SUPPORTED"},
                    {
                        "start": 3,
                        "end": 12,
                        "text": "Do thing.",
                        "reference_label": "UNSUPPORTED",
                    },
                ],
                "reference_label": "UNSUPPORTED",
                "locality": "LOCALIZED",
                "hallucination_burden": 0.5,
            }
        ],
    )
    records = load_phase0_audit_records(audit_path, data_dir)
    assert len(records) == 1
    assert records[0].question == "Question?"
    assert records[0].reference_unsupported_sentence_ids == (2,)
    assert "LIST_MARKER_ONLY" in records[0].sentences[0].suspicious_reasons

    v2_path = tmp_path / "manual_audit_v2.jsonl"
    _write_jsonl(
        v2_path,
        [
            {
                "run_id": "phase0-segmentation-v2",
                "response_id": "r1",
                "source_id": "s1",
                "split": "train",
                "response_text": "1. Do thing.",
                "annotations": [
                    {
                        "start": 3,
                        "end": 12,
                        "text": "Do thing.",
                        "label_type": "Baseless Info",
                        "implicit_true": False,
                    }
                ],
                "sentences": [
                    {
                        "start": 0,
                        "end": 12,
                        "text": "1. Do thing.",
                        "reference_label": "UNSUPPORTED",
                    }
                ],
                "reference_label": "UNSUPPORTED",
                "locality": "LOCALIZED",
                "hallucination_burden": 0.5,
            }
        ],
    )
    repaired = load_phase0_audit_records(v2_path, data_dir)
    assert repaired[0].run_id == "phase0-segmentation-v2"
    assert len(repaired[0].sentences) == 1
    assert repaired[0].reference_unsupported_sentence_ids == (1,)


def test_phase1_adapter_loads_both_views_and_derives_disagreement_targets(
    tmp_path: Path,
) -> None:
    data_dir = _write_corpus(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    results = tmp_path / "results.jsonl"
    features = tmp_path / "features.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "record_id": "r1",
                "source_id": "s1",
                "split": "train",
                "reference_label": "UNSUPPORTED",
                "reference_unsupported_sentence_ids": [2],
                "burden_stratum": "medium",
            }
        ],
    )
    _write_jsonl(
        results,
        [
            {
                "call_key": "run:r1:whole:prompt",
                "record_id": "r1",
                "source_id": "s1",
                "run_id": "run",
                "view": "whole",
                "status": "completed",
                "semantic_prediction": "SUPPORTED",
                "prompt_version": "whole-v1",
            },
            {
                "call_key": "run:r1:local:prompt",
                "record_id": "r1",
                "source_id": "s1",
                "run_id": "run",
                "view": "local",
                "status": "completed",
                "semantic_prediction": [1],
                "prompt_version": "local-v1",
            },
        ],
    )
    _write_jsonl(
        features,
        [
            {
                "response_id": "r1",
                "locality": "LOCALIZED",
                "hallucination_burden": 0.5,
            }
        ],
    )
    records = load_phase1_canary_records(manifest, results, features, data_dir)
    assert [record.view for record in records] == ["whole", "local"]
    local = records[1]
    assert local.false_positive_sentence_ids == (1,)
    assert local.false_negative_sentence_ids == (2,)
    targets = phase1_disagreement_targets(records)
    assert [(target.identity.view, target.identity.sentence_id) for target in targets] == [
        ("local", 1),
        ("local", 2),
        ("whole", None),
    ]

import json
from pathlib import Path

from evalprobe.review.loaders import (
    load_phase0_audit_records,
    load_phase1_canary_records,
    phase1_current_local_disagreement_targets,
    phase1_disagreement_targets,
    phase1_segmentation_repair_outcomes,
)
from evalprobe.review.models import (
    Adjudication,
    HumanClassification,
    ReviewIdentity,
    ReviewKind,
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
    current_local = phase1_current_local_disagreement_targets(records)
    assert [
        (target.identity.view, target.identity.sentence_id, target.mismatch_type)
        for target in current_local
    ] == [
        ("local", 1, "JUDGE_ONLY"),
        ("local", 2, "REFERENCE_ONLY"),
    ]


def test_phase1_v2_adapter_renders_merged_units_with_matching_v2_labels(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    answer = "Intro. 1. Unsupported claim. 2. Exact unsupported."
    _write_jsonl(
        data_dir / "source_info.jsonl",
        [
            {
                "source_id": "source-12839",
                "task_type": "QA",
                "source": "fixture",
                "source_info": {"question": "Question?", "passages": "Evidence."},
                "prompt": "fixture",
            }
        ],
    )
    _write_jsonl(
        data_dir / "response.jsonl",
        [
            {
                "id": "12839-like",
                "source_id": "source-12839",
                "model": "generator",
                "temperature": 0.0,
                "labels": [
                    {
                        "start": 10,
                        "end": 28,
                        "text": "Unsupported claim.",
                        "label_type": "Baseless Info",
                        "implicit_true": False,
                    },
                    {
                        "start": 32,
                        "end": 50,
                        "text": "Exact unsupported.",
                        "label_type": "Baseless Info",
                        "implicit_true": False,
                    },
                ],
                "split": "train",
                "quality": "good",
                "response": answer,
            }
        ],
    )
    manifest = tmp_path / "manifest-v2.jsonl"
    primary_results = tmp_path / "results-v2.jsonl"
    phase1a_results = tmp_path / "results-v1.jsonl"
    features = tmp_path / "features.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "record_id": "12839-like",
                "source_id": "source-12839",
                "split": "train",
                "reference_label": "UNSUPPORTED",
                "reference_unsupported_sentence_ids": [2, 3],
                "burden_stratum": "high",
            }
        ],
    )
    _write_jsonl(
        primary_results,
        [
            {
                "call_key": "v2:12839-like:local:prompt",
                "record_id": "12839-like",
                "source_id": "source-12839",
                "run_id": "v2",
                "view": "local",
                "status": "completed",
                "semantic_prediction": [2, 3],
                "prompt_version": "local-grounding-v1",
                "local_units_version": "sentence-v2",
            }
        ],
    )
    _write_jsonl(
        phase1a_results,
        [
            {
                "call_key": "v1:12839-like:whole:prompt",
                "record_id": "12839-like",
                "source_id": "source-12839",
                "run_id": "v1",
                "view": "whole",
                "status": "completed",
                "semantic_prediction": "UNSUPPORTED",
                "prompt_version": "whole-grounding-v1",
            },
            {
                "call_key": "v1:12839-like:local:prompt",
                "record_id": "12839-like",
                "source_id": "source-12839",
                "run_id": "v1",
                "view": "local",
                "status": "completed",
                "semantic_prediction": [3],
                "prompt_version": "local-grounding-v1",
            },
        ],
    )
    _write_jsonl(features, [{"response_id": "12839-like"}])

    records = load_phase1_canary_records(
        manifest,
        primary_results,
        features,
        data_dir,
        whole_results_path=phase1a_results,
        run_id_override="v2",
        local_units_version="sentence-v2",
    )

    assert [record.view for record in records] == ["whole", "local"]
    for record in records:
        assert record.local_units_version == "sentence-v2"
        assert [
            (unit.sentence_id, unit.start, unit.end, unit.text, unit.reference_label)
            for unit in record.sentences
        ] == [
            (1, 0, 6, "Intro.", "SUPPORTED"),
            (2, 7, 28, "1. Unsupported claim.", "UNSUPPORTED"),
            (3, 29, 50, "2. Exact unsupported.", "UNSUPPORTED"),
        ]
        assert all(unit.text not in {"1.", "2."} for unit in record.sentences)
    assert records[1].judge_prediction == (2, 3)


def test_phase1_v2_adapter_exposes_reference_only_record_without_local_result(
    tmp_path: Path,
) -> None:
    data_dir = _write_corpus(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    primary_results = tmp_path / "results-v2.jsonl"
    phase1a_results = tmp_path / "results-v1.jsonl"
    features = tmp_path / "features.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "record_id": "r1",
                "source_id": "s1",
                "split": "train",
                "reference_label": "UNSUPPORTED",
                "reference_unsupported_sentence_ids": [1],
                "burden_stratum": "high",
            }
        ],
    )
    _write_jsonl(
        primary_results,
        [
            {
                "call_key": "v2:r1:local:prompt",
                "record_id": "r1",
                "source_id": "s1",
                "run_id": "v2",
                "view": "local",
                "status": "provider_error",
            }
        ],
    )
    _write_jsonl(
        phase1a_results,
        [
            {
                "call_key": "v1:r1:whole:prompt",
                "record_id": "r1",
                "source_id": "s1",
                "run_id": "v1",
                "view": "whole",
                "status": "completed",
                "semantic_prediction": "UNSUPPORTED",
                "prompt_version": "whole-grounding-v1",
            },
            {
                "call_key": "v1:r1:local:prompt",
                "record_id": "r1",
                "source_id": "s1",
                "run_id": "v1",
                "view": "local",
                "status": "completed",
                "semantic_prediction": [2],
                "prompt_version": "local-grounding-v1",
            },
        ],
    )
    _write_jsonl(features, [{"response_id": "r1"}])

    records = load_phase1_canary_records(
        manifest,
        primary_results,
        features,
        data_dir,
        whole_results_path=phase1a_results,
        run_id_override="v2",
        local_units_version="sentence-v2",
        allow_missing_local_result=True,
    )

    local = next(record for record in records if record.view == "local")
    assert local.judge_prediction is None
    assert local.prompt_version is None
    assert local.local_units_version == "sentence-v2"
    assert [(unit.text, unit.reference_label) for unit in local.sentences] == [
        ("1. Do thing.", "UNSUPPORTED")
    ]


def test_v1_segmentation_defects_that_become_both_are_resolved_not_queued(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    answer = "Intro. 1. Unsupported claim. 2. Exact unsupported."
    _write_jsonl(
        data_dir / "source_info.jsonl",
        [
            {
                "source_id": "source-12839",
                "task_type": "QA",
                "source": "fixture",
                "source_info": {"question": "Question?", "passages": "Evidence."},
                "prompt": "fixture",
            }
        ],
    )
    _write_jsonl(
        data_dir / "response.jsonl",
        [
            {
                "id": "12839-like",
                "source_id": "source-12839",
                "model": "generator",
                "temperature": 0.0,
                "labels": [],
                "split": "train",
                "quality": "good",
                "response": answer,
            }
        ],
    )
    manifest = tmp_path / "manifest.jsonl"
    primary_results = tmp_path / "results-v2.jsonl"
    phase1a_results = tmp_path / "results-v1.jsonl"
    features = tmp_path / "features.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "record_id": "12839-like",
                "source_id": "source-12839",
                "split": "train",
                "reference_label": "UNSUPPORTED",
                "reference_unsupported_sentence_ids": [2, 3],
                "burden_stratum": "high",
            }
        ],
    )
    _write_jsonl(
        primary_results,
        [
            {
                "call_key": "v2:12839-like:local:prompt",
                "record_id": "12839-like",
                "source_id": "source-12839",
                "run_id": "v2",
                "view": "local",
                "status": "completed",
                "semantic_prediction": [2, 3],
                "prompt_version": "local-grounding-v1",
                "local_units_version": "sentence-v2",
            }
        ],
    )
    _write_jsonl(
        phase1a_results,
        [
            {
                "call_key": "v1:12839-like:whole:prompt",
                "record_id": "12839-like",
                "source_id": "source-12839",
                "run_id": "v1",
                "view": "whole",
                "status": "completed",
                "semantic_prediction": "UNSUPPORTED",
                "prompt_version": "whole-grounding-v1",
            }
        ],
    )
    _write_jsonl(features, [{"response_id": "12839-like"}])
    records = load_phase1_canary_records(
        manifest,
        primary_results,
        features,
        data_dir,
        whole_results_path=phase1a_results,
        run_id_override="v2",
        local_units_version="sentence-v2",
        allow_missing_local_result=True,
    )
    decisions = [
        Adjudication.create(
            identity=ReviewIdentity("train-canary-v1", "12839-like", "local", sentence_id),
            source_id="source-12839",
            review_kind=ReviewKind.JUDGE_DISAGREEMENT,
            classification=HumanClassification.SEGMENTATION_DEFECT,
            reviewed_at="2026-08-28T00:00:00+00:00",
        )
        for sentence_id in (2, 4)
    ]
    decisions.append(
        Adjudication.create(
            identity=ReviewIdentity("train-canary-v1", "12839-like", "whole"),
            source_id="source-12839",
            review_kind=ReviewKind.JUDGE_DISAGREEMENT,
            classification=HumanClassification.SEGMENTATION_DEFECT,
            reviewed_at="2026-08-28T00:00:00+00:00",
        )
    )

    outcomes = phase1_segmentation_repair_outcomes(records, decisions)
    assert [
        (
            outcome.view,
            outcome.old_sentence_id,
            outcome.new_sentence_id,
            outcome.current_category,
            outcome.status,
        )
        for outcome in outcomes
    ] == [
        ("local", 2, 2, "BOTH", "RESOLVED_BY_METHODOLOGY_REPAIR"),
        ("local", 4, 3, "BOTH", "RESOLVED_BY_METHODOLOGY_REPAIR"),
        ("whole", None, None, "WHOLE_VERDICT", "HISTORICAL_V1_ONLY"),
    ]
    assert phase1_current_local_disagreement_targets(records) == []

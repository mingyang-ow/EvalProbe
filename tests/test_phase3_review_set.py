import json
from collections import Counter
from pathlib import Path

from evalprobe.phase3.review_set import (
    LOCAL_JUDGE_ONLY_SAMPLE,
    LOCAL_REFERENCE_ONLY,
    WHOLE_DISAGREEMENTS,
    build_phase3_review_set,
    load_phase3_review_items,
    phase3_adjudication_summary,
    prepare_phase3_review_set,
    write_phase3_error_analysis,
)
from evalprobe.review.models import Adjudication, HumanClassification, ReviewKind
from evalprobe.review.storage import load_adjudications

REPOSITORY_ROOT = Path(__file__).parents[1]
PHASE2 = REPOSITORY_ROOT / "reports/phase2/frozen-test-v1"


def _read(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _build() -> tuple[list[dict[str, object]], dict[str, object]]:
    return build_phase3_review_set(
        _read(PHASE2 / "manifest.jsonl"),
        _read(PHASE2 / "results.jsonl"),
        _read(PHASE2 / "review_queue.jsonl"),
    )


def test_phase3_review_set_is_deterministic_bounded_and_separated() -> None:
    rows, summary = _build()
    repeated_rows, repeated_summary = _build()

    assert rows == repeated_rows
    assert summary == repeated_summary
    groups = Counter(row["review_group"] for row in rows)
    assert groups == {
        WHOLE_DISAGREEMENTS: 12,
        LOCAL_REFERENCE_ONLY: 12,
        LOCAL_JUDGE_ONLY_SAMPLE: 20,
    }
    assert len(rows) == 44
    assert len({row["review_id"] for row in rows}) == 44
    assert all(
        row["view"] == "whole" and row["mismatch_type"] in {"FALSE_POSITIVE", "FALSE_NEGATIVE"}
        for row in rows
        if row["review_group"] == WHOLE_DISAGREEMENTS
    )
    assert all(
        row["view"] == "local" and row["mismatch_type"] == "REFERENCE_ONLY"
        for row in rows
        if row["review_group"] == LOCAL_REFERENCE_ONLY
    )
    assert all(
        row["view"] == "local" and row["mismatch_type"] == "JUDGE_ONLY"
        for row in rows
        if row["review_group"] == LOCAL_JUDGE_ONLY_SAMPLE
    )


def test_phase3_judge_only_sample_has_record_diversity_and_planned_coverage() -> None:
    rows, summary = _build()
    sampled = [row for row in rows if row["review_group"] == LOCAL_JUDGE_ONLY_SAMPLE]
    record_counts = Counter(row["record_id"] for row in sampled)

    assert len(sampled) == 20
    assert len(record_counts) == 20
    assert max(record_counts.values()) <= 2
    assert len({row["source_id"] for row in sampled}) == 20
    assert {row["official_reference_label"] for row in sampled} == {
        "SUPPORTED",
        "UNSUPPORTED",
    }
    assert {row["whole_relation"] for row in sampled} == {"AGREEMENT", "DISAGREEMENT"}
    assert {row["burden_stratum"] for row in sampled} >= {"low", "medium", "high"}
    assert {row["locality"] for row in sampled} >= {"LOCALIZED", "DISTRIBUTED"}
    densities = {
        "few" if int(row["local_judge_only_count"]) <= 2 else "many"
        for row in sampled
        if int(row["local_judge_only_count"]) <= 2 or int(row["local_judge_only_count"]) >= 4
    }
    assert densities == {"few", "many"}
    assert summary["sampling"]["seed"] == 20260828  # type: ignore[index]


def test_phase3_artifact_fields_are_safe_and_single_whole_miss_is_flagged() -> None:
    rows, summary = _build()
    forbidden = {
        "question",
        "evidence",
        "answer",
        "passages",
        "spans",
        "annotations",
        "prompt",
        "prompt_text",
        "model_input",
    }
    assert all(not forbidden.intersection(row) for row in rows)
    flagged = [row for row in rows if row["diagnostic_priority"] is not None]
    assert len(flagged) == 1
    assert flagged[0]["mismatch_type"] == "FALSE_NEGATIVE"
    assert summary["provider_calls"] == 0
    assert summary["api_cost_usd"] == 0.0
    assert summary["human_adjudications_completed"] == 0


def test_phase3_safe_summary_keeps_populations_separate(tmp_path: Path) -> None:
    prepare_phase3_review_set(
        PHASE2 / "manifest.jsonl",
        PHASE2 / "results.jsonl",
        PHASE2 / "review_queue.jsonl",
        tmp_path,
    )
    items = load_phase3_review_items(tmp_path / "review_set.jsonl")
    first = items[0]
    decision = Adjudication.create(
        identity=first.target.identity,
        source_id=first.source_id,
        review_kind=ReviewKind.JUDGE_DISAGREEMENT,
        classification=HumanClassification.JUDGE_ERROR,
        reviewed_at="2026-08-28T00:00:00+00:00",
    )

    summary = phase3_adjudication_summary(items, [decision])

    assert summary["status"] == "HUMAN_REVIEW_REQUIRED"
    assert summary["official_metrics_rewritten"] is False
    assert summary["groups"][WHOLE_DISAGREEMENTS]["reviewed_count"] == 1
    assert summary["groups"][LOCAL_REFERENCE_ONLY]["reviewed_count"] == 0
    assert summary["groups"][LOCAL_JUDGE_ONLY_SAMPLE]["reviewed_count"] == 0
    false_positives = summary["groups"][WHOLE_DISAGREEMENTS]["mismatch_populations"][
        "FALSE_POSITIVE"
    ]
    assert false_positives["classification_counts"]["JUDGE_ERROR"] == 1

    analysis = write_phase3_error_analysis(items, [decision], tmp_path)
    assert analysis["official_metrics_status"] == "UNCHANGED"
    assert analysis["interpretive_limits"]["human_corrected_primary_metric_created"] is False
    assert (tmp_path / "error_analysis.json").is_file()
    assert (tmp_path / "error_analysis.md").is_file()


def test_completed_phase3_human_review_counts_are_frozen() -> None:
    items = load_phase3_review_items(
        REPOSITORY_ROOT / "reports/phase3/frozen-test-error-analysis/review_set.jsonl"
    )
    decisions = load_adjudications(REPOSITORY_ROOT / "reports/review/adjudications.jsonl")

    summary = phase3_adjudication_summary(items, decisions)

    assert summary["status"] == "COMPLETE"
    assert summary["groups"][WHOLE_DISAGREEMENTS]["classification_counts"] == {
        "JUDGE_ERROR": 2,
        "SEGMENTATION_DEFECT": 1,
        "REFERENCE_MAPPING_ARTIFACT": 0,
        "BENCHMARK_AMBIGUITY": 9,
        "RUBRIC_AMBIGUITY": 0,
    }
    assert summary["groups"][LOCAL_REFERENCE_ONLY]["classification_counts"] == {
        "JUDGE_ERROR": 8,
        "SEGMENTATION_DEFECT": 0,
        "REFERENCE_MAPPING_ARTIFACT": 0,
        "BENCHMARK_AMBIGUITY": 4,
        "RUBRIC_AMBIGUITY": 0,
    }
    assert summary["groups"][LOCAL_JUDGE_ONLY_SAMPLE]["classification_counts"] == {
        "JUDGE_ERROR": 0,
        "SEGMENTATION_DEFECT": 0,
        "REFERENCE_MAPPING_ARTIFACT": 0,
        "BENCHMARK_AMBIGUITY": 20,
        "RUBRIC_AMBIGUITY": 0,
    }

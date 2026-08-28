import json
from pathlib import Path

import pytest

from evalprobe.review.loaders import filter_review_targets
from evalprobe.review.models import (
    Adjudication,
    HumanClassification,
    ReviewIdentity,
    ReviewKind,
    ReviewTarget,
    SentenceAuditFailureType,
    SentenceAuditStatus,
)
from evalprobe.review.storage import load_adjudications, review_summary, save_adjudication


def test_review_identity_is_stable_and_includes_sentence_id() -> None:
    first = ReviewIdentity("run", "record", "local", 2)
    second = ReviewIdentity("run", "record", "local", 2)
    other = ReviewIdentity("run", "record", "local", 3)
    assert first.review_id == second.review_id
    assert first.review_id != other.review_id


def test_save_creates_then_updates_without_duplicate_or_corpus_fields(tmp_path: Path) -> None:
    path = tmp_path / "adjudications.jsonl"
    identity = ReviewIdentity("run", "record", "whole")
    first = Adjudication.create(
        identity=identity,
        source_id="source",
        review_kind=ReviewKind.JUDGE_DISAGREEMENT,
        classification=HumanClassification.JUDGE_ERROR,
        note="Initial human note.",
        reviewed_at="2026-08-28T00:00:00+00:00",
    )
    save_adjudication(path, first)
    updated = Adjudication.create(
        identity=identity,
        source_id="source",
        review_kind=ReviewKind.JUDGE_DISAGREEMENT,
        classification=HumanClassification.BENCHMARK_AMBIGUITY,
        note="Updated human note.",
        reviewed_at="2026-08-28T01:00:00+00:00",
    )
    save_adjudication(path, updated)
    decisions = load_adjudications(path)
    assert decisions == [updated]
    serialized = path.read_text(encoding="utf-8")
    assert "Question corpus text" not in serialized
    assert "Evidence corpus text" not in serialized
    assert "Answer corpus text" not in serialized
    persisted_fields = set(json.loads(serialized))
    assert persisted_fields == set(updated.to_dict())
    assert not persisted_fields.intersection(
        {"question", "evidence", "answer", "spans", "annotations", "prompt"}
    )


def test_human_classification_and_sentence_audit_status_are_validated() -> None:
    disagreement = ReviewIdentity("run", "record", "whole")
    with pytest.raises(ValueError, match="exactly one"):
        Adjudication.create(
            identity=disagreement,
            source_id="source",
            review_kind=ReviewKind.JUDGE_DISAGREEMENT,
            reviewed_at="now",
        )
    audit = ReviewIdentity("phase0", "record", "sentence_audit")
    with pytest.raises(ValueError, match="only valid"):
        Adjudication.create(
            identity=audit,
            source_id="source",
            review_kind=ReviewKind.SENTENCE_AUDIT,
            sentence_audit_status=SentenceAuditStatus.PASS,
            failure_type=SentenceAuditFailureType.SEGMENTATION_DEFECT,
            reviewed_at="now",
        )
    decision = Adjudication.create(
        identity=audit,
        source_id="source",
        review_kind=ReviewKind.SENTENCE_AUDIT,
        sentence_audit_status=SentenceAuditStatus.FAIL,
        failure_type=SentenceAuditFailureType.REFERENCE_MAPPING_ARTIFACT,
        reviewed_at="now",
    )
    assert decision.sentence_audit_status == "FAIL"


def test_persisted_enum_values_and_safe_summary_are_validated() -> None:
    identity = ReviewIdentity("run", "record", "whole")
    decision = Adjudication.create(
        identity=identity,
        source_id="source",
        review_kind=ReviewKind.JUDGE_DISAGREEMENT,
        classification=HumanClassification.RUBRIC_AMBIGUITY,
        reviewed_at="now",
    )
    invalid = decision.to_dict()
    invalid["classification"] = "NOT_A_CLASSIFICATION"
    with pytest.raises(ValueError):
        Adjudication.from_dict(invalid)
    summary = review_summary([decision])
    assert summary["phase1a_disagreements"]["reviewed_count"] == 1
    assert summary["phase1a_disagreements"]["classification_counts"]["RUBRIC_AMBIGUITY"] == 1
    assert summary["methodological_decision"] == "HUMAN_REVIEW_REQUIRED"


def test_safe_summary_reports_sentence_audit_versions_separately() -> None:
    decisions = [
        Adjudication.create(
            identity=ReviewIdentity(run_id, record_id, "sentence_audit"),
            source_id="source",
            review_kind=ReviewKind.SENTENCE_AUDIT,
            sentence_audit_status=status,
            failure_type=(
                SentenceAuditFailureType.SEGMENTATION_DEFECT
                if status == SentenceAuditStatus.FAIL
                else None
            ),
            reviewed_at="now",
        )
        for run_id, record_id, status in (
            ("phase0-sentence-audit-v1", "v1-pass", SentenceAuditStatus.PASS),
            ("phase0-sentence-audit-v1", "v1-fail", SentenceAuditStatus.FAIL),
            ("phase0-segmentation-v2", "v2-pass", SentenceAuditStatus.PASS),
        )
    ]

    summary = review_summary(decisions)

    assert summary["phase0_sentence_audit"]["reviewed_count"] == 3
    assert summary["phase0_sentence_audit_versions"]["sentence-v1"]["status_counts"] == {
        "PASS": 1,
        "PASS_WITH_LIMITATION": 0,
        "FAIL": 1,
    }
    assert summary["phase0_sentence_audit_versions"]["sentence-v2"]["status_counts"] == {
        "PASS": 1,
        "PASS_WITH_LIMITATION": 0,
        "FAIL": 0,
    }


def test_review_status_filtering_uses_persisted_identity() -> None:
    reviewed_target = ReviewTarget(
        ReviewIdentity("run", "one", "whole"),
        "source-1",
        ReviewKind.JUDGE_DISAGREEMENT,
    )
    unreviewed_target = ReviewTarget(
        ReviewIdentity("run", "two", "whole"),
        "source-2",
        ReviewKind.JUDGE_DISAGREEMENT,
    )
    decision = Adjudication.create(
        identity=reviewed_target.identity,
        source_id="source-1",
        review_kind=ReviewKind.JUDGE_DISAGREEMENT,
        classification=HumanClassification.JUDGE_ERROR,
        reviewed_at="now",
    )
    decisions = {decision.review_id: decision}
    targets = [reviewed_target, unreviewed_target]
    assert filter_review_targets(targets, decisions, "reviewed") == [reviewed_target]
    assert filter_review_targets(targets, decisions, "unreviewed") == [unreviewed_target]

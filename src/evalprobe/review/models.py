from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from typing import Any


class HumanClassification(StrEnum):
    JUDGE_ERROR = "JUDGE_ERROR"
    SEGMENTATION_DEFECT = "SEGMENTATION_DEFECT"
    REFERENCE_MAPPING_ARTIFACT = "REFERENCE_MAPPING_ARTIFACT"
    BENCHMARK_AMBIGUITY = "BENCHMARK_AMBIGUITY"
    RUBRIC_AMBIGUITY = "RUBRIC_AMBIGUITY"


CLASSIFICATION_DEFINITIONS = {
    HumanClassification.JUDGE_ERROR: (
        "The reference/unit is sensible and the judge prediction is incorrect."
    ),
    HumanClassification.SEGMENTATION_DEFECT: (
        "The deterministic splitter created an inappropriate local semantic unit."
    ),
    HumanClassification.REFERENCE_MAPPING_ARTIFACT: (
        "Segmentation is reasonable, but span overlap produces a misleading sentence reference."
    ),
    HumanClassification.BENCHMARK_AMBIGUITY: (
        "The official annotation is debatable, incomplete, or boundary-sensitive."
    ),
    HumanClassification.RUBRIC_AMBIGUITY: (
        "The current judge rubric reasonably permits the model's interpretation."
    ),
}


class SentenceAuditStatus(StrEnum):
    PASS = "PASS"
    PASS_WITH_LIMITATION = "PASS_WITH_LIMITATION"
    FAIL = "FAIL"


class SentenceAuditFailureType(StrEnum):
    SEGMENTATION_DEFECT = "SEGMENTATION_DEFECT"
    REFERENCE_MAPPING_ARTIFACT = "REFERENCE_MAPPING_ARTIFACT"
    OTHER = "OTHER"


class ReviewKind(StrEnum):
    SENTENCE_AUDIT = "sentence_audit"
    JUDGE_DISAGREEMENT = "judge_disagreement"


@dataclass(frozen=True, slots=True)
class ReviewIdentity:
    run_id: str
    record_id: str
    view: str
    sentence_id: int | None = None

    def __post_init__(self) -> None:
        if not all(value.strip() for value in (self.run_id, self.record_id, self.view)):
            raise ValueError("Review identity fields must be non-empty")
        if self.sentence_id is not None and self.sentence_id < 1:
            raise ValueError("sentence_id must be positive")

    @property
    def review_id(self) -> str:
        value = f"{self.run_id}\x1f{self.record_id}\x1f{self.view}\x1f{self.sentence_id}"
        return hashlib.sha256(value.encode()).hexdigest()[:24]


@dataclass(frozen=True, slots=True)
class SentenceUnit:
    sentence_id: int
    start: int
    end: int
    text: str
    reference_label: str
    suspicious_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HallucinationSpan:
    start: int
    end: int
    text: str
    label_type: str
    implicit_true: bool | None


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    run_id: str
    record_id: str
    source_id: str
    split: str
    view: str
    question: str
    evidence: str
    answer: str
    sentences: tuple[SentenceUnit, ...]
    spans: tuple[HallucinationSpan, ...]
    reference_verdict: str
    reference_unsupported_sentence_ids: tuple[int, ...]
    judge_prediction: str | tuple[int, ...] | None = None
    false_positive_sentence_ids: tuple[int, ...] = ()
    false_negative_sentence_ids: tuple[int, ...] = ()
    prompt_version: str | None = None
    locality: str | None = None
    hallucination_burden: float | None = None
    burden_stratum: str | None = None

    @property
    def has_disagreement(self) -> bool:
        if self.view == "whole":
            return self.judge_prediction != self.reference_verdict
        return bool(self.false_positive_sentence_ids or self.false_negative_sentence_ids)


@dataclass(frozen=True, slots=True)
class ReviewTarget:
    identity: ReviewIdentity
    source_id: str
    kind: ReviewKind
    mismatch_type: str | None = None


@dataclass(frozen=True, slots=True)
class Adjudication:
    review_id: str
    run_id: str
    record_id: str
    source_id: str
    view: str
    sentence_id: int | None
    review_kind: str
    status: str
    classification: str | None
    sentence_audit_status: str | None
    failure_type: str | None
    note: str
    reviewer: str
    reviewed_at: str

    @classmethod
    def create(
        cls,
        *,
        identity: ReviewIdentity,
        source_id: str,
        review_kind: ReviewKind,
        reviewed_at: str,
        classification: HumanClassification | None = None,
        sentence_audit_status: SentenceAuditStatus | None = None,
        failure_type: SentenceAuditFailureType | None = None,
        note: str = "",
    ) -> Adjudication:
        normalized_note = " ".join(note.split())
        if len(normalized_note) > 500:
            raise ValueError("Review note must be at most 500 characters")
        if review_kind == ReviewKind.JUDGE_DISAGREEMENT:
            if (
                classification is None
                or sentence_audit_status is not None
                or failure_type is not None
            ):
                raise ValueError("Disagreement reviews require exactly one human classification")
        elif classification is not None or sentence_audit_status is None:
            raise ValueError("Sentence audits require one sentence-audit status")
        if failure_type is not None and sentence_audit_status != SentenceAuditStatus.FAIL:
            raise ValueError("A failure type is only valid for a failed sentence audit")
        return cls(
            review_id=identity.review_id,
            run_id=identity.run_id,
            record_id=identity.record_id,
            source_id=source_id,
            view=identity.view,
            sentence_id=identity.sentence_id,
            review_kind=review_kind.value,
            status="REVIEWED",
            classification=classification.value if classification else None,
            sentence_audit_status=(sentence_audit_status.value if sentence_audit_status else None),
            failure_type=failure_type.value if failure_type else None,
            note=normalized_note,
            reviewer="human",
            reviewed_at=reviewed_at,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Adjudication:
        expected = {field.name for field in fields(cls)}
        if set(value) != expected:
            raise ValueError("Adjudication contains unknown or missing fields")
        item = cls(**value)
        identity = ReviewIdentity(item.run_id, item.record_id, item.view, item.sentence_id)
        if item.review_id != identity.review_id or item.status != "REVIEWED":
            raise ValueError("Adjudication identity or status is invalid")
        if item.reviewer != "human":
            raise ValueError("Reviewer must be human")
        if item.review_kind == ReviewKind.JUDGE_DISAGREEMENT:
            HumanClassification(item.classification)
            if item.sentence_audit_status is not None or item.failure_type is not None:
                raise ValueError("Disagreement adjudication has sentence-audit fields")
        elif item.review_kind == ReviewKind.SENTENCE_AUDIT:
            audit_status = SentenceAuditStatus(item.sentence_audit_status)
            if item.classification is not None:
                raise ValueError("Sentence audit has a disagreement classification")
            if item.failure_type is not None:
                SentenceAuditFailureType(item.failure_type)
                if audit_status != SentenceAuditStatus.FAIL:
                    raise ValueError("Non-failed audit has a failure type")
        else:
            raise ValueError("Unknown review kind")
        if len(item.note) > 500:
            raise ValueError("Review note must be at most 500 characters")
        return item

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

ReferenceLabel = Literal["SUPPORTED", "UNSUPPORTED"]
Locality = Literal["NONE", "LOCALIZED", "DISTRIBUTED"]


@dataclass(frozen=True)
class TextSpan:
    start: int
    end: int


@dataclass(frozen=True)
class SentenceSpan:
    start: int
    end: int
    label: ReferenceLabel = "SUPPORTED"


@dataclass(frozen=True)
class DerivedResponse:
    response_id: str
    source_id: str
    split: str
    reference_label: ReferenceLabel
    hallucination_burden: float
    span_count: int
    sentence_count: int
    unsupported_sentence_count: int
    affected_sentence_count: int
    locality: Locality
    has_implicit_true: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SpanCheck:
    status: Literal["matched", "mismatch", "malformed", "out_of_range"]
    start: int | None = None
    end: int | None = None
    expected: str | None = None
    observed: str | None = None

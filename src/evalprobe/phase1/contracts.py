from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import Any, Literal

Verdict = Literal["SUPPORTED", "UNSUPPORTED"]

FORBIDDEN_JUDGE_FIELDS = frozenset(
    {
        "labels",
        "annotation",
        "annotations",
        "annotation_text",
        "annotation_comments",
        "implicit_true",
        "reference_label",
        "reference_verdict",
        "hallucination_burden",
        "burden_stratum",
        "locality",
        "model",
        "generator_model",
        "temperature",
        "quality",
        "split",
        "record_id",
        "response_id",
        "source_id",
        "sampling",
    }
)


class StructuredOutputError(ValueError):
    """The provider output was not valid JSON."""


class ContractValidationError(ValueError):
    """The parsed provider output violated the application contract."""


@dataclass(frozen=True, slots=True)
class SafeJudgeSource:
    question: str
    evidence: str
    answer: str

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value.strip() for value in fields_as_values(self)):
            raise ValueError("Judge source fields must be non-empty strings")


@dataclass(frozen=True, slots=True)
class NumberedSentence:
    sentence_id: int
    text: str


@dataclass(frozen=True, slots=True)
class WholeJudgeInput:
    question: str
    evidence: str
    answer: str

    @classmethod
    def from_source(cls, source: SafeJudgeSource) -> WholeJudgeInput:
        return cls(source.question, source.evidence, source.answer)

    def to_model_dict(self) -> dict[str, str]:
        return {"question": self.question, "evidence": self.evidence, "answer": self.answer}


@dataclass(frozen=True, slots=True)
class LocalJudgeInput:
    question: str
    evidence: str
    numbered_sentences: tuple[NumberedSentence, ...]

    def to_model_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "evidence": self.evidence,
            "numbered_sentences": [
                {"sentence_id": sentence.sentence_id, "text": sentence.text}
                for sentence in self.numbered_sentences
            ],
        }

    @property
    def sentence_ids(self) -> frozenset[int]:
        return frozenset(sentence.sentence_id for sentence in self.numbered_sentences)


WHOLE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["SUPPORTED", "UNSUPPORTED"]}},
    "required": ["verdict"],
    "additionalProperties": False,
}

LOCAL_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "unsupported_sentence_ids": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
        }
    },
    "required": ["unsupported_sentence_ids"],
    "additionalProperties": False,
}


def fields_as_values(value: object) -> tuple[object, ...]:
    return tuple(getattr(value, field.name) for field in fields(value))


def assert_no_leakage(model_value: object) -> None:
    if isinstance(model_value, dict):
        forbidden = FORBIDDEN_JUDGE_FIELDS.intersection(model_value)
        if forbidden:
            raise ValueError(f"Reference-only fields reached judge input: {sorted(forbidden)}")
        for nested in model_value.values():
            assert_no_leakage(nested)
    elif isinstance(model_value, (list, tuple)):
        for nested in model_value:
            assert_no_leakage(nested)


def render_model_input(value: WholeJudgeInput | LocalJudgeInput) -> str:
    model_dict = value.to_model_dict()
    assert_no_leakage(model_dict)
    return json.dumps(model_dict, ensure_ascii=False, separators=(",", ":"))


def _parse_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise StructuredOutputError("Provider output was not valid JSON") from error
    if not isinstance(value, dict):
        raise ContractValidationError("Structured output must be an object")
    return value


def parse_whole_output(text: str) -> Verdict:
    value = _parse_object(text)
    if set(value) != {"verdict"} or value["verdict"] not in {"SUPPORTED", "UNSUPPORTED"}:
        raise ContractValidationError("Whole output must contain only a valid verdict")
    return value["verdict"]


def parse_local_output(text: str, valid_sentence_ids: frozenset[int]) -> tuple[int, ...]:
    value = _parse_object(text)
    if set(value) != {"unsupported_sentence_ids"}:
        raise ContractValidationError("Local output must contain only unsupported_sentence_ids")
    sentence_ids = value["unsupported_sentence_ids"]
    if not isinstance(sentence_ids, list) or any(type(item) is not int for item in sentence_ids):
        raise ContractValidationError("unsupported_sentence_ids must be an integer list")
    if len(sentence_ids) != len(set(sentence_ids)):
        raise ContractValidationError("unsupported_sentence_ids must be unique")
    unknown = set(sentence_ids) - valid_sentence_ids
    if unknown:
        raise ContractValidationError(f"Unknown sentence IDs: {sorted(unknown)}")
    return tuple(sorted(sentence_ids))

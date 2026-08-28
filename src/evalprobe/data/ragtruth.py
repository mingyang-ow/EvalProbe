from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from evalprobe.data.models import ReferenceLabel, SpanCheck, TextSpan

SOURCE_REQUIRED_FIELDS = {"source_id", "task_type", "source", "source_info", "prompt"}
RESPONSE_REQUIRED_FIELDS = {
    "id",
    "source_id",
    "model",
    "temperature",
    "labels",
    "split",
    "quality",
    "response",
}
LABEL_REQUIRED_FIELDS = {"start", "end", "text", "label_type"}


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    issues: list[str] = []
    if not path.is_file():
        raise FileNotFoundError(f"Required RAGTruth file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                issues.append(f"{path.name}:{line_number}: invalid JSON: {error.msg}")
                continue
            if not isinstance(value, dict):
                issues.append(f"{path.name}:{line_number}: expected a JSON object")
                continue
            records.append(value)
    return records, issues


def load_dataset(data_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    sources, source_issues = read_jsonl(data_dir / "source_info.jsonl")
    responses, response_issues = read_jsonl(data_dir / "response.jsonl")
    return sources, responses, [*source_issues, *response_issues]


def missing_fields(record: dict[str, Any], required: set[str]) -> list[str]:
    return sorted(required - record.keys())


def reference_label(labels: object) -> ReferenceLabel:
    if not isinstance(labels, list):
        raise ValueError("labels must be a list")
    return "UNSUPPORTED" if labels else "SUPPORTED"


def validate_annotation(response_text: str, annotation: object) -> SpanCheck:
    if not isinstance(annotation, dict):
        return SpanCheck("malformed")
    start = annotation.get("start")
    end = annotation.get("end")
    expected = annotation.get("text")
    if type(start) is not int or type(end) is not int or not isinstance(expected, str):
        return SpanCheck("malformed")
    if start < 0 or end <= start:
        return SpanCheck("malformed", start, end, expected)
    if end > len(response_text):
        return SpanCheck("out_of_range", start, end, expected)
    observed = response_text[start:end]
    if observed != expected:
        return SpanCheck("mismatch", start, end, expected, observed)
    return SpanCheck("matched", start, end, expected, observed)


def matched_spans(response_text: str, labels: Iterable[object]) -> list[TextSpan]:
    spans: list[TextSpan] = []
    for annotation in labels:
        check = validate_annotation(response_text, annotation)
        if check.status != "matched":
            raise ValueError(f"Cannot derive features from {check.status} annotation")
        assert check.start is not None and check.end is not None
        spans.append(TextSpan(check.start, check.end))
    return spans

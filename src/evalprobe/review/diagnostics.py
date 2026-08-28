from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from evalprobe.data.ragtruth import load_dataset
from evalprobe.phase0.sentences import sentence_spans

_LIST_MARKER = re.compile(r"(?:\d+|[A-Za-z])[.)\]:-]|[-*+•]")
SUSPICIOUS_REASON_NAMES = (
    "NUMBER_ONLY",
    "PUNCTUATION_ONLY",
    "LIST_MARKER_ONLY",
    "NO_ALPHABETIC_CHARACTERS",
    "VERY_SHORT_FORMATTING_TOKEN",
    "EMPTY",
)


def _complete_reason_counts(counter: Counter[str]) -> dict[str, int]:
    return {reason: counter[reason] for reason in SUSPICIOUS_REASON_NAMES}


def suspicious_unit_reasons(text: str) -> tuple[str, ...]:
    stripped = text.strip()
    if not stripped:
        return ("EMPTY",)
    reasons: list[str] = []
    if stripped.isdecimal():
        reasons.append("NUMBER_ONLY")
    if all(not character.isalnum() for character in stripped):
        reasons.append("PUNCTUATION_ONLY")
    if _LIST_MARKER.fullmatch(stripped):
        reasons.append("LIST_MARKER_ONLY")
    if not any(character.isalpha() for character in stripped):
        reasons.append("NO_ALPHABETIC_CHARACTERS")
    if len(stripped) <= 3 and not any(character.isalpha() for character in stripped):
        reasons.append("VERY_SHORT_FORMATTING_TOKEN")
    return tuple(reasons)


def aggregate_suspicious_units(data_dir: Path) -> dict[str, Any]:
    sources, responses, issues = load_dataset(data_dir)
    if issues:
        raise ValueError(f"Invalid RAGTruth data: {issues}")
    qa_source_ids = {
        str(source["source_id"]) for source in sources if source.get("task_type") == "QA"
    }
    split_counts: dict[str, dict[str, Any]] = {}
    overall_reasons: Counter[str] = Counter()
    total_responses = 0
    total_units = 0
    total_suspicious = 0
    for split in ("train", "test"):
        eligible = [
            response
            for response in responses
            if str(response.get("source_id")) in qa_source_ids
            and response.get("quality") == "good"
            and response.get("split") == split
        ]
        reasons: Counter[str] = Counter()
        unit_count = 0
        suspicious_count = 0
        for response in eligible:
            answer = response.get("response")
            if not isinstance(answer, str):
                raise ValueError(f"Eligible response {response.get('id')} has invalid text")
            for sentence in sentence_spans(answer):
                unit_count += 1
                flags = suspicious_unit_reasons(answer[sentence.start : sentence.end])
                if flags:
                    suspicious_count += 1
                    reasons.update(flags)
        split_counts[split] = {
            "eligible_response_count": len(eligible),
            "sentence_unit_count": unit_count,
            "suspicious_unit_count": suspicious_count,
            "reason_counts": _complete_reason_counts(reasons),
        }
        total_responses += len(eligible)
        total_units += unit_count
        total_suspicious += suspicious_count
        overall_reasons.update(reasons)
    return {
        "scope": "eligible QA responses; deterministic diagnostics only",
        "methodology_changed": False,
        "judge_results_used": False,
        "splits": split_counts,
        "overall": {
            "eligible_response_count": total_responses,
            "sentence_unit_count": total_units,
            "suspicious_unit_count": total_suspicious,
            "reason_counts": _complete_reason_counts(overall_reasons),
        },
    }

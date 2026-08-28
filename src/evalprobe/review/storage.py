from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from evalprobe.review.models import (
    Adjudication,
    HumanClassification,
    ReviewKind,
    SentenceAuditFailureType,
    SentenceAuditStatus,
)


def load_adjudications(path: Path) -> list[Adjudication]:
    if not path.exists():
        return []
    decisions: list[Adjudication] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid adjudication JSONL at line {line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Invalid adjudication object at line {line_number}")
        decision = Adjudication.from_dict(value)
        if decision.review_id in seen:
            raise ValueError(f"Duplicate adjudication identity at line {line_number}")
        seen.add(decision.review_id)
        decisions.append(decision)
    return decisions


def adjudications_by_id(path: Path) -> dict[str, Adjudication]:
    return {decision.review_id: decision for decision in load_adjudications(path)}


def save_adjudication(path: Path, decision: Adjudication) -> None:
    existing = adjudications_by_id(path)
    existing[decision.review_id] = decision
    ordered = sorted(
        existing.values(),
        key=lambda item: (item.run_id, item.record_id, item.view, item.sentence_id or 0),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for item in ordered:
                handle.write(
                    json.dumps(item.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise


def review_summary(decisions: list[Adjudication]) -> dict[str, Any]:
    phase0 = [
        decision for decision in decisions if decision.review_kind == ReviewKind.SENTENCE_AUDIT
    ]
    phase1 = [
        decision for decision in decisions if decision.review_kind == ReviewKind.JUDGE_DISAGREEMENT
    ]
    audit_statuses = Counter(decision.sentence_audit_status for decision in phase0)
    failure_types = Counter(decision.failure_type for decision in phase0 if decision.failure_type)
    classifications = Counter(decision.classification for decision in phase1)
    return {
        "phase0_sentence_audit": {
            "reviewed_count": len(phase0),
            "status_counts": {
                status.value: audit_statuses[status.value] for status in SentenceAuditStatus
            },
            "failure_type_counts": {
                failure.value: failure_types[failure.value] for failure in SentenceAuditFailureType
            },
        },
        "phase1a_disagreements": {
            "reviewed_count": len(phase1),
            "classification_counts": {
                classification.value: classifications[classification.value]
                for classification in HumanClassification
            },
        },
        "methodological_decision": "HUMAN_REVIEW_REQUIRED",
    }


def write_safe_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

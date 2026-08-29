from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalprobe.phase1.persistence import write_json, write_jsonl
from evalprobe.review.models import (
    Adjudication,
    HumanClassification,
    ReviewIdentity,
    ReviewKind,
    ReviewTarget,
)

WHOLE_DISAGREEMENTS = "WHOLE_DISAGREEMENTS"
LOCAL_REFERENCE_ONLY = "LOCAL_REFERENCE_ONLY"
LOCAL_JUDGE_ONLY_SAMPLE = "LOCAL_JUDGE_ONLY_SAMPLE"
PHASE3_GROUPS = (WHOLE_DISAGREEMENTS, LOCAL_REFERENCE_ONLY, LOCAL_JUDGE_ONLY_SAMPLE)
DEFAULT_SEED = 20260828
DEFAULT_JUDGE_ONLY_TARGET = 20
DEFAULT_MAX_PER_RECORD = 2


@dataclass(frozen=True, slots=True)
class Phase3ReviewItem:
    review_id: str
    run_id: str
    record_id: str
    source_id: str
    view: str
    sentence_id: int | None
    mismatch_type: str
    review_group: str
    official_reference_label: str
    whole_judge_prediction: str
    whole_relation: str
    burden_stratum: str
    locality: str
    local_judge_only_count: int
    sampling_seed: int | None
    sample_rank: int | None
    diagnostic_priority: str | None

    @property
    def target(self) -> ReviewTarget:
        identity = ReviewIdentity(self.run_id, self.record_id, self.view, self.sentence_id)
        if identity.review_id != self.review_id:
            raise ValueError("Phase 3 review item identity hash changed")
        return ReviewTarget(
            identity=identity,
            source_id=self.source_id,
            kind=ReviewKind.JUDGE_DISAGREEMENT,
            mismatch_type=self.mismatch_type,
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def _stable_key(seed: int, row: dict[str, Any]) -> str:
    value = f"{seed}\x1f{row['record_id']}\x1f{row['sentence_id']}"
    return hashlib.sha256(value.encode()).hexdigest()


def _density(count: int) -> str:
    if count <= 2:
        return "few"
    if count >= 4:
        return "many"
    return "moderate"


def _coverage_tags(row: dict[str, Any]) -> set[str]:
    tags = {
        f"reference:{row['official_reference_label']}",
        f"whole:{row['whole_relation']}",
        f"density:{_density(int(row['local_judge_only_count']))}",
    }
    if row["burden_stratum"] in {"low", "medium", "high"}:
        tags.add(f"burden:{row['burden_stratum']}")
    if row["locality"] in {"LOCALIZED", "DISTRIBUTED"}:
        tags.add(f"locality:{row['locality']}")
    return tags


def _sample_judge_only(
    candidates: list[dict[str, Any]],
    *,
    target: int,
    seed: int,
    max_per_record: int,
) -> list[dict[str, Any]]:
    if target < 1 or max_per_record not in {1, 2}:
        raise ValueError("Phase 3 sample target/max-per-record is invalid")
    if len(candidates) < target:
        raise ValueError("Not enough local JUDGE_ONLY candidates")
    desired = {
        "reference:SUPPORTED",
        "reference:UNSUPPORTED",
        "whole:AGREEMENT",
        "whole:DISAGREEMENT",
        "burden:low",
        "burden:medium",
        "burden:high",
        "locality:LOCALIZED",
        "locality:DISTRIBUTED",
        "density:few",
        "density:many",
    }
    available = set().union(*(_coverage_tags(row) for row in candidates))
    uncovered = desired & available
    remaining = list(candidates)
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    while len(selected) < target:
        needed = target - len(selected)
        new_record_candidates = [row for row in remaining if counts[str(row["record_id"])] == 0]
        eligible_records = {str(row["record_id"]) for row in new_record_candidates}
        pool = (
            new_record_candidates
            if len(eligible_records) >= needed
            else [row for row in remaining if counts[str(row["record_id"])] < max_per_record]
        )
        if not pool:
            raise ValueError("Phase 3 diversity constraint cannot satisfy sample target")
        chosen = min(
            pool,
            key=lambda row: (
                -len(_coverage_tags(row) & uncovered),
                _stable_key(seed, row),
            ),
        )
        selected.append(chosen)
        counts[str(chosen["record_id"])] += 1
        uncovered -= _coverage_tags(chosen)
        remaining.remove(chosen)
    if uncovered:
        raise ValueError(f"Phase 3 sample missed available coverage: {sorted(uncovered)}")
    return selected


def build_phase3_review_set(
    manifest: list[dict[str, Any]],
    results: list[dict[str, Any]],
    queue: list[dict[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    judge_only_target: int = DEFAULT_JUDGE_ONLY_TARGET,
    max_per_record: int = DEFAULT_MAX_PER_RECORD,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_by_record = {str(row["record_id"]): row for row in manifest}
    latest = {str(row["call_key"]): row for row in results}
    completed = [row for row in latest.values() if row.get("status") == "completed"]
    if len(completed) != 120:
        raise ValueError("Phase 3 requires all 120 frozen TEST results")
    whole_by_record = {
        str(row["record_id"]): str(row["semantic_prediction"])
        for row in completed
        if row.get("view") == "whole"
    }
    judge_only_counts = Counter(
        str(row["record_id"]) for row in queue if row.get("mismatch_type") == "JUDGE_ONLY"
    )
    queue_counts = Counter(str(row.get("mismatch_type")) for row in queue)
    if queue_counts != Counter(
        {"FALSE_NEGATIVE": 1, "FALSE_POSITIVE": 11, "REFERENCE_ONLY": 12, "JUDGE_ONLY": 71}
    ):
        raise ValueError("Frozen TEST review queue counts changed")

    def enrich(row: dict[str, Any], group: str) -> dict[str, Any]:
        record_id = str(row["record_id"])
        metadata = manifest_by_record[record_id]
        reference = str(metadata["reference_label"])
        whole_prediction = whole_by_record[record_id]
        identity = ReviewIdentity(
            str(row["run_id"]),
            record_id,
            str(row["view"]),
            row.get("sentence_id"),
        )
        return {
            "review_id": identity.review_id,
            "run_id": str(row["run_id"]),
            "record_id": record_id,
            "source_id": str(row["source_id"]),
            "view": str(row["view"]),
            "sentence_id": row.get("sentence_id"),
            "mismatch_type": str(row["mismatch_type"]),
            "review_group": group,
            "official_reference_label": reference,
            "whole_judge_prediction": whole_prediction,
            "whole_relation": ("AGREEMENT" if reference == whole_prediction else "DISAGREEMENT"),
            "burden_stratum": str(metadata["burden_stratum"]),
            "locality": str(metadata["locality"]),
            "local_judge_only_count": judge_only_counts[record_id],
            "sampling_seed": seed if group == LOCAL_JUDGE_ONLY_SAMPLE else None,
            "sample_rank": None,
            "diagnostic_priority": (
                "SINGLE_WHOLE_FALSE_NEGATIVE" if row["mismatch_type"] == "FALSE_NEGATIVE" else None
            ),
        }

    whole = [
        enrich(row, WHOLE_DISAGREEMENTS)
        for row in queue
        if row["mismatch_type"] in {"FALSE_POSITIVE", "FALSE_NEGATIVE"}
    ]
    reference_only = [
        enrich(row, LOCAL_REFERENCE_ONLY)
        for row in queue
        if row["mismatch_type"] == "REFERENCE_ONLY"
    ]
    judge_only_candidates = [
        enrich(row, LOCAL_JUDGE_ONLY_SAMPLE)
        for row in queue
        if row["mismatch_type"] == "JUDGE_ONLY"
    ]
    sampled = _sample_judge_only(
        judge_only_candidates,
        target=judge_only_target,
        seed=seed,
        max_per_record=max_per_record,
    )
    sampled = [dict(row, sample_rank=index) for index, row in enumerate(sampled, 1)]
    rows = (
        sorted(
            whole + reference_only,
            key=lambda row: (
                PHASE3_GROUPS.index(str(row["review_group"])),
                str(row["record_id"]),
                int(row["sentence_id"] or 0),
            ),
        )
        + sampled
    )
    review_ids = [str(row["review_id"]) for row in rows]
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("Phase 3 review set contains duplicate identities")
    sampled_counts = Counter(str(row["record_id"]) for row in sampled)
    coverage = Counter(tag for row in sampled for tag in _coverage_tags(row))
    summary = {
        "phase": "phase3",
        "run_id": "frozen-test-v1",
        "status": "READY_FOR_HUMAN_REVIEW",
        "group_counts": {
            WHOLE_DISAGREEMENTS: len(whole),
            LOCAL_REFERENCE_ONLY: len(reference_only),
            LOCAL_JUDGE_ONLY_SAMPLE: len(sampled),
        },
        "total_review_items": len(rows),
        "local_judge_only_population": len(judge_only_candidates),
        "sampling": {
            "seed": seed,
            "target": judge_only_target,
            "selected_record_count": len(sampled_counts),
            "maximum_items_per_record": max(sampled_counts.values()),
            "configured_maximum_items_per_record": max_per_record,
            "coverage_counts": dict(sorted(coverage.items())),
            "selection_basis": "safe metadata only; corpus text excluded",
        },
        "duplicate_review_ids": 0,
        "provider_calls": 0,
        "api_cost_usd": 0.0,
        "human_adjudications_completed": 0,
    }
    return rows, summary


def prepare_phase3_review_set(
    manifest_path: Path,
    results_path: Path,
    queue_path: Path,
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    judge_only_target: int = DEFAULT_JUDGE_ONLY_TARGET,
    max_per_record: int = DEFAULT_MAX_PER_RECORD,
) -> dict[str, Any]:
    rows, summary = build_phase3_review_set(
        _read_jsonl(manifest_path),
        _read_jsonl(results_path),
        _read_jsonl(queue_path),
        seed=seed,
        judge_only_target=judge_only_target,
        max_per_record=max_per_record,
    )
    write_jsonl(output_dir / "review_set.jsonl", rows)
    write_json(output_dir / "review_set_summary.json", summary)
    return summary


def load_phase3_review_items(path: Path) -> list[Phase3ReviewItem]:
    items = [Phase3ReviewItem(**row) for row in _read_jsonl(path)]
    if any(item.review_group not in PHASE3_GROUPS for item in items):
        raise ValueError("Unknown Phase 3 review group")
    review_ids = [item.review_id for item in items]
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("Duplicate Phase 3 review identity")
    for item in items:
        _ = item.target
    return items


def phase3_targets_by_group(
    items: list[Phase3ReviewItem],
) -> dict[str, list[ReviewTarget]]:
    return {
        group: [item.target for item in items if item.review_group == group]
        for group in PHASE3_GROUPS
    }


def phase3_adjudication_summary(
    items: list[Phase3ReviewItem], decisions: list[Adjudication]
) -> dict[str, Any]:
    decisions_by_id = {decision.review_id: decision for decision in decisions}

    def summarize(group_items: list[Phase3ReviewItem]) -> dict[str, Any]:
        reviewed = [
            decisions_by_id[item.review_id]
            for item in group_items
            if item.review_id in decisions_by_id
        ]
        classifications = Counter(decision.classification for decision in reviewed)
        return {
            "target_count": len(group_items),
            "reviewed_count": len(reviewed),
            "unreviewed_count": len(group_items) - len(reviewed),
            "classification_counts": {
                classification.value: classifications[classification.value]
                for classification in HumanClassification
            },
        }

    groups = {
        group: summarize([item for item in items if item.review_group == group])
        for group in PHASE3_GROUPS
    }
    whole_items = [item for item in items if item.review_group == WHOLE_DISAGREEMENTS]
    groups[WHOLE_DISAGREEMENTS]["mismatch_populations"] = {
        mismatch: summarize([item for item in whole_items if item.mismatch_type == mismatch])
        for mismatch in ("FALSE_POSITIVE", "FALSE_NEGATIVE")
    }
    return {
        "status": (
            "COMPLETE"
            if all(group["unreviewed_count"] == 0 for group in groups.values())
            else "HUMAN_REVIEW_REQUIRED"
        ),
        "groups": groups,
        "official_metrics_rewritten": False,
    }


def write_phase3_error_analysis(
    items: list[Phase3ReviewItem],
    decisions: list[Adjudication],
    output_dir: Path,
) -> dict[str, Any]:
    summary = phase3_adjudication_summary(items, decisions)
    groups = summary["groups"]
    whole = groups[WHOLE_DISAGREEMENTS]
    false_positives = whole["mismatch_populations"]["FALSE_POSITIVE"]
    false_negatives = whole["mismatch_populations"]["FALSE_NEGATIVE"]
    reference_only = groups[LOCAL_REFERENCE_ONLY]
    judge_only = groups[LOCAL_JUDGE_ONLY_SAMPLE]
    analysis = {
        **summary,
        "reviewed_total": sum(group["reviewed_count"] for group in groups.values()),
        "target_total": sum(group["target_count"] for group in groups.values()),
        "official_metrics_status": "UNCHANGED",
        "interpretive_limits": {
            "judge_only_sample_is_purposive": True,
            "judge_only_population_size": 71,
            "judge_only_sample_size": judge_only["target_count"],
            "granularity_recovery_numerator": 0,
            "granularity_recovery_denominator": 1,
            "human_corrected_primary_metric_created": False,
        },
    }
    write_json(output_dir / "error_analysis.json", analysis)

    def count(population: dict[str, Any], classification: str) -> int:
        return int(population["classification_counts"][classification])

    def classification_row(label: str, population: dict[str, Any]) -> str:
        values = (
            count(population, "JUDGE_ERROR"),
            count(population, "BENCHMARK_AMBIGUITY"),
            count(population, "SEGMENTATION_DEFECT"),
            count(population, "REFERENCE_MAPPING_ARTIFACT"),
            count(population, "RUBRIC_AMBIGUITY"),
        )
        rendered = " | ".join(str(value) for value in values)
        return f"| {label} (n={population['target_count']}) | {rendered} |"

    whole_rows = "\n".join(
        (
            classification_row("False positive", false_positives),
            classification_row("False negative", false_negatives),
        )
    )
    reference_row = classification_row("Reference only", reference_only)
    judge_row = classification_row("Judge-only sample", judge_only)
    status_line = (
        f"Status: **{summary['status']}**. Reviewed: "
        f"**{analysis['reviewed_total']} / {analysis['target_total']}**."
    )
    whole_fp_line = (
        f"- Apparent whole false positives: "
        f"{count(false_positives, 'BENCHMARK_AMBIGUITY')}/"
        f"{false_positives['target_count']} benchmark ambiguity; "
        f"{count(false_positives, 'JUDGE_ERROR')} judge error; "
        f"{count(false_positives, 'SEGMENTATION_DEFECT')} segmentation defect."
    )
    whole_fn_line = (
        f"- Whole false negatives: {count(false_negatives, 'JUDGE_ERROR')}/"
        f"{false_negatives['target_count']} judge error."
    )
    reference_line = (
        f"Of {reference_only['target_count']} official unsupported units missed by the local "
        f"judge, {count(reference_only, 'JUDGE_ERROR')} were classified as judge errors and "
        f"{count(reference_only, 'BENCHMARK_AMBIGUITY')} as benchmark ambiguity."
    )
    judge_line = (
        f"Of {judge_only['target_count']} sampled judge-only units, "
        f"{count(judge_only, 'BENCHMARK_AMBIGUITY')} were classified as benchmark ambiguity and "
        f"{count(judge_only, 'SEGMENTATION_DEFECT')} as a segmentation defect."
    )

    interpretation = (
        """- The review does not support treating every apparent whole false positive or local
  judge-only flag as judge over-calling; benchmark incompleteness was the dominant human
  classification in both reviewed populations.
- Local judging also made genuine misses: 8/12 local reference-only items were classified as judge
  errors.
- The frozen burden counts (9/10, 10/10, 10/10) remain weak evidence for a burden effect.
- Granularity recovery remains 0/1 and therefore underpowered and inconclusive.
- Two segmentation classifications are recorded for future methodology review. Phase 3 makes no
  segmentation repair and changes no frozen result."""
        if summary["status"] == "COMPLETE"
        else "Interpretation is deferred until all 44 bounded review items are complete."
    )
    report = f"""# Frozen TEST human error analysis

{status_line}
Official Sol-versus-RAGTruth metrics remain unchanged; no human-corrected primary metric is created.

## Whole disagreements

| Official mismatch | Judge error | Benchmark ambiguity | Segmentation | Mapping | Rubric |
|---|---:|---:|---:|---:|---:|
{whole_rows}

{whole_fp_line}
{whole_fn_line}

A whole-view segmentation classification is retained as entered, but whole judging does not
consume sentence units, so it is a methodological flag rather than a recoded verdict.

## Local reference-only misses

| Population | Judge error | Benchmark ambiguity | Segmentation | Mapping | Rubric |
|---|---:|---:|---:|---:|---:|
{reference_row}

{reference_line}

## Sampled local judge-only flags

| Population | Judge error | Benchmark ambiguity | Segmentation | Mapping | Rubric |
|---|---:|---:|---:|---:|---:|
{judge_row}

{judge_line} This was a deterministic coverage sample, not a probability sample; its exact rate
must not be projected onto all 71 judge-only units.

## Interpretation boundaries

{interpretation}
"""
    (output_dir / "error_analysis.md").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "error_analysis.md").write_text(report, encoding="utf-8")
    return analysis

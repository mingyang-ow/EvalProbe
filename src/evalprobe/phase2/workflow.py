from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, cast

import yaml

from evalprobe.data.ragtruth import load_dataset, matched_spans
from evalprobe.phase0.sampling import hallucination_burden
from evalprobe.phase0.sentences import affected_sentence_indices, segment_local_units
from evalprobe.phase1.contracts import SafeJudgeSource, Verdict
from evalprobe.phase1.persistence import write_json, write_jsonl
from evalprobe.phase1.prompts import get_prompt
from evalprobe.phase1.runner import (
    CanaryPlan,
    build_plan_for_records,
    manifest_rows,
    preflight_summary,
)
from evalprobe.phase1.selection import CanaryReference, SelectedCanaryRecord


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Invalid JSONL object at {path}:{line_number}")
        rows.append(value)
    return rows


def load_phase2_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Phase 2 config must be a YAML mapping")
    canary = value.get("canary", {})
    judge = value.get("judge", {})
    budget = value.get("budget", {})
    local_units = value.get("local_units", {})
    frozen_pilot = value.get("frozen_pilot", {})
    if canary.get("split") != "test" or canary.get("total") != 60:
        raise ValueError("Phase 2 requires the frozen 60-record TEST pilot")
    if canary.get("supported") != 30 or canary.get("unsupported") != 30:
        raise ValueError("Phase 2 requires a 30/30 TEST balance")
    if canary.get("unsupported_strata") != {"low": 10, "medium": 10, "high": 10}:
        raise ValueError("Phase 2 requires 10/10/10 unsupported burden strata")
    if canary.get("max_per_source_id") != 1 or canary.get("expected_calls") != 120:
        raise ValueError("Phase 2 requires 120 calls and unique source IDs")
    if judge.get("model") != "gpt-5.6-sol" or judge.get("reasoning_effort") != "low":
        raise ValueError("Phase 2 freezes gpt-5.6-sol with low reasoning")
    if judge.get("automatic_retries") != 0 or judge.get("fallback_models") != 0:
        raise ValueError("Phase 2 prohibits retries and fallback models")
    if judge.get("views") != ["whole", "local"]:
        raise ValueError("Phase 2 requires independent whole and local calls")
    if judge.get("prompts") != {
        "whole": "whole-grounding-v1",
        "local": "local-grounding-v1",
    }:
        raise ValueError("Phase 2 prompt versions are frozen")
    if local_units.get("version") != "sentence-v2":
        raise ValueError("Phase 2 local units are frozen at sentence-v2")
    if float(budget.get("hard_cap_usd", 0)) != 1.50:
        raise ValueError("Phase 2 hard cap must remain USD $1.50")
    if budget.get("maximum_paid_calls") != 120:
        raise ValueError("Phase 2 permits at most 120 paid calls")
    manifest_hash = frozen_pilot.get("sha256")
    if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
        raise ValueError("Phase 2 requires a frozen pilot SHA-256")
    for view in ("whole", "local"):
        prompt = get_prompt(str(judge["prompts"][view]))
        if prompt.version != judge["prompts"][view]:
            raise ValueError(f"Frozen prompt was not found: {view}")
    return value


def validate_frozen_manifest(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    canary = config["canary"]
    required = {
        "response_id",
        "source_id",
        "reference_label",
        "hallucination_burden",
        "burden_stratum",
        "locality",
        "span_count",
        "sentence_count",
        "affected_sentence_count",
    }
    if any(not required.issubset(row) for row in rows):
        raise ValueError("Frozen manifest is missing required safe metadata")
    reference_counts = Counter(str(row["reference_label"]) for row in rows)
    stratum_counts = Counter(
        str(row["burden_stratum"])
        for row in rows
        if row["reference_label"] == "UNSUPPORTED"
    )
    unique_sources = len({str(row["source_id"]) for row in rows})
    unique_records = len({str(row["response_id"]) for row in rows})
    if len(rows) != canary["total"] or unique_records != canary["total"]:
        raise ValueError("Frozen manifest must contain 60 unique records")
    if reference_counts != Counter({"SUPPORTED": 30, "UNSUPPORTED": 30}):
        raise ValueError("Frozen manifest does not have the required 30/30 labels")
    if stratum_counts != Counter({"low": 10, "medium": 10, "high": 10}):
        raise ValueError("Frozen manifest does not have 10/10/10 burden strata")
    if unique_sources != canary["total"]:
        raise ValueError("Frozen manifest source IDs are not unique")
    return {
        "record_count": len(rows),
        "reference_counts": dict(sorted(reference_counts.items())),
        "unsupported_stratum_counts": dict(sorted(stratum_counts.items())),
        "unique_record_ids": unique_records,
        "unique_source_ids": unique_sources,
    }


def validate_train_freeze(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    v2 = summary["phase0_sentence_audit_versions"]["sentence-v2"]
    phase1c = summary["phase1c_sentence_v2_disagreements"]
    classifications = phase1c["classification_counts"]
    if v2["reviewed_count"] != 20 or v2["status_counts"] != {
        "FAIL": 0,
        "PASS": 20,
        "PASS_WITH_LIMITATION": 0,
    }:
        raise ValueError("Sentence-v2 human validation is incomplete")
    if phase1c["target_count"] != 7 or phase1c["unreviewed_count"] != 0:
        raise ValueError("Phase 1C disagreement adjudication is incomplete")
    for defect in ("SEGMENTATION_DEFECT", "REFERENCE_MAPPING_ARTIFACT", "RUBRIC_AMBIGUITY"):
        if classifications[defect] != 0:
            raise ValueError(f"TRAIN freeze blocked by {defect}")
    return {
        "sentence_v2_pass": v2["status_counts"]["PASS"],
        "phase1c_reviewed": phase1c["reviewed_count"],
        "phase1c_unreviewed": phase1c["unreviewed_count"],
        "phase1c_classification_counts": classifications,
    }


def _records_from_frozen_manifest(
    rows: list[dict[str, Any]], data_dir: Path
) -> list[SelectedCanaryRecord]:
    sources, responses, issues = load_dataset(data_dir)
    if issues:
        raise ValueError(f"Invalid RAGTruth files: {issues}")
    selected_ids = {str(row["response_id"]) for row in rows}
    row_by_id = {str(row["response_id"]): row for row in rows}
    source_by_id = {str(source["source_id"]): source for source in sources}
    response_by_id = {
        str(response["id"]): response
        for response in responses
        if str(response.get("id")) in selected_ids
    }
    if set(response_by_id) != selected_ids:
        raise ValueError("Frozen TEST IDs were not all found in RAGTruth")
    records: list[SelectedCanaryRecord] = []
    for record_id in sorted(selected_ids):
        frozen = row_by_id[record_id]
        response = response_by_id[record_id]
        if response.get("split") != "test" or response.get("quality") != "good":
            raise ValueError(f"Frozen response {record_id} is not eligible TEST data")
        source_id = str(response["source_id"])
        if source_id != str(frozen["source_id"]):
            raise ValueError(f"Frozen source identity changed for {record_id}")
        source = source_by_id[source_id]
        if source.get("task_type") != "QA":
            raise ValueError(f"Frozen response {record_id} is not QA")
        source_info = source.get("source_info")
        if not isinstance(source_info, dict):
            raise ValueError(f"Frozen source {source_id} has invalid source_info")
        question = source_info.get("question")
        evidence = source_info.get("passages")
        answer = response.get("response")
        labels = response.get("labels")
        if not all(isinstance(value, str) for value in (question, evidence, answer)):
            raise ValueError(f"Frozen response {record_id} has invalid judge-visible fields")
        if not isinstance(labels, list):
            raise ValueError(f"Frozen response {record_id} has invalid labels")
        assert isinstance(question, str) and isinstance(evidence, str) and isinstance(answer, str)
        spans = matched_spans(answer, labels)
        if len(spans) != len(labels) or len(spans) != int(frozen["span_count"]):
            raise ValueError(f"Frozen span count changed for {record_id}")
        burden = hallucination_burden(answer, spans)
        if not math.isclose(burden, float(frozen["hallucination_burden"]), abs_tol=1e-15):
            raise ValueError(f"Frozen burden changed for {record_id}")
        units = segment_local_units(answer, "sentence-v2").units
        unsupported_ids = tuple(
            sorted(index + 1 for index in affected_sentence_indices(list(units), spans))
        )
        reference_label = cast(Verdict, frozen["reference_label"])
        if reference_label not in {"SUPPORTED", "UNSUPPORTED"}:
            raise ValueError(f"Invalid frozen reference label for {record_id}")
        if (reference_label == "SUPPORTED") != (not unsupported_ids):
            raise ValueError(f"Frozen whole/local reference mismatch for {record_id}")
        records.append(
            SelectedCanaryRecord(
                source=SafeJudgeSource(question, evidence, answer),
                reference=CanaryReference(
                    record_id=record_id,
                    source_id=source_id,
                    reference_label=reference_label,
                    burden_stratum=str(frozen["burden_stratum"]),
                    reference_unsupported_sentence_ids=unsupported_ids,
                    split="test",
                    hallucination_burden=burden,
                    locality=str(frozen["locality"]),
                ),
            )
        )
    return records


def build_frozen_test_plan(
    config: dict[str, Any], data_dir: Path, repository_root: Path
) -> tuple[CanaryPlan, dict[str, Any]]:
    frozen_path = repository_root / str(config["frozen_pilot"]["path"])
    actual_hash = sha256_file(frozen_path)
    expected_hash = str(config["frozen_pilot"]["sha256"])
    if actual_hash != expected_hash:
        raise ValueError("Frozen TEST manifest SHA-256 changed")
    rows = _read_jsonl(frozen_path)
    manifest_gate = validate_frozen_manifest(rows, config)
    train_gate = validate_train_freeze(
        repository_root / str(config["prerequisites"]["review_summary"])
    )
    plan = build_plan_for_records(config, _records_from_frozen_manifest(rows, data_dir))
    return plan, {
        "frozen_manifest_path": str(config["frozen_pilot"]["path"]),
        "frozen_manifest_sha256": actual_hash,
        "manifest_gate": manifest_gate,
        "train_freeze_gate": train_gate,
    }


def phase2_preflight_summary(
    plan: CanaryPlan,
    freeze: dict[str, Any],
    output_dir: Path,
    max_cost_usd: float,
    repository_root: Path,
) -> dict[str, Any]:
    summary = preflight_summary(plan, max_cost_usd)
    required_docs = [
        repository_root / "story.md",
        repository_root / "docs/INDEX.md",
        repository_root / "docs/decisions/methodology-freeze.md",
        repository_root / "docs/experiments/frozen-test.md",
    ]
    summary.update(freeze)
    summary.update(
        {
            "run_manifest_record_count": len(manifest_rows(plan)),
            "prompt_versions": plan.config["judge"]["prompts"],
            "automatic_retries": plan.config["judge"]["automatic_retries"],
            "fallback_models": plan.config["judge"]["fallback_models"],
            "existing_results_artifact": (output_dir / "results.jsonl").exists(),
            "documentation_freeze_files_present": all(path.is_file() for path in required_docs),
            "conservative_cost_gate": (
                "pass" if summary["estimated_max_cost_usd"] <= max_cost_usd else "fail"
            ),
        }
    )
    summary["pre_execution_gate"] = (
        "pass"
        if summary["conservative_cost_gate"] == "pass"
        and not summary["existing_results_artifact"]
        and summary["documentation_freeze_files_present"]
        else "fail"
    )
    return summary


def write_phase2_dry_run(
    plan: CanaryPlan,
    freeze: dict[str, Any],
    output_dir: Path,
    max_cost_usd: float,
    repository_root: Path,
) -> dict[str, Any]:
    summary = phase2_preflight_summary(
        plan, freeze, output_dir, max_cost_usd, repository_root
    )
    write_jsonl(output_dir / "manifest.jsonl", manifest_rows(plan))
    write_json(output_dir / "dry_run.json", summary)
    return summary

from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from evalprobe.data.models import DerivedResponse
from evalprobe.data.ragtruth import (
    LABEL_REQUIRED_FIELDS,
    RESPONSE_REQUIRED_FIELDS,
    SOURCE_REQUIRED_FIELDS,
    load_dataset,
    matched_spans,
    missing_fields,
    reference_label,
    validate_annotation,
)
from evalprobe.phase0.sampling import hallucination_burden, linear_quantile, sample_pilot
from evalprobe.phase0.sentences import (
    affected_sentence_indices,
    classify_locality,
    label_sentences,
    sentence_spans,
)


def _jsonable_counter(counter: Counter[Any]) -> dict[str, int]:
    return {
        str(key): value for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True) + "\n")


def _distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "p25": linear_quantile(values, 0.25),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p75": linear_quantile(values, 0.75),
        "max": max(values),
    }


def _manual_sample(
    eligible_rows: list[tuple[dict[str, Any], DerivedResponse, list[Any]]], size: int, seed: int
) -> list[dict[str, Any]]:
    unsupported_burdens = sorted(
        feature.hallucination_burden
        for _, feature, _ in eligible_rows
        if feature.reference_label == "UNSUPPORTED"
    )
    low_cut = linear_quantile(unsupported_burdens, 0.25)
    high_cut = linear_quantile(unsupported_burdens, 0.75)

    def categories(row: tuple[dict[str, Any], DerivedResponse, list[Any]]) -> set[str]:
        _, feature, _ = row
        result: set[str] = set()
        if feature.reference_label == "SUPPORTED":
            result.add("supported")
        if feature.span_count == 1:
            result.add("one_span")
        if feature.span_count > 1:
            result.add("multiple_spans")
        if feature.reference_label == "UNSUPPORTED" and feature.hallucination_burden <= low_cut:
            result.add("low_burden")
        if feature.reference_label == "UNSUPPORTED" and feature.hallucination_burden >= high_cut:
            result.add("high_burden")
        if feature.locality == "LOCALIZED":
            result.add("localized")
        if feature.locality == "DISTRIBUTED":
            result.add("distributed")
        if feature.has_implicit_true:
            result.add("implicit_true")
        return result

    rng = random.Random(seed)
    ordered = sorted(eligible_rows, key=lambda row: row[1].response_id)
    rng.shuffle(ordered)
    selected: list[tuple[dict[str, Any], DerivedResponse, list[Any]]] = []
    seen_ids: set[str] = set()
    for target in [
        "supported",
        "one_span",
        "multiple_spans",
        "low_burden",
        "high_burden",
        "localized",
        "distributed",
        "implicit_true",
    ]:
        match = next(
            (
                row
                for row in ordered
                if row[1].response_id not in seen_ids and target in categories(row)
            ),
            None,
        )
        if match:
            selected.append(match)
            seen_ids.add(match[1].response_id)
    for row in ordered:
        if len(selected) >= size:
            break
        if row[1].response_id not in seen_ids:
            selected.append(row)
            seen_ids.add(row[1].response_id)

    artifact: list[dict[str, Any]] = []
    for response, feature, sentences in selected:
        artifact.append(
            {
                **feature.to_dict(),
                "coverage_categories": sorted(categories((response, feature, sentences))),
                "response_text": response["response"],
                "annotations": response["labels"],
                "sentences": [
                    {
                        "start": sentence.start,
                        "end": sentence.end,
                        "reference_label": sentence.label,
                        "text": response["response"][sentence.start : sentence.end],
                    }
                    for sentence in sentences
                ],
                "human_review": {
                    "offsets_sensible": None,
                    "sentence_mapping_sensible": None,
                    "notes": "",
                },
            }
        )
    return artifact


def run_audit(data_dir: Path, output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sources, responses, issues = load_dataset(data_dir)
    source_by_id: dict[str, dict[str, Any]] = {}
    malformed = list(issues)
    duplicate_source_ids: list[str] = []
    duplicate_response_ids: list[str] = []
    seen_response_ids: set[str] = set()

    for index, source in enumerate(sources):
        missing = missing_fields(source, SOURCE_REQUIRED_FIELDS)
        if missing:
            malformed.append(f"source record {index}: missing fields {missing}")
        source_id = source.get("source_id")
        if not isinstance(source_id, str):
            malformed.append(f"source record {index}: source_id is not a string")
            continue
        if source_id in source_by_id:
            duplicate_source_ids.append(source_id)
        source_by_id[source_id] = source

    for index, response in enumerate(responses):
        missing = missing_fields(response, RESPONSE_REQUIRED_FIELDS)
        if missing:
            malformed.append(f"response record {index}: missing fields {missing}")
        response_id = response.get("id")
        if not isinstance(response_id, str):
            malformed.append(f"response record {index}: id is not a string")
        elif response_id in seen_response_ids:
            duplicate_response_ids.append(response_id)
        else:
            seen_response_ids.add(response_id)
        if response.get("source_id") not in source_by_id:
            malformed.append(f"response {response_id}: source_id has no source record")

    qa_sources = [
        source for source in sources if source.get("task_type") == config["dataset"]["task"]
    ]
    qa_source_ids = {source.get("source_id") for source in qa_sources}
    qa_responses = [
        response for response in responses if response.get("source_id") in qa_source_ids
    ]
    known_qualities = set(config["dataset"]["eligible_quality"]) | set(
        config["dataset"]["known_excluded_quality"]
    )
    quality_values = {response.get("quality") for response in qa_responses}
    unexpected_qualities = sorted(str(value) for value in quality_values - known_qualities)
    eligible = [
        response
        for response in qa_responses
        if response.get("quality") in set(config["dataset"]["eligible_quality"])
    ]

    span_counts: Counter[str] = Counter()
    span_failures: list[dict[str, Any]] = []
    annotation_types: Counter[Any] = Counter()
    implicit_true_counts: Counter[Any] = Counter()
    label_shape_counts: Counter[str] = Counter()
    for response in qa_responses:
        labels = response.get("labels")
        if not isinstance(labels, list) or not isinstance(response.get("response"), str):
            malformed.append(f"response {response.get('id')}: labels/response has invalid type")
            continue
        for annotation_index, annotation in enumerate(labels):
            if isinstance(annotation, dict):
                label_shape_counts[str(tuple(sorted(annotation.keys())))] += 1
                missing_label = missing_fields(annotation, LABEL_REQUIRED_FIELDS)
                if missing_label:
                    malformed.append(
                        f"response {response.get('id')} annotation {annotation_index}: "
                        f"missing {missing_label}"
                    )
                annotation_types[annotation.get("label_type")] += 1
                implicit_true_counts[annotation.get("implicit_true", "MISSING")] += 1
            check = validate_annotation(response["response"], annotation)
            span_counts[check.status] += 1
            if check.status != "matched":
                span_failures.append(
                    {
                        "response_id": response.get("id"),
                        "annotation_index": annotation_index,
                        "status": check.status,
                        "start": check.start,
                        "end": check.end,
                    }
                )
    for status in ("matched", "mismatch", "malformed", "out_of_range"):
        span_counts.setdefault(status, 0)

    features: list[DerivedResponse] = []
    eligible_rows: list[tuple[dict[str, Any], DerivedResponse, list[Any]]] = []
    for response in eligible:
        response_text = response.get("response")
        labels = response.get("labels")
        if not isinstance(response_text, str) or not isinstance(labels, list):
            continue
        checks = [validate_annotation(response_text, annotation) for annotation in labels]
        if any(check.status != "matched" for check in checks):
            continue
        spans = matched_spans(response_text, labels)
        sentences = sentence_spans(response_text)
        labeled_sentences = label_sentences(sentences, spans)
        affected = affected_sentence_indices(sentences, spans)
        try:
            locality = classify_locality(sentences, spans)
        except ValueError as error:
            malformed.append(f"response {response.get('id')}: {error}")
            continue
        feature = DerivedResponse(
            response_id=str(response["id"]),
            source_id=str(response["source_id"]),
            split=str(response["split"]),
            reference_label=reference_label(labels),
            hallucination_burden=hallucination_burden(response_text, spans),
            span_count=len(spans),
            sentence_count=len(sentences),
            unsupported_sentence_count=sum(
                sentence.label == "UNSUPPORTED" for sentence in labeled_sentences
            ),
            affected_sentence_count=len(affected),
            locality=locality,
            has_implicit_true=any(
                isinstance(annotation, dict) and annotation.get("implicit_true") is True
                for annotation in labels
            ),
        )
        features.append(feature)
        eligible_rows.append((response, feature, labeled_sentences))

    feature_dicts = [feature.to_dict() for feature in features]
    _write_jsonl(output_dir / "derived_features.jsonl", feature_dicts)
    _write_jsonl(
        output_dir / "manual_audit.jsonl",
        _manual_sample(
            eligible_rows,
            int(config["manual_audit"]["size"]),
            int(config["sampling"]["seed"]),
        ),
    )

    blockers = []
    if malformed or duplicate_source_ids or duplicate_response_ids:
        blockers.append("malformed or duplicate records remain")
    if unexpected_qualities:
        blockers.append("unexpected quality values require a policy decision")
    if sum(value for key, value in span_counts.items() if key != "matched"):
        blockers.append("span validation failures remain")

    eligible_by_split_reference = Counter(
        (feature.split, feature.reference_label) for feature in features
    )
    unsupported_burdens = [
        feature.hallucination_burden
        for feature in features
        if feature.reference_label == "UNSUPPORTED"
    ]
    summary: dict[str, Any] = {
        "dataset": {
            "source_file_sha256": _sha256(data_dir / "source_info.jsonl"),
            "response_file_sha256": _sha256(data_dir / "response.jsonl"),
            "source_records": len(sources),
            "response_records": len(responses),
            "task_types": _jsonable_counter(Counter(source.get("task_type") for source in sources)),
            "split_values": _jsonable_counter(
                Counter(response.get("split") for response in responses)
            ),
            "qa_source_count": len(qa_sources),
            "qa_response_count": len(qa_responses),
            "qa_split_counts": _jsonable_counter(
                Counter(response.get("split") for response in qa_responses)
            ),
            "qa_test_response_count": sum(
                response.get("split") == "test" for response in qa_responses
            ),
            "qa_sources": _jsonable_counter(Counter(source.get("source") for source in qa_sources)),
            "qa_quality_counts": _jsonable_counter(
                Counter(response.get("quality") for response in qa_responses)
            ),
            "unexpected_quality_values": unexpected_qualities,
            "eligible_qa_response_count": len(features),
            "eligible_qa_split_counts": _jsonable_counter(
                Counter(feature.split for feature in features)
            ),
            "eligible_reference_counts": _jsonable_counter(
                Counter(feature.reference_label for feature in features)
            ),
            "eligible_by_split_reference": {
                f"{split}:{label}": count
                for (split, label), count in sorted(eligible_by_split_reference.items())
            },
            "generator_distribution_eligible": _jsonable_counter(
                Counter(response.get("model") for response in eligible)
            ),
            "generator_distribution_qa_raw": _jsonable_counter(
                Counter(response.get("model") for response in qa_responses)
            ),
            "responses_per_source_qa_raw": _jsonable_counter(
                Counter(Counter(response.get("source_id") for response in qa_responses).values())
            ),
            "responses_per_source_qa_eligible": _jsonable_counter(
                Counter(Counter(response.get("source_id") for response in eligible).values())
            ),
            "source_field_shapes": _jsonable_counter(
                Counter(tuple(sorted(source.keys())) for source in sources)
            ),
            "response_field_shapes": _jsonable_counter(
                Counter(tuple(sorted(response.keys())) for response in responses)
            ),
        },
        "annotations": {
            "qa_annotations_checked": sum(span_counts.values()),
            "span_validation": dict(span_counts),
            "span_failures": span_failures,
            "label_types": _jsonable_counter(annotation_types),
            "implicit_true": _jsonable_counter(implicit_true_counts),
            "label_field_shapes": dict(label_shape_counts),
        },
        "derived": {
            "unsupported_hallucination_burden": _distribution(unsupported_burdens),
            "locality": _jsonable_counter(Counter(feature.locality for feature in features)),
            "sentence_counts": _distribution(
                [float(feature.sentence_count) for feature in features]
            ),
            "affected_sentence_counts_unsupported": _distribution(
                [
                    float(feature.affected_sentence_count)
                    for feature in features
                    if feature.reference_label == "UNSUPPORTED"
                ]
            ),
        },
        "integrity": {
            "malformed_or_unexpected_records": malformed,
            "duplicate_source_ids": duplicate_source_ids,
            "duplicate_response_ids": duplicate_response_ids,
            "blockers": blockers,
        },
    }
    _write_json(output_dir / "audit_summary.json", summary)
    render_report(summary, output_dir / "report.md")
    return summary


def build_pilot(output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    summary_path = output_dir / "audit_summary.json"
    feature_path = output_dir / "derived_features.jsonl"
    if not summary_path.is_file() or not feature_path.is_file():
        raise FileNotFoundError("Run `evalprobe phase0 audit` before building the pilot")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    features, feature_issues = _read_generated_jsonl(feature_path)
    if feature_issues:
        raise ValueError(f"Invalid derived feature file: {feature_issues}")
    quotas = {key: int(config["unsupported_strata"][key]) for key in ("low", "medium", "high")}
    selected, thresholds = sample_pilot(
        features,
        seed=int(config["sampling"]["seed"]),
        supported_count=int(config["sampling"]["supported"]),
        unsupported_quotas=quotas,
    )
    manifest_fields = [
        "response_id",
        "source_id",
        "reference_label",
        "hallucination_burden",
        "burden_stratum",
        "locality",
        "span_count",
        "sentence_count",
        "affected_sentence_count",
    ]
    manifest = [{key: item[key] for key in manifest_fields} for item in selected]
    _write_jsonl(output_dir / "pilot_manifest.jsonl", manifest)
    pilot = {
        "seed": int(config["sampling"]["seed"]),
        "threshold_source": "eligible QA train unsupported responses only",
        "quantile_method": "linear interpolation at 1/3 and 2/3",
        "train_tertile_thresholds": {"low_upper": thresholds[0], "medium_upper": thresholds[1]},
        "selected_total": len(manifest),
        "reference_counts": _jsonable_counter(
            Counter(item["reference_label"] for item in manifest)
        ),
        "unsupported_stratum_counts": _jsonable_counter(
            Counter(
                item["burden_stratum"]
                for item in manifest
                if item["reference_label"] == "UNSUPPORTED"
            )
        ),
        "unique_source_ids": len({item["source_id"] for item in manifest}),
        "max_responses_per_source_id": max(
            Counter(item["source_id"] for item in manifest).values(), default=0
        ),
    }
    summary["pilot"] = pilot
    _write_json(summary_path, summary)
    render_report(summary, output_dir / "report.md")
    return pilot


def _read_generated_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    values: list[dict[str, Any]] = []
    issues: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            issues.append(f"line {line_number}: {error.msg}")
            continue
        if isinstance(value, dict):
            values.append(value)
        else:
            issues.append(f"line {line_number}: expected object")
    return values, issues


def _fmt_distribution(distribution: dict[str, Any]) -> str:
    if not distribution.get("count"):
        return "No observations."
    return (
        f"n={distribution['count']}, min={distribution['min']:.6f}, "
        f"p25={distribution['p25']:.6f}, median={distribution['median']:.6f}, "
        f"mean={distribution['mean']:.6f}, p75={distribution['p75']:.6f}, "
        f"max={distribution['max']:.6f}"
    )


def render_report(summary: dict[str, Any], path: Path) -> None:
    dataset = summary["dataset"]
    annotations = summary["annotations"]
    derived = summary["derived"]
    integrity = summary["integrity"]
    pilot = summary.get("pilot")
    failures = sum(
        count for status, count in annotations["span_validation"].items() if status != "matched"
    )
    pilot_ready = bool(
        pilot
        and pilot["selected_total"] == 60
        and pilot["max_responses_per_source_id"] == 1
        and not integrity["blockers"]
    )
    decision = "READY FOR PHASE 1" if pilot_ready else "NOT READY FOR PHASE 1"
    lines = [
        "# Phase 0 — RAGTruth audit and frozen pilot",
        "",
        "## Dataset",
        "",
        (
            f"- Source records: {dataset['source_records']}; "
            f"response records: {dataset['response_records']}."
        ),
        f"- Task types (sources): `{dataset['task_types']}`.",
        f"- Split values (all responses): `{dataset['split_values']}`.",
        (
            f"- QA sources: {dataset['qa_source_count']}; "
            f"QA responses: {dataset['qa_response_count']}; "
            f"raw split counts: `{dataset['qa_split_counts']}`."
        ),
        f"- QA upstream source values: `{dataset['qa_sources']}`.",
        (
            f"- QA quality values: `{dataset['qa_quality_counts']}`. Only `good` is "
            "eligible; `incorrect_refusal` and `truncated` are excluded."
        ),
        (
            f"- Eligible QA responses: {dataset['eligible_qa_response_count']}; "
            f"split counts: `{dataset['eligible_qa_split_counts']}`; "
            f"reference counts: `{dataset['eligible_reference_counts']}`; "
            f"split/reference counts: `{dataset['eligible_by_split_reference']}`."
        ),
        f"- Raw QA generator distribution: `{dataset['generator_distribution_qa_raw']}`.",
        f"- Eligible generator distribution: `{dataset['generator_distribution_eligible']}`.",
        f"- Raw QA responses-per-source distribution: `{dataset['responses_per_source_qa_raw']}`.",
        (
            "- Eligible QA responses-per-source distribution: "
            f"`{dataset['responses_per_source_qa_eligible']}`."
        ),
        "",
        (
            "Whole-response references are constructed only after quality filtering: no "
            "annotations means `SUPPORTED`; one or more annotations means `UNSUPPORTED`. "
            "An annotation with `implicit_true=true` still means unsupported because this "
            "experiment measures grounding in supplied evidence, not truth under outside knowledge."
        ),
        "",
        "## Annotation audit",
        "",
        f"- QA annotations checked: {annotations['qa_annotations_checked']}.",
        f"- Span validation: `{annotations['span_validation']}`; total failures: {failures}.",
        f"- Hallucination label types: `{annotations['label_types']}`.",
        f"- `implicit_true` values: `{annotations['implicit_true']}`.",
        (
            "- Malformed/unexpected record issues: "
            f"{len(integrity['malformed_or_unexpected_records'])}; duplicate source IDs: "
            f"{len(integrity['duplicate_source_ids'])}; duplicate response IDs: "
            f"{len(integrity['duplicate_response_ids'])}."
        ),
        "",
        (
            "Any mismatch, malformed span, or out-of-range span is retained as a visible "
            "failure and blocks readiness; offsets are never repaired."
        ),
        "",
        "## Derived characteristics",
        "",
        (
            "- Unsupported response burden (union span characters / response characters): "
            f"{_fmt_distribution(derived['unsupported_hallucination_burden'])}."
        ),
        f"- Locality: `{derived['locality']}`.",
        f"- Sentence-count distribution: {_fmt_distribution(derived['sentence_counts'])}.",
        (
            "- Affected-sentence distribution among unsupported responses: "
            f"{_fmt_distribution(derived['affected_sentence_counts_unsupported'])}."
        ),
        (
            "- Sentences use the deterministic `evalprobe_rule_v1` punctuation/newline "
            "segmenter with exact source offsets. Any overlapping hallucination span makes "
            "a sentence unsupported; spans touching two or more sentences are distributed."
        ),
        "",
        "## Pilot",
        "",
    ]
    if pilot:
        lines.extend(
            [
                f"- Tertile source: {pilot['threshold_source']}.",
                (
                    "- Train-derived thresholds: "
                    f"low ≤ {pilot['train_tertile_thresholds']['low_upper']:.10f}; "
                    f"medium ≤ {pilot['train_tertile_thresholds']['medium_upper']:.10f}; "
                    "high above that."
                ),
                (
                    f"- Selected: {pilot['selected_total']}; reference counts: "
                    f"`{pilot['reference_counts']}`; unsupported strata: "
                    f"`{pilot['unsupported_stratum_counts']}`."
                ),
                (
                    f"- Unique source IDs: {pilot['unique_source_ids']}; "
                    "maximum responses per source: "
                    f"{pilot['max_responses_per_source_id']}."
                ),
            ]
        )
    else:
        lines.append("Pilot not built. Run `uv run evalprobe phase0 build-pilot` after the audit.")
    lines.extend(
        [
            "",
            "## Risks and limitations",
            "",
            (
                "- RAGTruth predates the future judge, but benchmark exposure or "
                "contamination cannot be ruled out."
            ),
            "- Deterministic sentence boundaries are analysis units, not semantic claims.",
            "- Human exact-span annotations are benchmark references, not infallible truth.",
            (
                "- QA material includes upstream MS MARCO passages. Raw and text-bearing "
                "artifacts remain local and are not redistributed here."
            ),
            "",
            "## Phase 0 decision",
            "",
            f"**{decision}.**",
        ]
    )
    if integrity["blockers"]:
        lines.extend(["", f"Blockers: `{integrity['blockers']}`."])
    elif not pilot:
        lines.extend(
            ["", "The dataset audit passed, but the frozen pilot has not yet been produced."]
        )
    else:
        lines.extend(
            [
                "",
                (
                    "The QA audit has no unresolved integrity failures and the 60-response "
                    "pilot satisfies the frozen balance, train-only threshold, and "
                    "source-uniqueness rules."
                ),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

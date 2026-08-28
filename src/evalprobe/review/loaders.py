from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evalprobe.data.ragtruth import load_dataset
from evalprobe.phase0.sentences import segment_local_units
from evalprobe.review.diagnostics import suspicious_unit_reasons
from evalprobe.review.models import (
    Adjudication,
    HallucinationSpan,
    ReviewIdentity,
    ReviewKind,
    ReviewRecord,
    ReviewTarget,
    SentenceUnit,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required review artifact not found: {path}")
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        values.append(value)
    return values


def _corpus_indexes(
    data_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    sources, responses, issues = load_dataset(data_dir)
    if issues:
        raise ValueError(f"Invalid RAGTruth data: {issues}")
    return (
        {str(source["source_id"]): source for source in sources},
        {str(response["id"]): response for response in responses},
    )


def _question_and_evidence(source: dict[str, Any]) -> tuple[str, str]:
    source_info = source.get("source_info")
    if not isinstance(source_info, dict):
        raise ValueError(f"Source {source.get('source_id')} has invalid source_info")
    question = source_info.get("question")
    evidence = source_info.get("passages")
    if not isinstance(question, str) or not isinstance(evidence, str):
        raise ValueError(f"Source {source.get('source_id')} has invalid QA content")
    return question, evidence


def _spans(annotations: object) -> tuple[HallucinationSpan, ...]:
    if not isinstance(annotations, list):
        raise ValueError("Annotations must be a list")
    spans: list[HallucinationSpan] = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError("Annotation must be an object")
        start = annotation.get("start")
        end = annotation.get("end")
        text = annotation.get("text")
        if type(start) is not int or type(end) is not int or not isinstance(text, str):
            raise ValueError("Annotation span is invalid")
        spans.append(
            HallucinationSpan(
                start=start,
                end=end,
                text=text,
                label_type=str(annotation.get("label_type", "UNKNOWN")),
                implicit_true=(
                    annotation.get("implicit_true")
                    if isinstance(annotation.get("implicit_true"), bool)
                    else None
                ),
            )
        )
    return tuple(spans)


def load_phase0_audit_records(manual_audit_path: Path, data_dir: Path) -> list[ReviewRecord]:
    rows = _read_jsonl(manual_audit_path)
    source_by_id, _ = _corpus_indexes(data_dir)
    records: list[ReviewRecord] = []
    for row in rows:
        record_id = str(row["response_id"])
        source_id = str(row["source_id"])
        source = source_by_id.get(source_id)
        if source is None:
            raise ValueError(f"Missing source {source_id} for audit record {record_id}")
        question, evidence = _question_and_evidence(source)
        answer = row.get("response_text")
        raw_sentences = row.get("sentences")
        if not isinstance(answer, str) or not isinstance(raw_sentences, list):
            raise ValueError(f"Audit record {record_id} has invalid source-bearing fields")
        sentences: list[SentenceUnit] = []
        unsupported_ids: list[int] = []
        for sentence_id, sentence in enumerate(raw_sentences, start=1):
            if not isinstance(sentence, dict):
                raise ValueError(f"Audit record {record_id} has invalid sentence")
            text = sentence.get("text")
            start = sentence.get("start")
            end = sentence.get("end")
            label = sentence.get("reference_label")
            if not isinstance(text, str) or type(start) is not int or type(end) is not int:
                raise ValueError(f"Audit record {record_id} has invalid sentence fields")
            if label not in {"SUPPORTED", "UNSUPPORTED"}:
                raise ValueError(f"Audit record {record_id} has invalid sentence label")
            if label == "UNSUPPORTED":
                unsupported_ids.append(sentence_id)
            sentences.append(
                SentenceUnit(
                    sentence_id,
                    start,
                    end,
                    text,
                    label,
                    suspicious_unit_reasons(text),
                )
            )
        records.append(
            ReviewRecord(
                run_id=str(row.get("run_id", "phase0-sentence-audit-v1")),
                record_id=record_id,
                source_id=source_id,
                split=str(row["split"]),
                view="sentence_audit",
                question=question,
                evidence=evidence,
                answer=answer,
                sentences=tuple(sentences),
                spans=_spans(row.get("annotations")),
                reference_verdict=str(row["reference_label"]),
                reference_unsupported_sentence_ids=tuple(unsupported_ids),
                locality=str(row["locality"]),
                hallucination_burden=float(row["hallucination_burden"]),
            )
        )
    return records


def load_phase1_canary_records(
    manifest_path: Path,
    results_path: Path,
    feature_path: Path,
    data_dir: Path,
    whole_results_path: Path | None = None,
    run_id_override: str | None = None,
) -> list[ReviewRecord]:
    manifest = _read_jsonl(manifest_path)
    results = _read_jsonl(results_path)
    if whole_results_path is not None:
        results = [*_read_jsonl(whole_results_path), *results]
    features = _read_jsonl(feature_path)
    source_by_id, response_by_id = _corpus_indexes(data_dir)
    feature_by_id = {str(feature["response_id"]): feature for feature in features}
    latest_results = {str(result["call_key"]): result for result in results}
    completed_by_record_view = {
        (str(result["record_id"]), str(result["view"])): result
        for result in latest_results.values()
        if result.get("status") == "completed"
    }
    records: list[ReviewRecord] = []
    for reference in manifest:
        record_id = str(reference["record_id"])
        source_id = str(reference["source_id"])
        response = response_by_id.get(record_id)
        source = source_by_id.get(source_id)
        if response is None or source is None:
            raise ValueError(f"Missing corpus data for canary record {record_id}")
        question, evidence = _question_and_evidence(source)
        answer = response.get("response")
        if not isinstance(answer, str):
            raise ValueError(f"Canary record {record_id} has invalid answer")
        reference_ids = tuple(
            sorted(int(value) for value in reference["reference_unsupported_sentence_ids"])
        )
        feature = feature_by_id.get(record_id, {})
        for view in ("whole", "local"):
            result = completed_by_record_view.get((record_id, view))
            if result is None:
                raise ValueError(f"Missing completed {view} result for canary record {record_id}")
            prediction = result.get("semantic_prediction")
            local_units_version = (
                str(result.get("local_units_version", "sentence-v1"))
                if view == "local"
                else "sentence-v1"
            )
            segmentation = segment_local_units(
                answer,
                local_units_version,  # type: ignore[arg-type]
            )
            sentence_units = tuple(
                SentenceUnit(
                    sentence_id=index,
                    start=sentence.start,
                    end=sentence.end,
                    text=answer[sentence.start : sentence.end],
                    reference_label=("UNSUPPORTED" if index in reference_ids else "SUPPORTED"),
                    suspicious_reasons=suspicious_unit_reasons(
                        answer[sentence.start : sentence.end]
                    ),
                )
                for index, sentence in enumerate(segmentation.units, start=1)
            )
            false_positives: tuple[int, ...] = ()
            false_negatives: tuple[int, ...] = ()
            if view == "local":
                if not isinstance(prediction, list) or any(
                    type(value) is not int for value in prediction
                ):
                    raise ValueError(f"Canary record {record_id} has invalid local prediction")
                predicted_ids = tuple(sorted(prediction))
                false_positives = tuple(sorted(set(predicted_ids) - set(reference_ids)))
                false_negatives = tuple(sorted(set(reference_ids) - set(predicted_ids)))
                normalized_prediction: str | tuple[int, ...] = predicted_ids
            else:
                if prediction not in {"SUPPORTED", "UNSUPPORTED"}:
                    raise ValueError(f"Canary record {record_id} has invalid whole prediction")
                normalized_prediction = str(prediction)
            records.append(
                ReviewRecord(
                    run_id=run_id_override or str(result["run_id"]),
                    record_id=record_id,
                    source_id=source_id,
                    split=str(reference["split"]),
                    view=view,
                    question=question,
                    evidence=evidence,
                    answer=answer,
                    sentences=sentence_units,
                    spans=_spans(response.get("labels")),
                    reference_verdict=str(reference["reference_label"]),
                    reference_unsupported_sentence_ids=reference_ids,
                    judge_prediction=normalized_prediction,
                    false_positive_sentence_ids=false_positives,
                    false_negative_sentence_ids=false_negatives,
                    prompt_version=str(result["prompt_version"]),
                    locality=(str(feature["locality"]) if "locality" in feature else None),
                    hallucination_burden=(
                        float(feature["hallucination_burden"])
                        if "hallucination_burden" in feature
                        else None
                    ),
                    burden_stratum=str(reference["burden_stratum"]),
                )
            )
    return records


def phase0_review_targets(records: list[ReviewRecord]) -> list[ReviewTarget]:
    return [
        ReviewTarget(
            ReviewIdentity(record.run_id, record.record_id, record.view),
            record.source_id,
            ReviewKind.SENTENCE_AUDIT,
        )
        for record in records
    ]


def phase1_disagreement_targets(records: list[ReviewRecord]) -> list[ReviewTarget]:
    targets: list[ReviewTarget] = []
    for record in records:
        if record.view == "whole" and record.has_disagreement:
            targets.append(
                ReviewTarget(
                    ReviewIdentity(record.run_id, record.record_id, record.view),
                    record.source_id,
                    ReviewKind.JUDGE_DISAGREEMENT,
                    "WHOLE_VERDICT",
                )
            )
        if record.view == "local":
            for sentence_id in record.false_positive_sentence_ids:
                targets.append(
                    ReviewTarget(
                        ReviewIdentity(record.run_id, record.record_id, record.view, sentence_id),
                        record.source_id,
                        ReviewKind.JUDGE_DISAGREEMENT,
                        "JUDGE_ONLY",
                    )
                )
            for sentence_id in record.false_negative_sentence_ids:
                targets.append(
                    ReviewTarget(
                        ReviewIdentity(record.run_id, record.record_id, record.view, sentence_id),
                        record.source_id,
                        ReviewKind.JUDGE_DISAGREEMENT,
                        "REFERENCE_ONLY",
                    )
                )
    return sorted(
        targets,
        key=lambda target: (
            target.identity.record_id,
            target.identity.view,
            target.identity.sentence_id or 0,
        ),
    )


def filter_review_targets(
    targets: list[ReviewTarget],
    adjudications: dict[str, Adjudication],
    review_status: str,
) -> list[ReviewTarget]:
    if review_status not in {"all", "reviewed", "unreviewed"}:
        raise ValueError("Unknown review-status filter")
    if review_status == "all":
        return targets
    want_reviewed = review_status == "reviewed"
    return [
        target
        for target in targets
        if (target.identity.review_id in adjudications) is want_reviewed
    ]

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from evalprobe.data.models import TextSpan
from evalprobe.data.ragtruth import load_dataset, matched_spans, read_jsonl
from evalprobe.phase0.sampling import burden_stratum, train_tertile_thresholds
from evalprobe.phase0.sentences import affected_sentence_indices, sentence_spans
from evalprobe.phase1.contracts import SafeJudgeSource, Verdict


@dataclass(frozen=True, slots=True)
class CanaryReference:
    record_id: str
    source_id: str
    reference_label: Verdict
    burden_stratum: str
    reference_unsupported_sentence_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SelectedCanaryRecord:
    source: SafeJudgeSource
    reference: CanaryReference


def select_canary_features(features: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    thresholds = train_tertile_thresholds(features)
    training = [feature.copy() for feature in features if feature.get("split") == "train"]
    unsupported = [
        feature for feature in training if feature.get("reference_label") == "UNSUPPORTED"
    ]
    for feature in unsupported:
        feature["burden_stratum"] = burden_stratum(
            float(feature["hallucination_burden"]), thresholds
        )

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    for stratum in ("low", "medium", "high"):
        candidates = sorted(
            (feature for feature in unsupported if feature["burden_stratum"] == stratum),
            key=lambda feature: str(feature["response_id"]),
        )
        rng.shuffle(candidates)
        choice = next(
            (feature for feature in candidates if str(feature["source_id"]) not in used_sources),
            None,
        )
        if choice is None:
            raise ValueError(f"No source-unique TRAIN candidate for {stratum} burden")
        selected.append(choice)
        used_sources.add(str(choice["source_id"]))

    supported = sorted(
        (
            feature
            for feature in training
            if feature.get("reference_label") == "SUPPORTED"
            and str(feature["source_id"]) not in used_sources
        ),
        key=lambda feature: str(feature["response_id"]),
    )
    rng.shuffle(supported)
    for feature in supported:
        source_id = str(feature["source_id"])
        if source_id in used_sources:
            continue
        feature["burden_stratum"] = "none"
        selected.append(feature)
        used_sources.add(source_id)
        if len(selected) == 6:
            break
    if len(selected) != 6 or len(used_sources) != 6:
        raise ValueError("Could not select six source-unique TRAIN canary records")
    return sorted(
        selected,
        key=lambda feature: (
            str(feature["reference_label"]),
            str(feature["burden_stratum"]),
            str(feature["response_id"]),
        ),
    )


def _reference_sentence_ids(answer: str, labels: list[object]) -> tuple[int, ...]:
    spans: list[TextSpan] = matched_spans(answer, labels)
    affected = affected_sentence_indices(sentence_spans(answer), spans)
    return tuple(sorted(index + 1 for index in affected))


def load_canary_records(
    data_dir: Path, feature_path: Path, seed: int
) -> list[SelectedCanaryRecord]:
    features, feature_issues = read_jsonl(feature_path)
    if feature_issues:
        raise ValueError(f"Invalid Phase 0 feature file: {feature_issues}")
    selected_features = select_canary_features(features, seed)
    selected_ids = {str(feature["response_id"]) for feature in selected_features}
    feature_by_id = {str(feature["response_id"]): feature for feature in selected_features}

    sources, responses, issues = load_dataset(data_dir)
    if issues:
        raise ValueError(f"Invalid RAGTruth files: {issues}")
    source_by_id = {str(source["source_id"]): source for source in sources}
    response_by_id = {
        str(response["id"]): response
        for response in responses
        if str(response.get("id")) in selected_ids
    }
    if set(response_by_id) != selected_ids:
        raise ValueError("Selected canary IDs were not all found in RAGTruth responses")

    records: list[SelectedCanaryRecord] = []
    for response_id in sorted(selected_ids):
        feature = feature_by_id[response_id]
        response = response_by_id[response_id]
        if response.get("split") != "train" or response.get("quality") != "good":
            raise ValueError(f"Canary response {response_id} is not eligible TRAIN data")
        source = source_by_id[str(response["source_id"])]
        if source.get("task_type") != "QA":
            raise ValueError(f"Canary response {response_id} is not QA")
        source_info = source.get("source_info")
        if not isinstance(source_info, dict):
            raise ValueError(f"Canary source {response['source_id']} has invalid source_info")
        question = source_info.get("question")
        evidence = source_info.get("passages")
        answer = response.get("response")
        labels = response.get("labels")
        if not all(isinstance(value, str) for value in (question, evidence, answer)):
            raise ValueError(f"Canary response {response_id} has invalid judge-visible fields")
        if not isinstance(labels, list):
            raise ValueError(f"Canary response {response_id} has invalid labels")
        assert isinstance(question, str) and isinstance(evidence, str) and isinstance(answer, str)
        reference_label = cast(Verdict, feature["reference_label"])
        if reference_label not in {"SUPPORTED", "UNSUPPORTED"}:
            raise ValueError(f"Canary response {response_id} has invalid reference label")
        records.append(
            SelectedCanaryRecord(
                source=SafeJudgeSource(question=question, evidence=evidence, answer=answer),
                reference=CanaryReference(
                    record_id=response_id,
                    source_id=str(response["source_id"]),
                    reference_label=reference_label,
                    burden_stratum=str(feature["burden_stratum"]),
                    reference_unsupported_sentence_ids=_reference_sentence_ids(answer, labels),
                ),
            )
        )
    if len({record.reference.source_id for record in records}) != len(records):
        raise ValueError("Canary contains duplicate source_id values")
    return records

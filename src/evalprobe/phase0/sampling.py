from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from evalprobe.data.models import TextSpan


def union_coverage(spans: Iterable[TextSpan]) -> int:
    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    if not ordered:
        return 0
    coverage = 0
    current_start, current_end = ordered[0].start, ordered[0].end
    for span in ordered[1:]:
        if span.start <= current_end:
            current_end = max(current_end, span.end)
        else:
            coverage += current_end - current_start
            current_start, current_end = span.start, span.end
    return coverage + current_end - current_start


def hallucination_burden(response_text: str, spans: Iterable[TextSpan]) -> float:
    if not response_text:
        return 0.0
    return union_coverage(spans) / len(response_text)


def linear_quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a quantile from no values")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def train_tertile_thresholds(features: Iterable[dict[str, Any]]) -> tuple[float, float]:
    burdens = [
        float(feature["hallucination_burden"])
        for feature in features
        if feature["split"] == "train" and feature["reference_label"] == "UNSUPPORTED"
    ]
    return linear_quantile(burdens, 1 / 3), linear_quantile(burdens, 2 / 3)


def burden_stratum(burden: float, thresholds: tuple[float, float]) -> str:
    low, high = thresholds
    if low > high:
        raise ValueError("Tertile thresholds are not ordered")
    if burden <= low:
        return "low"
    if burden <= high:
        return "medium"
    return "high"


def _select_unsupported(
    candidates: list[dict[str, Any]], quotas: dict[str, int], seed: int
) -> list[dict[str, Any]]:
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_stratum[str(candidate["burden_stratum"])].append(candidate)

    for attempt in range(100):
        rng = random.Random(seed + attempt)
        remaining = dict(quotas)
        used_sources: set[str] = set()
        selected: list[dict[str, Any]] = []
        shuffled = {
            key: sorted(value, key=lambda item: str(item["response_id"]))
            for key, value in by_stratum.items()
        }
        for values in shuffled.values():
            rng.shuffle(values)

        while sum(remaining.values()):
            available: dict[str, list[dict[str, Any]]] = {
                stratum: [
                    item
                    for item in shuffled.get(stratum, [])
                    if str(item["source_id"]) not in used_sources
                ]
                for stratum, needed in remaining.items()
                if needed
            }
            if any(not values for values in available.values()):
                break
            stratum = min(
                available,
                key=lambda key: (
                    len({str(item["source_id"]) for item in available[key]}) / remaining[key],
                    key,
                ),
            )
            item = available[stratum][0]
            selected.append(item)
            used_sources.add(str(item["source_id"]))
            remaining[stratum] -= 1
        if not sum(remaining.values()):
            return selected

    unique_counts = {
        key: len({str(item["source_id"]) for item in values}) for key, values in by_stratum.items()
    }
    raise ValueError(
        "Unsupported quotas are infeasible under max one response per source_id; "
        f"requested={quotas}, unique candidates={unique_counts}"
    )


def sample_pilot(
    features: list[dict[str, Any]],
    *,
    seed: int,
    supported_count: int,
    unsupported_quotas: dict[str, int],
) -> tuple[list[dict[str, Any]], tuple[float, float]]:
    thresholds = train_tertile_thresholds(features)
    test_features = [feature.copy() for feature in features if feature["split"] == "test"]
    unsupported: list[dict[str, Any]] = []
    for feature in test_features:
        if feature["reference_label"] == "UNSUPPORTED":
            feature["burden_stratum"] = burden_stratum(
                float(feature["hallucination_burden"]), thresholds
            )
            unsupported.append(feature)

    selected_unsupported = _select_unsupported(unsupported, unsupported_quotas, seed)
    used_sources = {str(item["source_id"]) for item in selected_unsupported}
    supported = [
        feature
        for feature in test_features
        if feature["reference_label"] == "SUPPORTED"
        and str(feature["source_id"]) not in used_sources
    ]
    rng = random.Random(seed)
    supported.sort(key=lambda item: str(item["response_id"]))
    rng.shuffle(supported)
    selected_supported: list[dict[str, Any]] = []
    for feature in supported:
        source_id = str(feature["source_id"])
        if source_id in used_sources:
            continue
        feature["burden_stratum"] = "none"
        selected_supported.append(feature)
        used_sources.add(source_id)
        if len(selected_supported) == supported_count:
            break
    if len(selected_supported) != supported_count:
        raise ValueError(
            "Supported quota is infeasible after enforcing global source uniqueness; "
            f"requested={supported_count}, selected={len(selected_supported)}"
        )

    selected = [*selected_supported, *selected_unsupported]
    selected.sort(
        key=lambda item: (
            str(item["reference_label"]),
            str(item["burden_stratum"]),
            str(item["response_id"]),
        )
    )
    source_counts = Counter(str(item["source_id"]) for item in selected)
    if max(source_counts.values(), default=0) > 1:
        raise AssertionError("Pilot contains repeated source_id values")
    return selected, thresholds

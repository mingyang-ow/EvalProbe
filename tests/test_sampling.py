from collections import Counter

from evalprobe.phase0.sampling import burden_stratum, sample_pilot, train_tertile_thresholds


def _feature(
    response_id: str, source_id: str, split: str, label: str, burden: float
) -> dict[str, object]:
    return {
        "response_id": response_id,
        "source_id": source_id,
        "split": split,
        "reference_label": label,
        "hallucination_burden": burden,
        "locality": "NONE" if label == "SUPPORTED" else "LOCALIZED",
        "span_count": 0 if label == "SUPPORTED" else 1,
        "sentence_count": 1,
        "affected_sentence_count": 0 if label == "SUPPORTED" else 1,
    }


def test_test_values_do_not_affect_train_thresholds() -> None:
    train = [
        _feature("t1", "t1", "train", "UNSUPPORTED", 0.1),
        _feature("t2", "t2", "train", "UNSUPPORTED", 0.2),
        _feature("t3", "t3", "train", "UNSUPPORTED", 0.3),
        _feature("t4", "t4", "train", "UNSUPPORTED", 0.4),
    ]
    first = train_tertile_thresholds(train)
    second = train_tertile_thresholds([*train, _feature("x", "x", "test", "UNSUPPORTED", 999.0)])
    assert first == second == (0.2, 0.3)
    assert burden_stratum(0.2, first) == "low"
    assert burden_stratum(0.25, first) == "medium"
    assert burden_stratum(0.31, first) == "high"


def test_sampling_is_deterministic_balanced_and_source_unique() -> None:
    features = [
        _feature(f"train-{index}", f"train-{index}", "train", "UNSUPPORTED", burden)
        for index, burden in enumerate([0.1, 0.2, 0.3, 0.4])
    ]
    for index, burden in enumerate([0.1, 0.25, 0.4] * 3):
        features.append(_feature(f"u-{index}", f"u-source-{index}", "test", "UNSUPPORTED", burden))
    for index in range(8):
        features.append(_feature(f"s-{index}", f"s-source-{index}", "test", "SUPPORTED", 0.0))

    kwargs = {
        "seed": 20260828,
        "supported_count": 4,
        "unsupported_quotas": {"low": 2, "medium": 2, "high": 2},
    }
    first, thresholds = sample_pilot(features, **kwargs)
    second, _ = sample_pilot(features, **kwargs)
    assert first == second
    assert thresholds == (0.2, 0.3)
    assert len(first) == 10
    assert len({item["source_id"] for item in first}) == 10
    assert Counter(item["reference_label"] for item in first) == {"SUPPORTED": 4, "UNSUPPORTED": 6}
    assert Counter(
        item["burden_stratum"] for item in first if item["reference_label"] == "UNSUPPORTED"
    ) == {"low": 2, "medium": 2, "high": 2}

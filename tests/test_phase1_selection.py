from evalprobe.phase1.selection import select_canary_features


def _feature(index: int, label: str, burden: float) -> dict[str, object]:
    return {
        "response_id": str(index),
        "source_id": f"source-{index}",
        "split": "train",
        "reference_label": label,
        "hallucination_burden": burden,
    }


def test_train_canary_selection_is_deterministic_balanced_and_source_unique() -> None:
    features = [
        _feature(index, "UNSUPPORTED", burden)
        for index, burden in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ]
    features.extend(_feature(20 + index, "SUPPORTED", 0.0) for index in range(8))
    features.append(
        {
            **_feature(99, "UNSUPPORTED", 999.0),
            "split": "test",
        }
    )
    first = select_canary_features(features, 20260829)
    second = select_canary_features(features, 20260829)
    assert first == second
    assert len(first) == 6
    assert len({feature["source_id"] for feature in first}) == 6
    assert sum(feature["reference_label"] == "SUPPORTED" for feature in first) == 3
    assert {
        feature["burden_stratum"]
        for feature in first
        if feature["reference_label"] == "UNSUPPORTED"
    } == {"low", "medium", "high"}
    assert all(feature["split"] == "train" for feature in first)

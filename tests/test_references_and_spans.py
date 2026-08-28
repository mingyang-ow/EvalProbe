import pytest

from evalprobe.data.models import TextSpan
from evalprobe.data.ragtruth import matched_spans, reference_label, validate_annotation
from evalprobe.phase0.sampling import hallucination_burden, union_coverage


def test_union_coverage_does_not_double_count_overlaps() -> None:
    assert union_coverage([TextSpan(2, 7), TextSpan(5, 10), TextSpan(12, 14)]) == 10


def test_burden_uses_union_and_protects_empty_responses() -> None:
    assert hallucination_burden("abcdefghij", [TextSpan(0, 3), TextSpan(2, 5)]) == 0.5
    assert hallucination_burden("", []) == 0.0


def test_invalid_span_detection_is_visible() -> None:
    assert (
        validate_annotation("short", {"start": 0, "end": 8, "text": "shortish"}).status
        == "out_of_range"
    )
    assert validate_annotation("short", {"start": 1, "end": 3, "text": "xx"}).status == "mismatch"
    assert validate_annotation("short", {"start": 3, "end": 3, "text": ""}).status == "malformed"
    with pytest.raises(ValueError, match="mismatch"):
        matched_spans("short", [{"start": 1, "end": 3, "text": "xx"}])


def test_reference_construction_and_implicit_true() -> None:
    assert reference_label([]) == "SUPPORTED"
    labels = [{"start": 0, "end": 4, "text": "true", "implicit_true": True}]
    assert reference_label(labels) == "UNSUPPORTED"

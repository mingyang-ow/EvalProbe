from evalprobe.data.models import TextSpan
from evalprobe.phase0.sentences import (
    affected_sentence_indices,
    classify_locality,
    label_sentences,
    sentence_spans,
    spans_overlap,
)


def test_half_open_overlap_logic() -> None:
    assert spans_overlap(TextSpan(2, 5), TextSpan(4, 8))
    assert not spans_overlap(TextSpan(2, 5), TextSpan(5, 8))


def test_sentence_mapping_preserves_offsets_and_marks_all_overlaps() -> None:
    text = "Grounded first. Unsupported crosses. Final claim."
    sentences = sentence_spans(text)
    span = TextSpan(25, 40)
    labeled = label_sentences(sentences, [span])
    assert [text[item.start : item.end] for item in sentences] == [
        "Grounded first.",
        "Unsupported crosses.",
        "Final claim.",
    ]
    assert [item.label for item in labeled] == ["SUPPORTED", "UNSUPPORTED", "UNSUPPORTED"]
    assert affected_sentence_indices(sentences, [span]) == {1, 2}
    assert classify_locality(sentences, [span]) == "DISTRIBUTED"


def test_locality_values() -> None:
    sentences = sentence_spans("One claim. Two claim.")
    assert classify_locality(sentences, []) == "NONE"
    assert classify_locality(sentences, [TextSpan(0, 3)]) == "LOCALIZED"
    assert classify_locality(sentences, [TextSpan(0, 15)]) == "DISTRIBUTED"

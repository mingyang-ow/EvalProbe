from evalprobe.data.models import TextSpan
from evalprobe.phase0.sentences import (
    is_formatting_list_marker,
    label_sentences,
    segment_local_units,
)


def _texts(text: str, version: str) -> list[str]:
    result = segment_local_units(text, version)  # type: ignore[arg-type]
    return [text[unit.start : unit.end] for unit in result.units]


def test_detects_conservative_formatting_only_markers() -> None:
    for marker in ("1.", "10.", "2)", "•", "-", "*", "(a)", "b."):
        assert is_formatting_list_marker(marker)
    for prose in ("2015.", "In 2015.", "42 answers."):
        assert not is_formatting_list_marker(prose)


def test_merges_multiple_numbered_items_with_exact_offsets_and_mapping() -> None:
    text = "Intro.\n\n1. First item.\n\n2. Second item."
    result = segment_local_units(text, "sentence-v2")

    assert _texts(text, "sentence-v1") == ["Intro.", "1.", "First item.", "2.", "Second item."]
    assert _texts(text, "sentence-v2") == ["Intro.", "1. First item.", "2. Second item."]
    assert result.old_to_new == (1, 2, 2, 3, 3)
    assert result.merged_marker_old_ids == (2, 4)
    assert result.units[1].start == text.index("1.")
    assert result.units[1].end == text.index("\n\n2.")
    assert text[result.units[1].start : result.units[1].end] == "1. First item."


def test_merges_bullet_and_lettered_markers() -> None:
    text = "-\n\nAlpha item.\n\n*\n\nBeta item.\n\n(a)\n\nGamma item."
    assert _texts(text, "sentence-v2") == [
        "-\n\nAlpha item.",
        "*\n\nBeta item.",
        "(a)\n\nGamma item.",
    ]


def test_terminal_marker_is_retained_and_warned() -> None:
    text = "Answer.\n\n1."
    result = segment_local_units(text, "sentence-v2")
    assert _texts(text, "sentence-v2") == ["Answer.", "1."]
    assert result.unmerged_marker_old_ids == (2,)


def test_numeric_prose_is_not_accidentally_merged() -> None:
    text = "The year was 2015. Results followed."
    assert _texts(text, "sentence-v2") == ["The year was 2015.", "Results followed."]


def test_overlap_labels_recompute_after_merge_without_changing_raw_text() -> None:
    text = "1. Claim text."
    original = text[:]
    result = segment_local_units(text, "sentence-v2")
    span = TextSpan(0, 2)
    labelled = label_sentences(list(result.units), [span])

    assert text == original
    assert len(labelled) == 1
    assert labelled[0].label == "UNSUPPORTED"
    assert (labelled[0].start, labelled[0].end) == (0, len(text))
    assert segment_local_units(text, "sentence-v2") == result

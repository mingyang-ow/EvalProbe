from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from evalprobe.data.models import Locality, SentenceSpan, TextSpan

_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*(?=\s|$)|\n{2,}")
_FORMATTING_LIST_MARKER = re.compile(r"(?:\d{1,3}[.)]|[A-Za-z][.)]|\([A-Za-z0-9]{1,3}\)|[-*•])")

LocalUnitsVersion = Literal["sentence-v1", "sentence-v2"]


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    version: LocalUnitsVersion
    units: tuple[SentenceSpan, ...]
    old_to_new: tuple[int, ...]
    merged_marker_old_ids: tuple[int, ...]
    unmerged_marker_old_ids: tuple[int, ...]


def spans_overlap(left: TextSpan | SentenceSpan, right: TextSpan | SentenceSpan) -> bool:
    return max(left.start, right.start) < min(left.end, right.end)


def sentence_spans(text: str) -> list[SentenceSpan]:
    """Segment text with a fixed punctuation/newline rule while retaining source offsets."""
    sentences: list[SentenceSpan] = []
    cursor = 0
    for boundary in _BOUNDARY.finditer(text):
        raw_end = boundary.end()
        start = cursor
        end = raw_end
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            sentences.append(SentenceSpan(start, end))
        cursor = raw_end
    start = cursor
    end = len(text)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        sentences.append(SentenceSpan(start, end))
    return sentences


def is_formatting_list_marker(text: str) -> bool:
    """Return whether a complete unit is only a conservative enumeration marker."""
    return _FORMATTING_LIST_MARKER.fullmatch(text.strip()) is not None


def merge_formatting_list_markers(text: str, raw_units: list[SentenceSpan]) -> SegmentationResult:
    """Merge a marker with the immediately following textual unit, preserving offsets."""
    merged: list[SentenceSpan] = []
    old_to_new: list[int] = []
    merged_marker_ids: list[int] = []
    unmerged_marker_ids: list[int] = []
    index = 0
    while index < len(raw_units):
        current = raw_units[index]
        current_text = text[current.start : current.end]
        old_id = index + 1
        if is_formatting_list_marker(current_text):
            following = raw_units[index + 1] if index + 1 < len(raw_units) else None
            if following is not None:
                following_text = text[following.start : following.end]
                gap = text[current.end : following.start]
                if (
                    gap.isspace()
                    and not is_formatting_list_marker(following_text)
                    and any(character.isalpha() for character in following_text)
                ):
                    new_id = len(merged) + 1
                    merged.append(SentenceSpan(current.start, following.end))
                    old_to_new.extend((new_id, new_id))
                    merged_marker_ids.append(old_id)
                    index += 2
                    continue
            unmerged_marker_ids.append(old_id)
        new_id = len(merged) + 1
        merged.append(current)
        old_to_new.append(new_id)
        index += 1
    return SegmentationResult(
        version="sentence-v2",
        units=tuple(merged),
        old_to_new=tuple(old_to_new),
        merged_marker_old_ids=tuple(merged_marker_ids),
        unmerged_marker_old_ids=tuple(unmerged_marker_ids),
    )


def segment_local_units(
    text: str, version: LocalUnitsVersion = "sentence-v1"
) -> SegmentationResult:
    raw_units = sentence_spans(text)
    if version == "sentence-v1":
        return SegmentationResult(
            version=version,
            units=tuple(raw_units),
            old_to_new=tuple(range(1, len(raw_units) + 1)),
            merged_marker_old_ids=(),
            unmerged_marker_old_ids=tuple(
                index
                for index, unit in enumerate(raw_units, start=1)
                if is_formatting_list_marker(text[unit.start : unit.end])
            ),
        )
    if version != "sentence-v2":
        raise ValueError(f"Unknown local-units version: {version}")
    return merge_formatting_list_markers(text, raw_units)


def label_sentences(sentences: list[SentenceSpan], spans: list[TextSpan]) -> list[SentenceSpan]:
    return [
        SentenceSpan(
            sentence.start,
            sentence.end,
            "UNSUPPORTED" if any(spans_overlap(sentence, span) for span in spans) else "SUPPORTED",
        )
        for sentence in sentences
    ]


def affected_sentence_indices(sentences: list[SentenceSpan], spans: list[TextSpan]) -> set[int]:
    return {
        index
        for index, sentence in enumerate(sentences)
        if any(spans_overlap(sentence, span) for span in spans)
    }


def classify_locality(sentences: list[SentenceSpan], spans: list[TextSpan]) -> Locality:
    if not spans:
        return "NONE"
    affected = affected_sentence_indices(sentences, spans)
    if not affected:
        raise ValueError("A hallucination span did not overlap any sentence")
    return "LOCALIZED" if len(affected) == 1 else "DISTRIBUTED"

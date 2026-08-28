from __future__ import annotations

import re

from evalprobe.data.models import Locality, SentenceSpan, TextSpan

_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*(?=\s|$)|\n{2,}")


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

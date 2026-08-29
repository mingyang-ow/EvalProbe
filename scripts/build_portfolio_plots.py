#!/usr/bin/env python3
"""Build the two final, dependency-free portfolio SVGs from safe aggregate reports."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = ROOT / "reports/phase2/frozen-test-v1/analysis.json"
ERROR_ANALYSIS_PATH = (
    ROOT / "reports/phase3/frozen-test-error-analysis/error_analysis.json"
)
OUTPUT_DIR = ROOT / "reports/phase4"

INK = "#172033"
MUTED = "#5f6b7a"
PAPER = "#ffffff"
GRID = "#dce2ea"
SUPPORTED = "#dcefe8"
UNSUPPORTED = "#f8d9d4"
AMBIGUITY = "#3d7f85"
JUDGE_ERROR = "#d2675a"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _text(x: float, y: float, value: str, **attrs: object) -> str:
    attributes = " ".join(
        f'{name.rstrip("_").replace("_", "-")}="{item}"' for name, item in attrs.items()
    )
    return f'<text x="{x}" y="{y}" {attributes}>{html.escape(value)}</text>'


def _svg_document(width: int, height: int, title: str, body: list[str]) -> str:
    escaped_title = html.escape(title)
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" role="img" '
                f'aria-labelledby="title desc" viewBox="0 0 {width} {height}">'
            ),
            f"  <title id=\"title\">{escaped_title}</title>",
            (
                "  <desc id=\"desc\">Generated from safe aggregate EvalProbe reports; "
                "labels show raw counts.</desc>"
            ),
            f'  <rect width="{width}" height="{height}" fill="{PAPER}"/>',
            (
                "  <style>text { font-family: Inter, ui-sans-serif, system-ui, sans-serif; "
                f"fill: {INK}; }} .muted {{ fill: {MUTED}; }} "
                f".on-color {{ fill: {PAPER}; }}</style>"
            ),
            *[f"  {line}" for line in body],
            "</svg>",
            "",
        ]
    )


def build_confusion_matrix(analysis: dict[str, Any]) -> str:
    matrix = analysis["whole_response"]["confusion_matrix"]
    values = [
        matrix["reference_supported_predicted_supported"],
        matrix["reference_supported_predicted_unsupported"],
        matrix["reference_unsupported_predicted_supported"],
        matrix["reference_unsupported_predicted_unsupported"],
    ]
    if values != [19, 11, 1, 29]:
        raise ValueError(f"Frozen whole-response confusion matrix changed: {values}")

    body = [
        _text(60, 50, "Whole-response decisions", font_size=25, font_weight=700),
        _text(60, 78, "Frozen TEST · n = 60 · raw counts", font_size=15, class_="muted"),
        _text(470, 119, "Judge prediction", font_size=15, font_weight=650, text_anchor="middle"),
        _text(390, 151, "Supported", font_size=14, text_anchor="middle"),
        _text(550, 151, "Unsupported", font_size=14, text_anchor="middle"),
        _text(60, 249, "RAGTruth reference", font_size=15, font_weight=650),
        _text(185, 205, "Supported", font_size=14, text_anchor="end"),
        _text(185, 325, "Unsupported", font_size=14, text_anchor="end"),
    ]
    cells = [
        (310, 165, SUPPORTED, "19", "agreement"),
        (470, 165, UNSUPPORTED, "11", "false positive"),
        (310, 285, UNSUPPORTED, "1", "false negative"),
        (470, 285, SUPPORTED, "29", "agreement"),
    ]
    for x, y, color, count, label in cells:
        body.append(
            f'<rect x="{x}" y="{y}" width="160" height="120" fill="{color}" '
            f'stroke="{PAPER}" stroke-width="6"/>'
        )
        body.append(
            _text(x + 80, y + 57, count, font_size=34, font_weight=750, text_anchor="middle")
        )
        body.append(
            _text(x + 80, y + 83, label, font_size=13, text_anchor="middle", class_="muted")
        )
    body.extend(
        [
            _text(60, 448, "Accuracy 48/60 (80.0%)", font_size=16, font_weight=650),
            _text(60, 476, "Unsupported recall 29/30 (96.7%)", font_size=15),
            _text(60, 504, "Supported recall 19/30 (63.3%)", font_size=15),
        ]
    )
    return _svg_document(720, 550, "EvalProbe whole-response confusion matrix", body)


def build_human_classifications(error_analysis: dict[str, Any]) -> str:
    groups = error_analysis["groups"]
    whole = groups["WHOLE_DISAGREEMENTS"]["mismatch_populations"]
    populations = [
        ("Whole false positives (all)", whole["FALSE_POSITIVE"]),
        ("Whole false negative (all)", whole["FALSE_NEGATIVE"]),
        ("Local reference-only (all)", groups["LOCAL_REFERENCE_ONLY"]),
        ("Local judge-only (sampled 20/71)", groups["LOCAL_JUDGE_ONLY_SAMPLE"]),
    ]
    expected = [(1, 10), (1, 0), (8, 4), (0, 20)]
    actual = [
        (
            group["classification_counts"]["JUDGE_ERROR"],
            group["classification_counts"]["BENCHMARK_AMBIGUITY"],
        )
        for _, group in populations
    ]
    if actual != expected:
        raise ValueError(f"Final human classification counts changed: {actual}")

    left = 285
    scale = 23
    body = [
        _text(50, 48, "What human review found", font_size=25, font_weight=700),
        _text(
            50,
            76,
            "Bounded disagreement populations · raw counts",
            font_size=15,
            class_="muted",
        ),
    ]
    for tick in range(0, 21, 5):
        x = left + tick * scale
        body.append(f'<line x1="{x}" y1="112" x2="{x}" y2="400" stroke="{GRID}"/>')
        body.append(_text(x, 105, str(tick), font_size=12, text_anchor="middle", class_="muted"))

    for index, ((label, _), (judge_count, ambiguity_count)) in enumerate(
        zip(populations, actual, strict=True)
    ):
        y = 139 + index * 68
        body.append(_text(left - 15, y + 23, label, font_size=14, text_anchor="end"))
        if judge_count:
            width = judge_count * scale
            body.append(
                f'<rect x="{left}" y="{y}" width="{width}" height="34" '
                f'fill="{JUDGE_ERROR}"/>'
            )
            body.append(
                _text(
                    left + width / 2,
                    y + 23,
                    str(judge_count),
                    font_size=14,
                    font_weight=700,
                    text_anchor="middle",
                )
            )
        if ambiguity_count:
            x = left + judge_count * scale
            width = ambiguity_count * scale
            body.append(f'<rect x="{x}" y="{y}" width="{width}" height="34" fill="{AMBIGUITY}"/>')
            body.append(
                _text(
                    x + width / 2,
                    y + 23,
                    str(ambiguity_count),
                    font_size=14,
                    font_weight=700,
                    class_="on-color",
                    text_anchor="middle",
                )
            )
    body.extend(
        [
            f'<rect x="285" y="438" width="16" height="16" fill="{JUDGE_ERROR}"/>',
            _text(310, 451, "Judge error", font_size=14),
            f'<rect x="425" y="438" width="16" height="16" fill="{AMBIGUITY}"/>',
            _text(450, 451, "Benchmark ambiguity", font_size=14),
            _text(285, 492, "No final reviewed item was a methodology defect.", font_size=14),
            _text(
                285,
                518,
                "Human review explains official disagreement; it does not relabel metrics.",
                font_size=13,
                class_="muted",
            ),
        ]
    )
    return _svg_document(820, 560, "EvalProbe human disagreement classifications", body)


def main() -> None:
    analysis = _load_json(ANALYSIS_PATH)
    error_analysis = _load_json(ERROR_ANALYSIS_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "whole_confusion_matrix.svg").write_text(
        build_confusion_matrix(analysis), encoding="utf-8"
    )
    (OUTPUT_DIR / "human_disagreement_classifications.svg").write_text(
        build_human_classifications(error_analysis), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

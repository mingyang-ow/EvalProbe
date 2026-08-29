from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^]]*\]\(([^)]+)\)")


def test_local_markdown_links_resolve() -> None:
    documents = [ROOT / "README.md", ROOT / "story.md", *sorted((ROOT / "docs").rglob("*.md"))]
    broken: list[str] = []
    for document in documents:
        for raw_target in MARKDOWN_LINK.findall(document.read_text(encoding="utf-8")):
            target = raw_target.strip().strip("<>").split("#", maxsplit=1)[0]
            if not target or target.startswith(("https://", "http://", "mailto:")):
                continue
            if not (document.parent / target).resolve().exists():
                broken.append(f"{document.relative_to(ROOT)} -> {raw_target}")
    assert not broken, "Broken local Markdown links:\n" + "\n".join(broken)


def test_readme_mermaid_pipeline_is_complete() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.count("```mermaid") == 1
    assert readme.count("```") % 2 == 0
    assert "flowchart TD" in readme
    assert "Human adjudication" in readme


def test_final_portfolio_svgs_are_accessible_and_label_raw_counts() -> None:
    expected = {
        "whole_confusion_matrix.svg": ["19", "11", "1", "29", "raw counts"],
        "human_disagreement_classifications.svg": [
            "sampled 20/71",
            "Judge error",
            "Benchmark ambiguity",
            "raw counts",
        ],
    }
    for filename, labels in expected.items():
        svg = (ROOT / "reports/phase4" / filename).read_text(encoding="utf-8")
        assert svg.startswith('<?xml version="1.0"')
        assert 'role="img"' in svg
        assert "<title" in svg and "<desc" in svg
        for label in labels:
            assert label in svg

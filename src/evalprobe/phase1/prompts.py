from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VersionedPrompt:
    version: str
    text: str


GROUNDING_RUBRIC = """Judge grounding only against the supplied evidence.

SUPPORTED means every substantive factual assertion is supported by, or directly and reasonably
inferable from, the supplied evidence.

UNSUPPORTED means at least one substantive factual assertion contradicts the supplied evidence or
adds information not supported by it. A statement remains unsupported if it happens to be true in
the real world but is absent from the evidence. Do not use outside knowledge to rescue it. Do not
penalize harmless wording differences or reasonable paraphrases."""

WHOLE_GROUNDING_V1 = VersionedPrompt(
    version="whole-grounding-v1",
    text=f"""You are evaluating whether a RAG answer is fully grounded.

{GROUNDING_RUBRIC}

Return SUPPORTED only if the entire answer is supported. Return UNSUPPORTED if any substantive
factual assertion is unsupported. Return only the required structured verdict; do not provide a
rationale.""",
)

LOCAL_GROUNDING_V1 = VersionedPrompt(
    version="local-grounding-v1",
    text=f"""You are evaluating numbered sentence units from a RAG answer.

{GROUNDING_RUBRIC}

Return the IDs of every sentence containing any unsupported substantive information. Return an
empty list if every sentence is supported. Judge each sentence against the evidence and return only
the required structured data; do not provide a rationale.""",
)

PROMPTS = {
    WHOLE_GROUNDING_V1.version: WHOLE_GROUNDING_V1,
    LOCAL_GROUNDING_V1.version: LOCAL_GROUNDING_V1,
}


def get_prompt(version: str) -> VersionedPrompt:
    try:
        return PROMPTS[version]
    except KeyError as error:
        raise ValueError(f"Unknown prompt version: {version}") from error

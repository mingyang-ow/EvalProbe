from dataclasses import fields

import pytest

from evalprobe.phase1.contracts import (
    ContractValidationError,
    LocalJudgeInput,
    NumberedSentence,
    SafeJudgeSource,
    StructuredOutputError,
    WholeJudgeInput,
    assert_no_leakage,
    parse_local_output,
    parse_whole_output,
    render_model_input,
)


def test_judge_dtos_cannot_contain_reference_fields() -> None:
    source = SafeJudgeSource("question", "evidence", "answer")
    whole = WholeJudgeInput.from_source(source)
    local = LocalJudgeInput("question", "evidence", (NumberedSentence(1, "answer"),))
    assert {field.name for field in fields(whole)} == {"question", "evidence", "answer"}
    assert {field.name for field in fields(local)} == {
        "question",
        "evidence",
        "numbered_sentences",
    }
    assert "reference_label" not in render_model_input(whole)
    assert "reference_label" not in render_model_input(local)
    with pytest.raises(ValueError, match="Reference-only"):
        assert_no_leakage({"question": "q", "reference_label": "SUPPORTED"})


def test_whole_output_parses_strictly() -> None:
    assert parse_whole_output('{"verdict":"SUPPORTED"}') == "SUPPORTED"
    with pytest.raises(ContractValidationError):
        parse_whole_output('{"verdict":"SUPPORTED","reason":"extra"}')
    with pytest.raises(ContractValidationError):
        parse_whole_output('{"verdict":"MAYBE"}')
    with pytest.raises(StructuredOutputError):
        parse_whole_output("not json")


def test_local_output_rejects_duplicates_and_unknown_ids() -> None:
    valid = frozenset({1, 2, 3})
    assert parse_local_output('{"unsupported_sentence_ids":[3,1]}', valid) == (1, 3)
    with pytest.raises(ContractValidationError, match="unique"):
        parse_local_output('{"unsupported_sentence_ids":[1,1]}', valid)
    with pytest.raises(ContractValidationError, match="Unknown"):
        parse_local_output('{"unsupported_sentence_ids":[4]}', valid)
    with pytest.raises(ContractValidationError, match="integer"):
        parse_local_output('{"unsupported_sentence_ids":[true]}', valid)


def test_local_schema_leaves_uniqueness_to_application_validation() -> None:
    from evalprobe.phase1.contracts import LOCAL_OUTPUT_SCHEMA

    array_schema = LOCAL_OUTPUT_SCHEMA["properties"]["unsupported_sentence_ids"]
    assert "uniqueItems" not in array_schema

from collections.abc import Iterator
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from evalprobe.phase3.review_set import (
    LOCAL_JUDGE_ONLY_SAMPLE,
    LOCAL_REFERENCE_ONLY,
    WHOLE_DISAGREEMENTS,
    Phase3ReviewItem,
)
from evalprobe.review.models import ReviewIdentity, ReviewRecord, SentenceUnit

APP_PATH = Path(__file__).parents[1] / "src/evalprobe/review/app.py"


@pytest.fixture(autouse=True)
def _clear_streamlit_data_cache() -> Iterator[None]:
    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.fixture
def isolated_review_console(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "evalprobe.review.loaders.load_phase0_audit_records",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "evalprobe.review.loaders.load_phase1_canary_records",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "evalprobe.review.storage.adjudications_by_id",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "evalprobe.review.diagnostics.aggregate_suspicious_units",
        lambda *_args, **_kwargs: {
            "overall": {"suspicious_unit_count": 0, "sentence_unit_count": 0}
        },
    )


def _frozen_disagreement_records() -> list[ReviewRecord]:
    common = {
        "run_id": "synthetic-frozen",
        "record_id": "synthetic-record",
        "source_id": "synthetic-source",
        "split": "test",
        "question": "Synthetic question?",
        "evidence": "Synthetic evidence.",
        "answer": "One. Two.",
        "sentences": (
            SentenceUnit(1, 0, 4, "One.", "SUPPORTED"),
            SentenceUnit(2, 5, 9, "Two.", "UNSUPPORTED"),
        ),
        "spans": (),
        "reference_verdict": "UNSUPPORTED",
        "reference_unsupported_sentence_ids": (2,),
        "local_units_version": "sentence-v2",
        "locality": "LOCALIZED",
        "hallucination_burden": 0.4,
        "burden_stratum": "low",
    }
    return [
        ReviewRecord(
            **common,
            view="whole",
            judge_prediction="SUPPORTED",
            prompt_version="synthetic-whole-v1",
        ),
        ReviewRecord(
            **common,
            view="local",
            judge_prediction=(1,),
            false_positive_sentence_ids=(1,),
            false_negative_sentence_ids=(2,),
            prompt_version="synthetic-local-v1",
        ),
    ]


def _phase3_item(
    *, view: str, sentence_id: int | None, mismatch_type: str, review_group: str
) -> Phase3ReviewItem:
    identity = ReviewIdentity("synthetic-frozen", "synthetic-record", view, sentence_id)
    return Phase3ReviewItem(
        review_id=identity.review_id,
        run_id=identity.run_id,
        record_id=identity.record_id,
        source_id="synthetic-source",
        view=view,
        sentence_id=sentence_id,
        mismatch_type=mismatch_type,
        review_group=review_group,
        official_reference_label="UNSUPPORTED",
        whole_judge_prediction="SUPPORTED",
        whole_relation="DISAGREEMENT",
        burden_stratum="low",
        locality="LOCALIZED",
        local_judge_only_count=1,
        sampling_seed=20260828 if review_group == LOCAL_JUDGE_ONLY_SAMPLE else None,
        sample_rank=1 if review_group == LOCAL_JUDGE_ONLY_SAMPLE else None,
        diagnostic_priority=None,
    )


def _frozen_review_items() -> list[Phase3ReviewItem]:
    return [
        _phase3_item(
            view="whole",
            sentence_id=None,
            mismatch_type="FALSE_NEGATIVE",
            review_group=WHOLE_DISAGREEMENTS,
        ),
        _phase3_item(
            view="local",
            sentence_id=2,
            mismatch_type="REFERENCE_ONLY",
            review_group=LOCAL_REFERENCE_ONLY,
        ),
        _phase3_item(
            view="local",
            sentence_id=1,
            mismatch_type="JUDGE_ONLY",
            review_group=LOCAL_JUDGE_ONLY_SAMPLE,
        ),
    ]


def test_review_console_loads_both_segmentation_versions(
    isolated_review_console: None,
) -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    assert not app.exception
    assert any(control.label == "Judge run" for control in app.segmented_control)

    methodology = next(
        control for control in app.segmented_control if control.label == "Local-unit methodology"
    )
    methodology.set_value("sentence-v1")
    app.run(timeout=30)
    assert not app.exception
    assert methodology.value == "sentence-v1"

    methodology = next(
        control for control in app.segmented_control if control.label == "Local-unit methodology"
    )
    methodology.set_value("sentence-v2")
    app.run(timeout=30)
    assert not app.exception
    assert methodology.value == "sentence-v2"

    review_mode = next(
        control for control in app.segmented_control if control.label == "Review mode"
    )
    review_mode.set_value("Judge disagreement review")
    app.run(timeout=30)
    assert not app.exception
    assert any("No current v2" in message.value for message in app.info)


def test_review_console_renders_classification_for_current_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "evalprobe.review.loaders.load_phase0_audit_records",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "evalprobe.review.loaders.load_phase1_canary_records",
        lambda *_args, **_kwargs: _frozen_disagreement_records(),
    )
    monkeypatch.setattr(
        "evalprobe.phase3.review_set.load_phase3_review_items",
        lambda *_args, **_kwargs: _frozen_review_items(),
    )
    monkeypatch.setattr(
        "evalprobe.review.storage.adjudications_by_id",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "evalprobe.review.diagnostics.aggregate_suspicious_units",
        lambda *_args, **_kwargs: {
            "overall": {"suspicious_unit_count": 0, "sentence_unit_count": 1}
        },
    )

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    review_mode = next(
        control for control in app.segmented_control if control.label == "Review mode"
    )
    review_mode.set_value("Judge disagreement review")
    judge_run = next(control for control in app.segmented_control if control.label == "Judge run")
    judge_run.set_value("Frozen TEST")
    app.run(timeout=30)

    assert not app.exception
    assert any(
        control.label == "Phase 3 review group" for control in app.segmented_control
    )
    for group in (WHOLE_DISAGREEMENTS, LOCAL_REFERENCE_ONLY, LOCAL_JUDGE_ONLY_SAMPLE):
        phase3_group = next(
            control for control in app.segmented_control if control.label == "Phase 3 review group"
        )
        phase3_group.set_value(group)
        app.run(timeout=30)
        assert not app.exception
        classification = next(
            selectbox for selectbox in app.selectbox if selectbox.label == "Primary classification"
        )
        assert classification.value is None
        assert "JUDGE_ERROR" in classification.options

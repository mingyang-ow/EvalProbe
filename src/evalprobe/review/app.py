from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from evalprobe.review.diagnostics import aggregate_suspicious_units
from evalprobe.review.loaders import (
    filter_review_targets,
    load_phase0_audit_records,
    load_phase1_canary_records,
    phase0_review_targets,
    phase1_disagreement_targets,
)
from evalprobe.review.models import (
    CLASSIFICATION_DEFINITIONS,
    Adjudication,
    HumanClassification,
    ReviewKind,
    ReviewRecord,
    ReviewTarget,
    SentenceAuditFailureType,
    SentenceAuditStatus,
)
from evalprobe.review.storage import adjudications_by_id, save_adjudication

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPOSITORY_ROOT / "data/raw"
PHASE0_AUDIT = REPOSITORY_ROOT / "reports/phase0/manual_audit.jsonl"
PHASE1_DIR = REPOSITORY_ROOT / "reports/phase1/train-canary-v1"
PHASE0_FEATURES = REPOSITORY_ROOT / "reports/phase0/derived_features.jsonl"
ADJUDICATIONS = REPOSITORY_ROOT / "reports/review/adjudications.jsonl"

MODE_SENTENCE = "Sentence audit"
MODE_DISAGREEMENT = "Judge disagreement review"
MODE_INSPECTOR = "Record inspector"

st.set_page_config(
    page_title="EvalProbe review console",
    page_icon=":material/fact_check:",
    layout="wide",
)


@st.cache_data(show_spinner=False, max_entries=2)
def _load_phase0() -> list[ReviewRecord]:
    return load_phase0_audit_records(PHASE0_AUDIT, DATA_DIR)


@st.cache_data(show_spinner=False, max_entries=2)
def _load_phase1() -> list[ReviewRecord]:
    return load_phase1_canary_records(
        PHASE1_DIR / "manifest.jsonl",
        PHASE1_DIR / "results.jsonl",
        PHASE0_FEATURES,
        DATA_DIR,
    )


@st.cache_data(show_spinner=False, max_entries=1)
def _load_diagnostics() -> dict[str, object]:
    return aggregate_suspicious_units(DATA_DIR)


def _move_target(widget_key: str, target_ids: list[str], delta: int) -> None:
    current = st.session_state.get(widget_key)
    index = target_ids.index(current) if current in target_ids else 0
    st.session_state[widget_key] = target_ids[(index + delta) % len(target_ids)]


def _target_label(target: ReviewTarget) -> str:
    sentence = f" · sentence {target.identity.sentence_id}" if target.identity.sentence_id else ""
    mismatch = f" · {target.mismatch_type}" if target.mismatch_type else ""
    return f"{target.identity.record_id} · {target.identity.view}{sentence}{mismatch}"


def _plain_text_panel(title: str, text: str, height: int) -> None:
    with st.container(border=True, height=height):
        st.subheader(title)
        st.text(text)


def _sentence_category(record: ReviewRecord, sentence_id: int) -> str:
    reference = sentence_id in record.reference_unsupported_sentence_ids
    judge = isinstance(record.judge_prediction, tuple) and sentence_id in record.judge_prediction
    if reference and judge:
        return "BOTH"
    if reference:
        return "REFERENCE ONLY"
    if judge:
        return "JUDGE ONLY"
    return "NEITHER"


def _render_spans(record: ReviewRecord) -> None:
    with st.expander(
        f"RAGTruth hallucination spans ({len(record.spans)})",
        icon=":material/format_quote:",
    ):
        if not record.spans:
            st.caption("No hallucination spans on this response.")
        for index, span in enumerate(record.spans, start=1):
            with st.container(border=True):
                st.caption(
                    f"Span {index} · offsets {span.start}:{span.end} · "
                    f"{span.label_type} · implicit_true={span.implicit_true}"
                )
                st.text(span.text)


def _render_sentences(record: ReviewRecord, current_sentence_id: int | None = None) -> None:
    st.subheader("Numbered sentence units")
    for sentence in record.sentences:
        category = (
            _sentence_category(record, sentence.sentence_id)
            if record.view == "local"
            else sentence.reference_label
        )
        color = {
            "BOTH": "green",
            "REFERENCE ONLY": "orange",
            "JUDGE ONLY": "violet",
            "NEITHER": "gray",
            "UNSUPPORTED": "orange",
            "SUPPORTED": "gray",
        }[category]
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center", gap="xsmall"):
                st.badge(f"Sentence {sentence.sentence_id}", color="blue")
                st.badge(category, color=color)
                if sentence.sentence_id == current_sentence_id:
                    st.badge(
                        "Current review item", icon=":material/arrow_right_alt:", color="primary"
                    )
                if sentence.suspicious_reasons:
                    st.badge(
                        "Possible formatting-only unit",
                        icon=":material/warning:",
                        color="yellow",
                    )
            st.text(sentence.text)
            if sentence.suspicious_reasons:
                st.caption("Deterministic flags: " + ", ".join(sentence.suspicious_reasons))


def _render_record(record: ReviewRecord, current_sentence_id: int | None = None) -> None:
    with st.container(horizontal=True, gap="xsmall"):
        st.badge(f"Record {record.record_id}", color="blue")
        st.badge(f"Source {record.source_id}", color="gray")
        st.badge(record.split.upper(), color="gray")
        st.badge(record.view.upper(), color="primary")
        if record.prompt_version:
            st.badge(record.prompt_version, color="violet")
    if record.record_id == "12839":
        st.warning(
            "Record 12839 is a required segmentation diagnostic. Inspect its standalone list "
            "markers; this warning does not classify them.",
            icon=":material/warning:",
        )
    if record.view == "whole":
        with st.container(horizontal=True, gap="small"):
            st.badge(f"Reference: {record.reference_verdict}", color="orange")
            st.badge(f"Judge: {record.judge_prediction}", color="violet")
    elif record.view == "local":
        st.caption(
            f"Reference unsupported: {list(record.reference_unsupported_sentence_ids)} · "
            f"Judge unsupported: {list(record.judge_prediction or ())} · "
            f"False positives: {list(record.false_positive_sentence_ids)} · "
            f"False negatives: {list(record.false_negative_sentence_ids)}"
        )
    else:
        st.caption(
            f"Reference: {record.reference_verdict} · unsupported sentence IDs: "
            f"{list(record.reference_unsupported_sentence_ids)} · locality: {record.locality} · "
            f"hallucination burden: {record.hallucination_burden:.6f}"
        )
    st.subheader("Question")
    st.text(record.question)
    evidence_column, answer_column = st.columns([1.2, 1], gap="medium")
    with evidence_column:
        _plain_text_panel("Supplied evidence", record.evidence, 300)
    with answer_column:
        _plain_text_panel("Complete answer", record.answer, 300)
    _render_sentences(record, current_sentence_id)
    _render_spans(record)


def _render_progress(
    targets: list[ReviewTarget], decisions: dict[str, Adjudication], kind: ReviewKind
) -> None:
    reviewed = sum(target.identity.review_id in decisions for target in targets)
    with st.container(horizontal=True, gap="small"):
        st.metric("Reviewed", f"{reviewed} / {len(targets)}", border=True)
        st.metric("Remaining", len(targets) - reviewed, border=True)
    st.progress(reviewed / len(targets) if targets else 0.0)
    relevant = [decision for decision in decisions.values() if decision.review_kind == kind]
    if kind == ReviewKind.SENTENCE_AUDIT:
        counts = Counter(decision.sentence_audit_status for decision in relevant)
        labels = [f"{status.value}: {counts[status.value]}" for status in SentenceAuditStatus]
    else:
        counts = Counter(decision.classification for decision in relevant)
        labels = [
            f"{classification.value}: {counts[classification.value]}"
            for classification in HumanClassification
        ]
    st.caption(" · ".join(labels))


def _render_target_navigation(targets: list[ReviewTarget], key: str) -> ReviewTarget:
    target_ids = [target.identity.review_id for target in targets]
    target_by_id = {target.identity.review_id: target for target in targets}
    if st.session_state.get(key) not in target_ids:
        st.session_state[key] = target_ids[0]
    selected = st.selectbox(
        "Review item",
        target_ids,
        format_func=lambda value: _target_label(target_by_id[value]),
        key=key,
    )
    with st.container(horizontal=True, gap="small"):
        st.button(
            "Previous",
            icon=":material/arrow_back:",
            on_click=_move_target,
            args=(key, target_ids, -1),
        )
        st.button(
            "Next",
            icon=":material/arrow_forward:",
            on_click=_move_target,
            args=(key, target_ids, 1),
        )
    return target_by_id[selected]


def _save_sentence_audit_form(target: ReviewTarget, existing: Adjudication | None) -> None:
    default_status = (
        SentenceAuditStatus(existing.sentence_audit_status)
        if existing and existing.sentence_audit_status
        else SentenceAuditStatus.PASS
    )
    default_failure = existing.failure_type if existing else None
    with st.form(f"sentence-audit-{target.identity.review_id}"):
        st.subheader("Human sentence-audit decision")
        audit_status = st.segmented_control(
            "Status",
            list(SentenceAuditStatus),
            default=default_status,
            required=True,
            format_func=lambda value: value.value,
            width="stretch",
        )
        failure_options: list[SentenceAuditFailureType | None] = [
            None,
            *SentenceAuditFailureType,
        ]
        failure_type = st.selectbox(
            "Optional failure classification",
            failure_options,
            index=failure_options.index(
                SentenceAuditFailureType(default_failure) if default_failure else None
            ),
            format_func=lambda value: "None" if value is None else value.value,
        )
        note = st.text_area(
            "Concise reviewer note",
            value=existing.note if existing else "",
            max_chars=500,
            help="Use your own words; do not copy large evidence passages.",
        )
        submitted = st.form_submit_button("Save review", icon=":material/save:", type="primary")
    if submitted:
        try:
            decision = Adjudication.create(
                identity=target.identity,
                source_id=target.source_id,
                review_kind=target.kind,
                sentence_audit_status=audit_status,
                failure_type=failure_type,
                note=note,
                reviewed_at=datetime.now(UTC).isoformat(),
            )
            save_adjudication(ADJUDICATIONS, decision)
        except ValueError as error:
            st.error(str(error), icon=":material/error:")
        else:
            st.success("Review saved. Saving this item again will update it, not duplicate it.")


def _save_disagreement_form(target: ReviewTarget, existing: Adjudication | None) -> None:
    st.subheader("Human adjudication")
    with st.expander("Classification definitions", icon=":material/menu_book:"):
        for classification, definition in CLASSIFICATION_DEFINITIONS.items():
            st.markdown(f"**{classification.value}** — {definition}")
    options: list[HumanClassification | None] = [None, *HumanClassification]
    default = (
        HumanClassification(existing.classification)
        if existing and existing.classification
        else None
    )
    with st.form(f"disagreement-{target.identity.review_id}"):
        classification = st.selectbox(
            "Primary classification",
            options,
            index=options.index(default),
            format_func=lambda value: "Select one" if value is None else value.value,
        )
        note = st.text_area(
            "Concise reviewer note",
            value=existing.note if existing else "",
            max_chars=500,
            help="Use your own words; do not copy large evidence passages.",
        )
        submitted = st.form_submit_button("Save review", icon=":material/save:", type="primary")
    if submitted:
        try:
            decision = Adjudication.create(
                identity=target.identity,
                source_id=target.source_id,
                review_kind=target.kind,
                classification=classification,
                note=note,
                reviewed_at=datetime.now(UTC).isoformat(),
            )
            save_adjudication(ADJUDICATIONS, decision)
        except ValueError as error:
            st.error(str(error), icon=":material/error:")
        else:
            st.success("Review saved. Saving this item again will update it, not duplicate it.")


st.title("EvalProbe human review console")
st.caption(
    "Local inspection only. This app never executes judges, changes benchmark labels, or makes "
    "methodological decisions automatically."
)

with st.sidebar:
    mode = st.segmented_control(
        "Review mode",
        [MODE_SENTENCE, MODE_DISAGREEMENT, MODE_INSPECTOR],
        default=MODE_SENTENCE,
        required=True,
        key="review_mode",
        width="stretch",
    )
    review_status = st.segmented_control(
        "Review status",
        ["all", "unreviewed", "reviewed"],
        default="all",
        required=True,
        key="review_status",
        disabled=mode == MODE_INSPECTOR,
        width="stretch",
    )
    show_only_disagreements = st.toggle(
        "Show only disagreements",
        value=False,
        disabled=mode != MODE_INSPECTOR,
    )
    st.caption("No API key is accepted. Experiment execution is a separate CLI workflow.")

try:
    decisions = adjudications_by_id(ADJUDICATIONS)
    if mode == MODE_SENTENCE:
        phase0_records = _load_phase0()
        all_targets = phase0_review_targets(phase0_records)
        record_by_key = {(record.record_id, record.view): record for record in phase0_records}
        _render_progress(all_targets, decisions, ReviewKind.SENTENCE_AUDIT)
        targets = filter_review_targets(all_targets, decisions, str(review_status))
        if not targets:
            st.info("No review items match the current status filter.")
            st.stop()
        target = _render_target_navigation(targets, "phase0_target")
        record = record_by_key[(target.identity.record_id, target.identity.view)]
        _render_record(record)
        _save_sentence_audit_form(target, decisions.get(target.identity.review_id))
    else:
        phase1_records = _load_phase1()
        disagreement_targets = phase1_disagreement_targets(phase1_records)
        record_by_key = {(record.record_id, record.view): record for record in phase1_records}
        if mode == MODE_DISAGREEMENT:
            _render_progress(
                disagreement_targets,
                decisions,
                ReviewKind.JUDGE_DISAGREEMENT,
            )
            targets = filter_review_targets(
                disagreement_targets,
                decisions,
                str(review_status),
            )
            if not targets:
                st.info("No review items match the current status filter.")
                st.stop()
            target = _render_target_navigation(targets, "phase1_target")
            record = record_by_key[(target.identity.record_id, target.identity.view)]
            _render_record(record, target.identity.sentence_id)
            _save_disagreement_form(target, decisions.get(target.identity.review_id))
        else:
            records = [
                record
                for record in phase1_records
                if not show_only_disagreements or record.has_disagreement
            ]
            run_id = st.selectbox("Run ID", sorted({record.run_id for record in records}))
            view = st.selectbox("View", ["all", "whole", "local"])
            source_id = st.selectbox(
                "Source ID",
                ["all", *sorted({record.source_id for record in records})],
            )
            filtered_records = [
                record
                for record in records
                if record.run_id == run_id
                and (view == "all" or record.view == view)
                and (source_id == "all" or record.source_id == source_id)
            ]
            if not filtered_records:
                st.info("No records match the selected identifiers.")
                st.stop()
            record_ids = sorted({record.record_id for record in filtered_records})
            record_id = st.selectbox("Record ID", record_ids)
            candidate_views = [
                record for record in filtered_records if record.record_id == record_id
            ]
            selected_record = st.selectbox(
                "Available record view",
                candidate_views,
                format_func=lambda record: f"{record.record_id} · {record.view}",
            )
            _render_record(selected_record)
    with st.sidebar:
        diagnostics = _load_diagnostics()
        overall = diagnostics["overall"]
        st.caption(
            "Suspicious-unit diagnostic: "
            f"{overall['suspicious_unit_count']} / {overall['sentence_unit_count']} "
            "eligible QA sentence units flagged across TRAIN and TEST."
        )
except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
    st.error(str(error), icon=":material/error:")
    st.stop()

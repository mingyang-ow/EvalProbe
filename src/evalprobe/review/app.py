from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from evalprobe.phase3.review_set import (
    LOCAL_JUDGE_ONLY_SAMPLE,
    LOCAL_REFERENCE_ONLY,
    PHASE3_GROUPS,
    WHOLE_DISAGREEMENTS,
    Phase3ReviewItem,
    load_phase3_review_items,
    phase3_targets_by_group,
)
from evalprobe.review.diagnostics import aggregate_suspicious_units
from evalprobe.review.loaders import (
    filter_review_targets,
    load_phase0_audit_records,
    load_phase1_canary_records,
    phase0_review_targets,
    phase1_current_local_disagreement_targets,
    phase1_disagreement_targets,
    phase1_segmentation_repair_outcomes,
)
from evalprobe.review.models import (
    CLASSIFICATION_DEFINITIONS,
    Adjudication,
    HumanClassification,
    MethodologyRepairOutcome,
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
PHASE0_AUDIT_V2 = REPOSITORY_ROOT / "reports/phase0/manual_audit_v2.jsonl"
PHASE1_DIR = REPOSITORY_ROOT / "reports/phase1/train-canary-v1"
PHASE1C_DIR = REPOSITORY_ROOT / "reports/phase1c/train-canary-segmentation-v2"
PHASE2_DIR = REPOSITORY_ROOT / "reports/phase2/frozen-test-v1"
PHASE3_DIR = REPOSITORY_ROOT / "reports/phase3/frozen-test-error-analysis"
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


@st.cache_data(show_spinner=False, max_entries=4)
def _load_phase0(version: str) -> list[ReviewRecord]:
    path = PHASE0_AUDIT if version == "sentence-v1" else PHASE0_AUDIT_V2
    return load_phase0_audit_records(path, DATA_DIR)


@st.cache_data(show_spinner=False, max_entries=6)
def _load_phase1(version: str, judge_run: str = "TRAIN validation") -> list[ReviewRecord]:
    if judge_run == "Frozen TEST":
        if version != "sentence-v2":
            raise ValueError("Frozen TEST is available only with sentence-v2")
        return load_phase1_canary_records(
            PHASE2_DIR / "manifest.jsonl",
            PHASE2_DIR / "results.jsonl",
            PHASE0_FEATURES,
            DATA_DIR,
            run_id_override="frozen-test-v1",
            local_units_version="sentence-v2",
        )
    if version == "sentence-v2":
        return load_phase1_canary_records(
            PHASE1C_DIR / "manifest.jsonl",
            PHASE1C_DIR / "results.jsonl",
            PHASE0_FEATURES,
            DATA_DIR,
            whole_results_path=PHASE1_DIR / "results.jsonl",
            run_id_override="train-canary-segmentation-v2",
            local_units_version="sentence-v2",
            allow_missing_local_result=True,
        )
    return load_phase1_canary_records(
        PHASE1_DIR / "manifest.jsonl",
        PHASE1_DIR / "results.jsonl",
        PHASE0_FEATURES,
        DATA_DIR,
    )


@st.cache_data(show_spinner=False, max_entries=1)
def _load_phase3() -> list[Phase3ReviewItem]:
    return load_phase3_review_items(PHASE3_DIR / "review_set.jsonl")


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
    if record.judge_prediction is None:
        return "UNSUPPORTED" if reference else "SUPPORTED"
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
        st.badge(record.local_units_version, color="violet")
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
        burden = (
            "n/a" if record.hallucination_burden is None else f"{record.hallucination_burden:.6f}"
        )
        st.caption(
            f"Burden stratum: {record.burden_stratum} · locality: {record.locality} · "
            f"hallucination burden: {burden}"
        )
        st.info(
            "Review question: Does every substantive assertion in the complete answer remain "
            "grounded in the supplied evidence?",
            icon=":material/help:",
        )
    elif record.view == "local":
        if record.judge_prediction is None:
            st.info(
                "The Phase 1C local judge result is unavailable. Showing deterministic "
                "sentence-v2 units and official reference labels only.",
                icon=":material/info:",
            )
            st.caption(
                f"Reference unsupported: {list(record.reference_unsupported_sentence_ids)} · "
                "Judge unsupported: unavailable"
            )
        else:
            st.caption(
                f"Reference unsupported: {list(record.reference_unsupported_sentence_ids)} · "
                f"Judge unsupported: {list(record.judge_prediction)} · "
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


def _render_version_comparison(before: ReviewRecord, after: ReviewRecord) -> None:
    if before.local_units_version != "sentence-v1":
        raise ValueError("Comparison before-record is not sentence-v1")
    if after.local_units_version != "sentence-v2":
        raise ValueError("Comparison after-record is not sentence-v2")
    with st.expander("Sentence-v1 / sentence-v2 comparison", expanded=False):
        before_column, after_column = st.columns(2, gap="medium")
        for column, title, record in (
            (before_column, before.local_units_version, before),
            (after_column, after.local_units_version, after),
        ):
            with column:
                st.subheader(title)
                st.caption(
                    f"{len(record.sentences)} units · reference unsupported "
                    f"{list(record.reference_unsupported_sentence_ids)}"
                )
                for sentence in record.sentences:
                    with st.container(border=True):
                        st.caption(
                            f"Sentence {sentence.sentence_id} · {sentence.reference_label} · "
                            f"offsets {sentence.start}:{sentence.end}"
                        )
                        st.text(sentence.text)


def _render_historical_context(target: ReviewTarget, decisions: dict[str, Adjudication]) -> None:
    historical = [
        decision
        for decision in decisions.values()
        if decision.record_id == target.identity.record_id
        and decision.run_id != target.identity.run_id
    ]
    if not historical:
        return
    with st.expander("Historical v1 adjudication", expanded=True):
        for decision in sorted(historical, key=lambda item: (item.view, item.sentence_id or 0)):
            outcome = (
                decision.classification
                or decision.sentence_audit_status
                or decision.failure_type
                or "REVIEWED"
            )
            st.caption(
                f"{decision.run_id} · {decision.view} · sentence "
                f"{decision.sentence_id or 'record'} · {outcome}"
            )
            if decision.note:
                st.text(decision.note)


def _render_methodology_repair_outcomes(
    outcomes: list[MethodologyRepairOutcome],
) -> None:
    with st.container(border=True):
        st.subheader("Historical v1 methodology outcomes")
        st.caption(
            "These are derived statuses, not new human classifications. Historical v1 "
            "adjudications remain unchanged."
        )
        for outcome in outcomes:
            color = {
                "RESOLVED_BY_METHODOLOGY_REPAIR": "green",
                "CURRENT_V2_DISAGREEMENT": "orange",
                "PENDING_V2_JUDGE_RESULT": "yellow",
                "HISTORICAL_V1_ONLY": "gray",
            }[outcome.status]
            with st.container(border=True):
                with st.container(horizontal=True, gap="xsmall"):
                    st.badge(f"Record {outcome.record_id}", color="blue")
                    st.badge(outcome.view.upper(), color="gray")
                    st.badge(outcome.status, color=color)
                if outcome.old_sentence_id is not None:
                    st.caption(
                        f"v1 sentence {outcome.old_sentence_id} → "
                        f"v2 sentence {outcome.new_sentence_id} · "
                        f"current category: {outcome.current_category}"
                    )
                else:
                    st.caption(f"Current category: {outcome.current_category}")


def _render_progress(
    targets: list[ReviewTarget], decisions: dict[str, Adjudication], kind: ReviewKind
) -> None:
    reviewed = sum(target.identity.review_id in decisions for target in targets)
    with st.container(horizontal=True, gap="small"):
        st.metric("Reviewed", f"{reviewed} / {len(targets)}", border=True)
        st.metric("Remaining", len(targets) - reviewed, border=True)
    st.progress(reviewed / len(targets) if targets else 0.0)
    target_review_ids = {target.identity.review_id for target in targets}
    relevant = [
        decision
        for decision in decisions.values()
        if decision.review_kind == kind and decision.review_id in target_review_ids
    ]
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


def _render_phase3_progress(
    targets_by_group: dict[str, list[ReviewTarget]],
    decisions: dict[str, Adjudication],
) -> None:
    labels = {
        WHOLE_DISAGREEMENTS: "Whole disagreements",
        LOCAL_REFERENCE_ONLY: "Local reference only",
        LOCAL_JUDGE_ONLY_SAMPLE: "Local judge-only sample",
    }
    with st.container(horizontal=True, gap="small"):
        for group in PHASE3_GROUPS:
            targets = targets_by_group[group]
            reviewed = sum(target.identity.review_id in decisions for target in targets)
            st.metric(labels[group], f"{reviewed} / {len(targets)}", border=True)


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
    segmentation_version = st.segmented_control(
        "Local-unit methodology",
        ["sentence-v1", "sentence-v2"],
        default="sentence-v2",
        required=True,
        key="segmentation_version",
        width="stretch",
    )
    judge_run = st.segmented_control(
        "Judge run",
        ["TRAIN validation", "Frozen TEST"],
        default="TRAIN validation",
        required=True,
        key="judge_run",
        disabled=mode == MODE_SENTENCE,
        width="stretch",
    )
    phase3_group = st.segmented_control(
        "Phase 3 review group",
        list(PHASE3_GROUPS),
        default=WHOLE_DISAGREEMENTS,
        required=True,
        key="phase3_group",
        disabled=mode != MODE_DISAGREEMENT or judge_run != "Frozen TEST",
        width="stretch",
    )
    st.caption("No API key is accepted. Experiment execution is a separate CLI workflow.")

try:
    decisions = adjudications_by_id(ADJUDICATIONS)
    if mode == MODE_SENTENCE:
        phase0_records = _load_phase0(str(segmentation_version))
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
        if segmentation_version == "sentence-v2":
            previous = {item.record_id: item for item in _load_phase0("sentence-v1")}[
                record.record_id
            ]
            _render_version_comparison(previous, record)
            _render_historical_context(target, decisions)
        _save_sentence_audit_form(target, decisions.get(target.identity.review_id))
    else:
        phase1_records = _load_phase1(str(segmentation_version), str(judge_run))
        methodology_outcomes: list[MethodologyRepairOutcome] = []
        phase3_items_by_review_id: dict[str, Phase3ReviewItem] = {}
        phase3_groups: dict[str, list[ReviewTarget]] | None = None
        if judge_run == "TRAIN validation" and segmentation_version == "sentence-v2":
            methodology_outcomes = phase1_segmentation_repair_outcomes(
                phase1_records, list(decisions.values())
            )
            disagreement_targets = phase1_current_local_disagreement_targets(phase1_records)
        elif judge_run == "Frozen TEST" and mode == MODE_DISAGREEMENT:
            phase3_items = _load_phase3()
            phase3_items_by_review_id = {item.review_id: item for item in phase3_items}
            phase3_groups = phase3_targets_by_group(phase3_items)
            disagreement_targets = phase3_groups[str(phase3_group)]
        else:
            disagreement_targets = phase1_disagreement_targets(phase1_records)
        record_by_key = {(record.record_id, record.view): record for record in phase1_records}
        if mode == MODE_DISAGREEMENT:
            if methodology_outcomes:
                _render_methodology_repair_outcomes(methodology_outcomes)
            if phase3_groups is not None:
                _render_phase3_progress(phase3_groups, decisions)
                group_help = {
                    WHOLE_DISAGREEMENTS: (
                        "Review all official whole-response false positives and the single "
                        "false negative."
                    ),
                    LOCAL_REFERENCE_ONLY: (
                        "Review all official unsupported local units missed by the local judge."
                    ),
                    LOCAL_JUDGE_ONLY_SAMPLE: (
                        "Review the fixed metadata-only sample; the other judge-only units are "
                        "outside this bounded workload."
                    ),
                }
                st.caption(group_help[str(phase3_group)])
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
                st.info(
                    "No current v2 REFERENCE ONLY or JUDGE ONLY items match the queue. "
                    "Resolved historical targets require no new classification."
                )
                st.stop()
            target = _render_target_navigation(targets, "phase1_target")
            phase3_item = phase3_items_by_review_id.get(target.identity.review_id)
            if phase3_item and phase3_item.diagnostic_priority:
                st.warning(
                    "This is the only whole-response UNSUPPORTED → SUPPORTED miss. Inspect the "
                    "burden, span location, locality, and annotation defensibility without "
                    "generalizing from one record.",
                    icon=":material/priority_high:",
                )
            record = record_by_key[(target.identity.record_id, target.identity.view)]
            _render_record(record, target.identity.sentence_id)
            if judge_run == "TRAIN validation" and segmentation_version == "sentence-v2":
                previous = {
                    (item.record_id, item.view): item for item in _load_phase1("sentence-v1")
                }[(record.record_id, record.view)]
                _render_version_comparison(previous, record)
                _render_historical_context(target, decisions)
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

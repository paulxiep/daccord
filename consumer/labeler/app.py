"""Tier-7C Streamlit labeler — bidirectional-aware reviewer for tiered pairs.

Reads `data/ensemble/tiered/{pair}.jsonl` (tier-6B output) + optional
`data/ensemble/bidirectional/{pair}.jsonl` (tier-6B+ cross-check) and
lets a human reviewer pick the correct target citation for each unreviewed
source clause. Verdicts append to `data/ensemble/validated/{pair}.jsonl`
(write-once per source_id; resume-by-source_id across sessions).

Run:
    docker compose run --rm --service-ports consumer \\
        uv run streamlit run labeler/app.py --server.address 0.0.0.0
Then visit http://localhost:8501.

Configuration via env vars (read at app startup):
  - DACCORD_TIERED_DIR        — tier-6B output (default: ../data/ensemble/tiered)
  - DACCORD_VALIDATED_DIR     — overlay outputs (default: ../data/ensemble/validated)
  - DACCORD_BIDIRECTIONAL_DIR — bidirectional overlay
                                (default: ../data/ensemble/bidirectional)
  - DACCORD_REVIEWER          — reviewer name embedded in each row
                                (default: $USER or "anonymous")

The labeler defaults skip bidirectionally-`consistent` rows — those are
auto-promotable to gold via `scripts/build_splits.py --promote-bidirectional-consistent`
and don't need human time. The reviewer can toggle them in the sidebar
if they want a sanity check.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import streamlit as st
from labeler.queue import (
    PairFile,
    QueuedPair,
    SortOrder,
    build_review_queue,
    list_pair_files,
    progress_summary,
    vote_choices,
)

from daccord.ensemble.bidirectional import BidirectionalStatus
from daccord.ensemble.schema import Tier, TieredPair
from daccord.ensemble.validated import ValidatedPair, append_validation

DEFAULT_TIERED_DIR = Path("../data/ensemble/tiered")
DEFAULT_VALIDATED_DIR = Path("../data/ensemble/validated")
DEFAULT_BIDIRECTIONAL_DIR = Path("../data/ensemble/bidirectional")

ALL_TIERS: tuple[Tier, ...] = ("HIGH", "MED", "LOW", "SALVAGE")
ALL_BIDIRECTIONAL_STATUSES: tuple[BidirectionalStatus, ...] = (
    "consistent",
    "inconsistent",
    "reverse_unknown",
    "missing_reverse_row",
    "missing_reverse_pair",
    "missing_in_registry",
    "no_forward_consensus",
)
DEFAULT_TIER_FILTER: tuple[Tier, ...] = ("MED", "LOW", "SALVAGE")
DEFAULT_BI_FILTER: tuple[BidirectionalStatus, ...] = (
    "inconsistent",
    "reverse_unknown",
    "missing_reverse_row",
    "missing_reverse_pair",
    "missing_in_registry",
    "no_forward_consensus",
)


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw) if raw else default


def _reviewer_name() -> str:
    return os.environ.get("DACCORD_REVIEWER") or os.environ.get("USER") or "anonymous"


def _render_pair_picker(pair_files: list[PairFile]) -> PairFile | None:
    options = [
        f"{f.name}  ({f.reviewed}/{f.total} reviewed, "
        f"bi: {f.bidirectional_consistent} consistent / "
        f"{f.bidirectional_inconsistent} inconsistent)"
        for f in pair_files
    ]
    if not options:
        st.warning("No tiered files found. Did you run scripts/tier_ensemble.py?")
        return None
    choice = st.selectbox("Framework pair", options, key="pair_choice")
    return pair_files[options.index(choice)]


_TIER_COLORS: dict[str, str] = {
    "HIGH": "green",
    "MED": "blue",
    "LOW": "orange",
    "SALVAGE": "gray",
}

_BIDIRECTIONAL_COLORS: dict[str, tuple[str, str]] = {
    "consistent": ("green", "✓ consistent"),
    "inconsistent": ("red", "✗ inconsistent"),
    "reverse_unknown": ("gray", "? reverse_unknown"),
    "missing_reverse_row": ("gray", "? missing_reverse_row"),
    "missing_reverse_pair": ("gray", "? missing_reverse_pair"),
    "missing_in_registry": ("orange", "⚠ missing_in_registry"),
    "no_forward_consensus": ("gray", "— no_forward_consensus"),
}


def _render_model_agreement_panel(p: TieredPair) -> None:
    """Left panel: forward (within-ensemble) model agreement signal."""
    tier_color = _TIER_COLORS.get(p.tier, "gray")
    st.markdown("#### Model agreement (forward)")
    st.markdown(f"**Tier:** :{tier_color}[**{p.tier}**] · score = **{p.agreement_score:.2f}**")
    st.markdown(f"**Consensus citation_id:** `{p.consensus_citation_id or '(none)'}`")
    st.markdown(
        f"**Valid votes:** {p.valid_vote_count} / {len(p.votes)} seats · "
        f"agreed on consensus: {p.consensus_vote_count}"
    )
    # One-line summary per seat for at-a-glance read. The tier-6B++ RAG
    # seat (model slug starts with "local-rag") gets a distinct icon so
    # reviewers can spot the retrieval-derived vote among the LLM votes.
    summary_rows = []
    for v in p.votes:
        is_rag = v.model.startswith("local-rag")
        icon = "🔎" if is_rag else "🤖"
        label = "retrieval" if is_rag else v.model
        row = f"- {icon} `{label}` → `{v.citation_id_raw or '(empty)'}`"
        if v.parse_error:
            row += f"  ⚠ parse_error: {v.parse_error}"
        summary_rows.append(row)
    st.markdown("\n".join(summary_rows))


def _render_bidirectional_panel(qp: QueuedPair) -> None:
    """Right panel: bidirectional (cross-direction) agreement signal."""
    st.markdown("#### Bidirectional agreement (reverse)")
    bi = qp.bidirectional
    if bi is None:
        st.markdown(
            ":gray[**No bidirectional data**] — "
            "`scripts/cross_check_ensemble.py` hasn't been run for this pair."
        )
        return
    color, label = _BIDIRECTIONAL_COLORS.get(bi.status, ("gray", bi.status))
    st.markdown(f"**Status:** :{color}[**{label}**]")
    st.markdown(f"**Reverse pair:** `{bi.reverse_pair}`")
    if bi.reverse_source_id is not None:
        st.markdown(f"**Reverse source_id:** `{bi.reverse_source_id}`")
    if bi.reverse_tier is not None:
        tier_color = _TIER_COLORS.get(bi.reverse_tier, "gray")
        score = (
            f"{bi.reverse_agreement_score:.2f}" if bi.reverse_agreement_score is not None else "—"
        )
        st.markdown(f"**Reverse tier:** :{tier_color}[**{bi.reverse_tier}**] · score = **{score}**")
    if bi.reverse_consensus is not None:
        st.markdown(f"**Reverse consensus citation_id:** `{bi.reverse_consensus or '(none)'}`")
    if bi.status == "consistent":
        st.success(
            "Reverse direction confirmed this mapping. Auto-promotable via "
            "`build_splits.py --promote-bidirectional-consistent`."
        )
    elif bi.status == "inconsistent":
        st.warning(
            "Reverse direction disagreed. Worth careful hand-val even if forward tier is HIGH."
        )


def _render_pair(qp: QueuedPair) -> None:
    p = qp.pair
    st.markdown(f"### {p.source_framework} / `{p.source_citation_id}` → {p.target_framework}")
    st.caption(f"source_id=`{p.source_id}`")

    # Two signal panels side by side — model agreement on the left,
    # bidirectional on the right. Both are first-class signals; the
    # reviewer compares them before reading the source clause.
    left, right = st.columns(2)
    with left:
        _render_model_agreement_panel(p)
    with right:
        _render_bidirectional_panel(qp)

    st.divider()
    st.markdown("**Source clause:**")
    st.text_area("source", p.source_mechanism, height=180, label_visibility="collapsed")

    st.markdown("**Votes from ensemble seats (full text):**")
    for v in p.votes:
        with st.expander(
            f"{v.model} — `{v.citation_id_raw or '(empty)'}`"
            + (f"  [parse_error: {v.parse_error}]" if v.parse_error else "")
        ):
            if v.target_mechanism:
                st.markdown("**Target mechanism:**")
                st.text(v.target_mechanism)
            if v.mapping_justification:
                st.markdown("**Justification:**")
                st.text(v.mapping_justification)


def _submit_verdict(
    qp: QueuedPair,
    chosen: str,
    note: str,
    validated_path: Path,
    reviewer: str,
) -> None:
    p = qp.pair
    row = ValidatedPair(
        source_id=p.source_id,
        source_jurisdiction=p.source_jurisdiction,
        source_framework=p.source_framework,
        target_jurisdiction=p.target_jurisdiction,
        target_framework=p.target_framework,
        chosen_citation_id=chosen,
        human_note=note,
        reviewer=reviewer,
        reviewed_at=datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        tier_at_review=p.tier,
        agreement_score_at_review=p.agreement_score,
    )
    append_validation(validated_path, row)


def main() -> None:
    st.set_page_config(page_title="d'accord — tier-7C labeler", layout="wide")
    st.title("d'accord — tier-7C reviewer")

    tiered_dir = _env_path("DACCORD_TIERED_DIR", DEFAULT_TIERED_DIR)
    validated_dir = _env_path("DACCORD_VALIDATED_DIR", DEFAULT_VALIDATED_DIR)
    bidirectional_dir = _env_path("DACCORD_BIDIRECTIONAL_DIR", DEFAULT_BIDIRECTIONAL_DIR)
    reviewer = _reviewer_name()

    pair_files = list_pair_files(tiered_dir, validated_dir, bidirectional_dir)

    with st.sidebar:
        st.subheader("Configuration")
        st.caption(f"reviewer = **{reviewer}**")
        st.caption(f"tiered_dir = `{tiered_dir}`")
        st.caption(f"validated_dir = `{validated_dir}`")
        st.caption(f"bidirectional_dir = `{bidirectional_dir}`")
        st.divider()

        progress = progress_summary(pair_files)
        st.metric(
            "Overall progress",
            f"{progress['reviewed']} / {progress['total']}",
            help=f"{progress['remaining']} rows remaining across all pairs.",
        )
        st.divider()

        st.subheader("Filters")
        tier_filter = cast(
            list[Tier],
            st.multiselect(
                "Tier filter",
                options=list(ALL_TIERS),
                default=list(DEFAULT_TIER_FILTER),
                help="HIGH usually goes through stratified spot-check, not full review.",
            ),
        )
        bidirectional_filter = cast(
            list[BidirectionalStatus],
            st.multiselect(
                "Bidirectional filter",
                options=list(ALL_BIDIRECTIONAL_STATUSES),
                default=list(DEFAULT_BI_FILTER),
                help=(
                    "Default excludes 'consistent' (those are auto-promotable via "
                    "scripts/build_splits.py --promote-bidirectional-consistent). "
                    "Surface 'inconsistent' first for highest-leverage hand-val."
                ),
            ),
        )
        sort_order: SortOrder = cast(
            SortOrder,
            st.radio(
                "Sort",
                options=["confidence-desc", "confidence-asc", "source-id"],
                index=0,
                help="confidence-desc surfaces easiest MEDs first (high-throughput).",
            ),
        )

    selected_pair_file = _render_pair_picker(pair_files)
    if selected_pair_file is None:
        return

    queue = build_review_queue(
        tiered_path=selected_pair_file.tiered_path,
        validated_path=selected_pair_file.validated_path,
        bidirectional_dir=bidirectional_dir,
        tier_filter=tier_filter,
        bidirectional_filter=bidirectional_filter or None,
        sort_order=sort_order,
    )

    st.markdown(
        f"**Queue for `{selected_pair_file.name}`:** {len(queue)} pairs match "
        f"filters (out of {selected_pair_file.total - selected_pair_file.reviewed} "
        f"unreviewed).  Reviewed in this pair: {selected_pair_file.reviewed}."
    )

    if not queue:
        st.success(
            "Queue empty — either every match has been reviewed, or your "
            "filters exclude everything. Adjust the sidebar and try again."
        )
        return

    qp = queue[0]
    _render_pair(qp)

    st.divider()
    st.markdown("### Your verdict")

    choices = vote_choices(qp.pair)
    radio_label_to_value = {label: value for value, label in choices}
    chosen_label = st.radio(
        "Which target citation is correct?",
        options=list(radio_label_to_value.keys()),
        key=f"choice__{qp.pair.source_id}",
    )
    chosen_value = radio_label_to_value[chosen_label]

    other_id = st.text_input(
        "Or type a different citation ID (overrides the radio when non-empty)",
        key=f"other__{qp.pair.source_id}",
    ).strip()
    final_chosen = other_id if other_id else chosen_value

    note = st.text_area(
        "Optional note",
        key=f"note__{qp.pair.source_id}",
        height=80,
    )

    if st.button("Submit verdict", type="primary", key=f"submit__{qp.pair.source_id}"):
        _submit_verdict(
            qp=qp,
            chosen=final_chosen,
            note=note,
            validated_path=selected_pair_file.validated_path,
            reviewer=reviewer,
        )
        # Streamlit caches the queue — force a rerun so the next item surfaces.
        st.rerun()


def _typecheck_imports_used() -> None:
    # Suppress unused-import lint for types we only reference via annotations.
    _: tuple = (TieredPair,)


main()

"""d'accord — side-by-side comparison view + CSV export.

Tier-12C deliverable. Replaces the originally-planned "Ask D'accord" chatbot
with a credential-surfacing comparison tool: three columns per query, each
tagged with provenance, CSV-exportable for compliance-team workflow fit.

Three columns:
  - **Retrieval baseline**: top-1 cosine over the gold-pair index. Provenance
    is always `gold-retrieval` when this column populates; otherwise
    `no-confident-match` (threshold below cutoff).
  - **d'accord (fine-tune)**: QLoRA-fine-tuned Qwen2.5-7B. Provenance is
    `fine-tune-generalization` when produced via the local adapter, or
    surfaced from the remote SageMaker endpoint's response.
  - **Base Qwen / Gold**: when input matches a gold-set pair (test/val
    membership), shows the gold answer; otherwise shows base Qwen2.5-7B
    (when local-mode + base model loaded). Empty when neither applies.

Two backing modes:
  - **Local mode** (default): wires `daccord.serving.HybridRouter` against
    the local FAISS index + LocalAdapterClient. Runs entirely on the
    consumer container's GPU. Used for the Loom demo + interview screenshare.
  - **Remote mode** (`?mode=remote` query string): calls the deployed
    SageMaker endpoint via boto3. Used for the live demo URL post-tier-15.

Run:
    docker compose run --rm --service-ports consumer \
        uv run streamlit run app.py --server.address 0.0.0.0
Then visit http://localhost:8501.

Configuration via env vars (read at app startup):
  - DACCORD_GOLD_PATH         — path to gold JSONL (for eval-set membership check)
  - DACCORD_INDEX_STEM        — path stem to FAISS index (.faiss + .jsonl)
  - DACCORD_ADAPTER_PATH      — path to QLoRA adapter dir
  - DACCORD_BASE_MODEL        — base Qwen model name (default: Qwen/Qwen2.5-7B-Instruct)
  - DACCORD_SAGEMAKER_ENDPOINT — endpoint name (remote mode only)
  - DACCORD_RETRIEVAL_THRESHOLD — cosine threshold (default: 0.75)
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from daccord.eval.clients import GroqClient, ModelClient, RetrievalClient
from daccord.eval.prompts import build_eval_prompt
from daccord.eval.schema import ModelResponse
from daccord.gold import GoldPair, GoldSet
from daccord.serving import HybridRouter, LocalAdapterClient

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_THRESHOLD = 0.75


@dataclass
class ColumnResult:
    """One column's worth of comparison output. Renders directly to Streamlit."""

    label: str
    provenance: str
    citation_id: str
    target_provision: str
    justification: str
    latency_ms: float
    parse_error: str | None
    model: str


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    return Path(raw) if raw else None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@st.cache_resource
def _load_gold() -> GoldSet | None:
    path = _env_path("DACCORD_GOLD_PATH")
    if path is None or not path.exists():
        return None
    return GoldSet.from_jsonl(path)


@st.cache_resource
def _load_retrieval() -> RetrievalClient | None:
    stem = _env_path("DACCORD_INDEX_STEM")
    if stem is None:
        return None
    try:
        return RetrievalClient(
            index_path=stem,
            embedder_name=os.environ.get(
                "DACCORD_EMBEDDER", "paraphrase-multilingual-mpnet-base-v2"
            ),
            score_threshold=_env_float("DACCORD_RETRIEVAL_THRESHOLD", DEFAULT_THRESHOLD),
        )
    except (FileNotFoundError, RuntimeError) as exc:
        st.warning(f"Retrieval client unavailable: {exc}")
        return None


@st.cache_resource
def _load_fine_tune() -> LocalAdapterClient | None:
    adapter = _env_path("DACCORD_ADAPTER_PATH")
    if adapter is None or not adapter.exists():
        # No adapter yet (pre-tier-12A). Surface as a warning so the demo
        # still runs retrieval-only.
        return None
    try:
        return LocalAdapterClient(
            adapter_path=adapter,
            base_model=os.environ.get("DACCORD_BASE_MODEL", DEFAULT_BASE_MODEL),
        )
    except (FileNotFoundError, RuntimeError) as exc:
        st.warning(f"Fine-tune adapter unavailable: {exc}")
        return None


@st.cache_resource
def _load_router(
    retrieval: RetrievalClient | None, fine_tune: LocalAdapterClient | None
) -> HybridRouter | None:
    if retrieval is None or fine_tune is None:
        return None
    return HybridRouter(retrieval_client=retrieval, fine_tune_client=fine_tune)


def _gold_lookup(gold: GoldSet | None, query: GoldPair) -> GoldPair | None:
    """Find a matching gold pair (same source_citation_id + target_jurisdiction)."""
    if gold is None:
        return None
    for p in gold.pairs:
        if (
            p.source_citation_id == query.source_citation_id
            and p.source_framework == query.source_framework
            and p.target_jurisdiction == query.target_jurisdiction
        ):
            return p
    return None


def _column_from_response(label: str, resp: ModelResponse, provenance: str) -> ColumnResult:
    cand = resp.top1
    return ColumnResult(
        label=label,
        provenance=provenance,
        citation_id=cand.citation_id if cand else "",
        target_provision=cand.target_mechanism if cand else "",
        justification=cand.mapping_justification if cand else "",
        latency_ms=resp.latency_ms,
        parse_error=resp.parse_error,
        model=resp.model,
    )


def _column_from_gold(label: str, gold: GoldPair) -> ColumnResult:
    return ColumnResult(
        label=label,
        provenance="gold-truth",
        citation_id=gold.target_citation_id,
        target_provision=gold.target_mechanism,
        justification="(hand-validated gold mapping)",
        latency_ms=0.0,
        parse_error=None,
        model="gold-set",
    )


def _column_empty(label: str, reason: str) -> ColumnResult:
    return ColumnResult(
        label=label,
        provenance="unavailable",
        citation_id="",
        target_provision="",
        justification="",
        latency_ms=0.0,
        parse_error=reason,
        model="-",
    )


def _run_local_mode(
    query: GoldPair,
    retrieval: RetrievalClient | None,
    fine_tune: LocalAdapterClient | None,
    base: ModelClient | None,
    gold: GoldSet | None,
) -> tuple[ColumnResult, ColumnResult, ColumnResult]:
    """Compute all three columns. Each column runs independently — even if the
    HybridRouter would route retrieval-first, the side-by-side view wants to
    SEE both retrieval and fine-tune outputs for comparison.
    """
    prompt = build_eval_prompt(query)

    if retrieval is not None:
        r = retrieval.generate(prompt, run_id="streamlit", batch_id=query.id)
        retrieval_col = _column_from_response(
            "Retrieval baseline",
            r,
            "gold-retrieval" if r.top1 is not None else "no-confident-match",
        )
    else:
        retrieval_col = _column_empty(
            "Retrieval baseline", "DACCORD_INDEX_STEM not set or load failed"
        )

    if fine_tune is not None:
        f = fine_tune.generate(prompt, run_id="streamlit", batch_id=query.id)
        fine_tune_col = _column_from_response(
            "d'accord (fine-tune)",
            f,
            "fine-tune-generalization" if f.top1 is not None else "no-confident-match",
        )
    else:
        fine_tune_col = _column_empty(
            "d'accord (fine-tune)", "DACCORD_ADAPTER_PATH not set; train at tier 12A"
        )

    # Third column: gold if input is in the eval set; otherwise base Qwen.
    gold_match = _gold_lookup(gold, query)
    if gold_match is not None:
        third_col = _column_from_gold("Gold answer", gold_match)
    elif base is not None:
        b = base.generate(prompt, run_id="streamlit", batch_id=query.id)
        third_col = _column_from_response("Base Qwen", b, "base-model")
    else:
        third_col = _column_empty(
            "Gold / Base", "no gold match and no base-model client configured"
        )

    return retrieval_col, fine_tune_col, third_col


def _render_column(col: ColumnResult) -> None:
    st.markdown(f"### {col.label}")
    st.caption(f"`{col.provenance}` · {col.model} · {col.latency_ms:.0f}ms")
    if col.parse_error and col.provenance in ("no-confident-match", "unavailable"):
        st.info(col.parse_error)
        return
    if not col.citation_id:
        st.warning("(no citation returned)")
        return
    st.markdown(f"**Citation**: `{col.citation_id}`")
    st.markdown(f"**Target provision**: {col.target_provision}")
    st.markdown(f"**Justification**: _{col.justification}_")


def _csv_bytes(query: GoldPair, columns: list[ColumnResult]) -> bytes:
    rows = [
        {
            "source_jurisdiction": query.source_jurisdiction,
            "source_framework": query.source_framework,
            "source_citation_id": query.source_citation_id,
            "source_clause": query.source_mechanism,
            "target_jurisdiction": query.target_jurisdiction,
            "target_framework": query.target_framework,
            "model": col.model,
            "column_label": col.label,
            "provenance": col.provenance,
            "citation_id": col.citation_id,
            "target_provision": col.target_provision,
            "justification": col.justification,
            "latency_ms": col.latency_ms,
            "parse_error": col.parse_error or "",
        }
        for col in columns
    ]
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def main() -> None:
    st.set_page_config(page_title="d'accord — side-by-side", layout="wide")
    st.title("d'accord — cross-jurisdiction mapping (side-by-side comparison)")
    st.caption(
        "Compare retrieval baseline vs fine-tuned d'accord vs gold/base on the same query. "
        "Export the row as CSV for compliance-team workflows."
    )

    gold = _load_gold()
    retrieval = _load_retrieval()
    fine_tune = _load_fine_tune()
    # Optional base-model comparator (Llama-70B via Groq, free-tier). Only
    # used as the third column when the input isn't in the gold set.
    base: ModelClient | None = None
    if os.environ.get("GROQ_API_KEY"):
        try:
            base = GroqClient()
        except RuntimeError:
            base = None

    with st.sidebar:
        st.subheader("Configuration")
        n_pairs = len(gold.pairs) if gold else 0
        st.markdown(f"- Gold set loaded: **{'yes' if gold else 'no'}** ({n_pairs} pairs)")
        st.markdown(f"- Retrieval client: **{'ready' if retrieval else 'unavailable'}**")
        st.markdown(f"- Fine-tune adapter: **{'ready' if fine_tune else 'unavailable'}**")
        st.markdown(f"- Base Qwen (via Groq): **{'ready' if base else 'unavailable'}**")
        threshold = _env_float("DACCORD_RETRIEVAL_THRESHOLD", DEFAULT_THRESHOLD)
        st.markdown(f"- Retrieval threshold: **{threshold:.2f}**")
        st.divider()
        st.caption("See `consumer/app.py` docstring for environment-variable configuration.")

    # Form: source clause + target jurisdiction
    with st.form("query"):
        col1, col2 = st.columns(2)
        with col1:
            source_jurisdiction = st.text_input("Source jurisdiction", value="eu")
            source_framework = st.text_input("Source framework", value="gdpr")
            source_citation_id = st.text_input("Source citation ID", value="Art. 32")
            source_clause = st.text_area(
                "Source clause text",
                value="Security of processing — appropriate technical and organisational measures.",
                height=120,
            )
        with col2:
            target_jurisdiction = st.text_input("Target jurisdiction", value="sg")
            target_framework = st.text_input("Target framework", value="pdpa_sg")
        submitted = st.form_submit_button("Compare")

    if not submitted:
        return

    query = GoldPair(
        id="streamlit-query",
        source_jurisdiction=source_jurisdiction,
        source_framework=source_framework,
        source_citation_id=source_citation_id,
        source_mechanism=source_clause,
        source_language="en",
        target_jurisdiction=target_jurisdiction,
        target_framework=target_framework,
        target_citation_id="",
        target_mechanism="",
        target_language="",
        notes=None,
    )

    retrieval_col, fine_tune_col, third_col = _run_local_mode(
        query, retrieval, fine_tune, base, gold
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        _render_column(retrieval_col)
    with c2:
        _render_column(fine_tune_col)
    with c3:
        _render_column(third_col)

    st.divider()
    st.download_button(
        label="Export this comparison as CSV",
        data=_csv_bytes(query, [retrieval_col, fine_tune_col, third_col]),
        file_name=f"daccord_comparison_{source_framework}_{target_framework}.csv",
        mime="text/csv",
    )


# Streamlit invokes the script top-down on each interaction; `main()` is
# the entry point, but the imports + cache_resource calls also fire on
# every rerun (cheap on cache hit).
main()

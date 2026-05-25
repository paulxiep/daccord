"""Tier-2D parser bake-off CLI: Marker vs Typhoon-OCR on a 5-page Thai sample.

End-to-end flow:

1. Read `data/parser_bakeoff/sample_pages.json` (page indices + rationale).
2. Rasterise each page → per-page PDF (for Marker) + PNG (for Typhoon).
3. Run each parser (skippable via `--skip-marker` / `--skip-typhoon`).
4. Extract `'มาตรา N'` citations and score precision/recall against
   `data/parser_bakeoff/expected_citation_ids.json` (the Thai-reader gold).
5. Emit `score_table.csv` with columns the reviewer fills in by hand
   (reading_order_1_to_5, structure_preserved_0_or_1, notes).
6. Log one MLflow run per parser under experiment `daccord-bakeoff-thai`.

The reading-order + structure-preservation columns require Thai literacy, so
the script writes them as empty strings; the reviewer edits the CSV in place
and reruns `--summarize-only` to regenerate `summary.md` from the filled CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import mlflow

from daccord.bakeoff.sample import PageArtifact, rasterize_pages
from daccord.bakeoff.scoring import (
    PageScore,
    aggregate,
    extract_citations,
    score_page,
)
from daccord.tracking import log_standard_params, setup_mlflow
from daccord.validation import validated

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = REPO_ROOT / "data" / "raw" / "th" / "pdpa_th" / "pdpa_th_thai_2019.pdf"
DEFAULT_OUT = REPO_ROOT / "data" / "parser_bakeoff"
EXPERIMENT_NAME = "daccord-bakeoff-thai"
PROJECT_SEED = 42

log = logging.getLogger("bakeoff_thai")


@validated
def _load_sample_pages(out_dir: Path) -> list[int]:
    """Read `sample_pages.json`; return the list of 0-indexed pages in order."""
    path = out_dir / "sample_pages.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; create it with {{'pages': [<idx>, ...], 'rationale': {{...}}}}"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    pages = raw["pages"]
    if not isinstance(pages, list) or not all(isinstance(p, int) for p in pages):
        raise ValueError(f"{path}: 'pages' must be a list of ints")
    return pages


@validated
def _load_expected_citations(out_dir: Path, page_indices: list[int]) -> dict[int, list[str]]:
    """Read `expected_citation_ids.json`; return a dict keyed by 0-indexed page.

    The on-disk file uses string keys (JSON requirement). Missing pages map to
    an empty list — that's a legitimate state for pages without `มาตรา N`
    headings, and the scorer treats `expected=[]` as `recall=None`.
    """
    path = out_dir / "expected_citation_ids.json"
    if not path.exists():
        log.warning("[skip ] %s missing — scoring will report no expected citations", path)
        return {idx: [] for idx in page_indices}
    raw = json.loads(path.read_text(encoding="utf-8"))
    mapping = raw.get("page_to_citation_ids", {})
    return {idx: list(mapping.get(str(idx), [])) for idx in page_indices}


@validated
def _run_marker(artifacts: list[PageArtifact], out_dir: Path) -> dict[int, str]:
    """Run Marker on every per-page PDF, return {page_index: markdown}."""
    from daccord.bakeoff import marker_runner

    log.info("[marker] starting (%d pages)", len(artifacts))
    outputs = marker_runner.parse_pages([a.pdf_path for a in artifacts], out_dir / "marker")
    for o in outputs:
        log.info("[marker] page %d → %s (%.1fs)", o.page_index, o.md_path.name, o.seconds_elapsed)
    return {o.page_index: o.markdown for o in outputs}


@validated
def _run_typhoon(artifacts: list[PageArtifact], out_dir: Path) -> dict[int, str]:
    """Run Typhoon-OCR on every per-page PNG, return {page_index: markdown}."""
    from daccord.bakeoff import typhoon_runner

    log.info("[typhoon] starting (%d pages)", len(artifacts))
    outputs = typhoon_runner.parse_pages([a.png_path for a in artifacts], out_dir / "typhoon")
    for o in outputs:
        log.info("[typhoon] page %d → %s (%.1fs)", o.page_index, o.md_path.name, o.seconds_elapsed)
    return {o.page_index: o.markdown for o in outputs}


@validated
def _score_parser(
    parser: str,
    page_to_md: dict[int, str],
    expected: dict[int, list[str]],
) -> list[PageScore]:
    return [
        score_page(
            parser=parser,
            page_index=idx,
            expected=expected.get(idx, []),
            extracted=extract_citations(md),
        )
        for idx, md in sorted(page_to_md.items())
    ]


@validated
def _write_score_table(csv_path: Path, scores: list[PageScore]) -> None:
    """Write the per-page scoring rows. Reviewer fills in the last three cols."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "parser",
        "page_index",
        "expected_citations",
        "extracted_citations",
        "true_positives",
        "false_positives",
        "false_negatives",
        "recall",
        "precision",
        "reading_order_1_to_5",
        "structure_preserved_0_or_1",
        "notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in scores:
            writer.writerow(
                {
                    "parser": s.parser,
                    "page_index": s.page_index,
                    "expected_citations": "; ".join(s.expected_citations),
                    "extracted_citations": "; ".join(s.extracted_citations),
                    "true_positives": "; ".join(s.true_positives),
                    "false_positives": "; ".join(s.false_positives),
                    "false_negatives": "; ".join(s.false_negatives),
                    "recall": "" if s.recall is None else f"{s.recall:.3f}",
                    "precision": "" if s.precision is None else f"{s.precision:.3f}",
                    "reading_order_1_to_5": (
                        "" if s.reading_order_1_to_5 is None else str(s.reading_order_1_to_5)
                    ),
                    "structure_preserved_0_or_1": (
                        ""
                        if s.structure_preserved_0_or_1 is None
                        else str(s.structure_preserved_0_or_1)
                    ),
                    "notes": s.notes,
                }
            )


@validated
def _write_summary(
    summary_path: Path,
    scores_by_parser: dict[str, list[PageScore]],
    source_pdf: Path,
    page_indices: list[int],
) -> None:
    """Write the prose `summary.md` that closes M1's parser-rationale artifact."""
    lines = [
        "# Thai parser bake-off — summary",
        "",
        f"Source PDF: `{source_pdf.relative_to(REPO_ROOT)}`",
        f"Sample pages (0-indexed): {page_indices}",
        f"Parsers compared: {sorted(scores_by_parser.keys())}",
        "",
        "## Per-parser aggregates",
        "",
        ("| Parser | Pages | Recall | Precision | Reading order | Structure preserved |"),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for parser, scores in sorted(scores_by_parser.items()):
        agg = aggregate(parser, scores)
        lines.append(
            f"| {agg.parser} | {agg.page_count}"
            f" | {_fmt(agg.citation_recall_mean)}"
            f" | {_fmt(agg.citation_precision_mean)}"
            f" | {_fmt(agg.reading_order_mean)}"
            f" | {_fmt(agg.structure_preserved_frac)} |"
        )
    lines += [
        "",
        "## Picked-winner rule",
        "",
        "1. Highest reading-order mean wins (Thai-reader judgment, weight 1).",
        "2. Tie-break on highest citation recall (M1 registry extraction depends on this).",
        "3. If both parsers' reading-order mean is below 3.0, invoke the M1 cut criterion",
        "   (drop Royal Gazette amendments, retain PDPA-TH core only) per",
        "   `docs/development_plan.md §M1`.",
        "",
    ]
    decision = _pick_winner(scores_by_parser)
    if decision is not None:
        lines += [
            "## Decision",
            "",
            *decision,
            "",
        ]
    else:
        lines += [
            "## Reviewer notes",
            "",
            "Reviewer (Thai reader) fills reading_order_1_to_5 + structure_preserved_0_or_1",
            "+ notes in `score_table.csv` after inspecting each parser's per-page markdown",
            "under `data/parser_bakeoff/{marker,typhoon}/page_<n>.md`. Then rerun with",
            "`--summarize-only` to regenerate this file.",
            "",
        ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")


@validated
def _pick_winner(scores_by_parser: dict[str, list[PageScore]]) -> list[str] | None:
    """Emit the Decision section when the rubric can name a single winner.

    Returns `None` if reviewer columns are still blank for at least one parser
    (no reading-order mean → no winner yet) or if both parsers' reading-order
    means tie AND their citation-recall means also tie (no rubric tiebreaker
    yields a unique pick — leave the Decision section out and keep the
    Reviewer-notes prompt instead).
    """
    aggs = {p: aggregate(p, s) for p, s in scores_by_parser.items()}
    if any(a.reading_order_mean is None for a in aggs.values()):
        return None

    sorted_by_ro = sorted(
        aggs.items(), key=lambda kv: kv[1].reading_order_mean or 0.0, reverse=True
    )
    top_parser, top_agg = sorted_by_ro[0]

    runner_line = ""
    tiebreak = "reading-order mean"
    if len(sorted_by_ro) > 1:
        runner_parser, runner_agg = sorted_by_ro[1]
        if top_agg.reading_order_mean == runner_agg.reading_order_mean:
            if top_agg.citation_recall_mean == runner_agg.citation_recall_mean:
                return None
            tiebreak = "citation-recall tiebreak"
        runner_line = (
            f" vs. `{runner_parser}` "
            f"(ro={_fmt(runner_agg.reading_order_mean)}, "
            f"recall={_fmt(runner_agg.citation_recall_mean)})"
        )

    return [
        f"**Locked parser: `{top_parser}`** — wins on {tiebreak} "
        f"(ro={_fmt(top_agg.reading_order_mean)}, recall={_fmt(top_agg.citation_recall_mean)})"
        + runner_line
        + ".",
        "",
        (
            "This is the M1 parser-choice artifact: tier 4 (full-corpus parse to markdown)"
            f" uses `{top_parser}` for Thai PDFs; English-language frameworks already use"
            " `marker` by default."
        ),
    ]


def _fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


@validated
def _mlflow_log(
    parser: str,
    parser_version_str: str,
    extra_params: dict[str, str],
    scores: list[PageScore],
    source_pdf: Path,
    page_indices: list[int],
) -> None:
    """Log one MLflow run per parser. Metrics use `_mean` suffix per docs/MLFLOW.md."""
    run_name = f"bakeoff-{parser}-v1"
    with mlflow.start_run(run_name=run_name):
        log_standard_params(
            run_name=run_name,
            seed=PROJECT_SEED,
            extra={
                "parser": parser,
                "parser_version": parser_version_str,
                "sample_pdf": str(source_pdf.relative_to(REPO_ROOT)),
                "page_count": str(len(page_indices)),
                **extra_params,
            },
        )
        agg = aggregate(parser, scores)
        if agg.citation_recall_mean is not None:
            mlflow.log_metric("citation_recall_mean", agg.citation_recall_mean)
        if agg.citation_precision_mean is not None:
            mlflow.log_metric("citation_precision_mean", agg.citation_precision_mean)
        if agg.reading_order_mean is not None:
            mlflow.log_metric("reading_order_mean", agg.reading_order_mean)
        if agg.structure_preserved_frac is not None:
            mlflow.log_metric("structure_preserved_frac", agg.structure_preserved_frac)


@validated
def _read_existing_outputs(
    parser: str, page_indices: list[int], out_dir: Path
) -> dict[int, str] | None:
    """Return `{page_index: markdown}` if every page's MD file already exists.

    Used by `--summarize-only` to score + log without re-running the parsers.
    Returns `None` if any expected page output is missing (caller skips parser).
    """
    parser_dir = out_dir / parser
    page_to_md: dict[int, str] = {}
    for idx in page_indices:
        md_path = parser_dir / f"page_{idx}.md"
        if not md_path.exists():
            return None
        page_to_md[idx] = md_path.read_text(encoding="utf-8")
    return page_to_md


@validated
def _read_csv_reviewer_fills(csv_path: Path) -> dict[tuple[str, int], dict[str, str]]:
    """Read any reviewer-filled reading_order / structure_preserved / notes columns.

    `--summarize-only` overwrites the auto-derived columns but preserves the
    three human-judgement columns by reading the existing CSV first and merging
    back in. Returns `{(parser, page_index): {reading_order, structure_preserved, notes}}`.
    """
    if not csv_path.exists():
        return {}
    out: dict[tuple[str, int], dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["parser"], int(row["page_index"]))
            out[key] = {
                "reading_order_1_to_5": row.get("reading_order_1_to_5", ""),
                "structure_preserved_0_or_1": row.get("structure_preserved_0_or_1", ""),
                "notes": row.get("notes", ""),
            }
    return out


@validated
def _apply_reviewer_fills(
    scores: list[PageScore],
    fills: dict[tuple[str, int], dict[str, str]],
) -> None:
    """Mutate `scores` in place, copying reviewer-filled cells from `fills`."""
    for s in scores:
        cells = fills.get((s.parser, s.page_index), {})
        if cells.get("reading_order_1_to_5"):
            s.reading_order_1_to_5 = int(cells["reading_order_1_to_5"])
        if cells.get("structure_preserved_0_or_1"):
            s.structure_preserved_0_or_1 = int(cells["structure_preserved_0_or_1"])
        if cells.get("notes"):
            s.notes = cells["notes"]


@validated
def run(
    source_pdf: Path,
    out_dir: Path,
    skip_marker: bool,
    skip_typhoon: bool,
    use_mlflow: bool,
    summarize_only: bool,
) -> int:
    if not source_pdf.exists():
        log.error("source PDF not found: %s", source_pdf)
        return 2

    page_indices = _load_sample_pages(out_dir)
    expected = _load_expected_citations(out_dir, page_indices)

    csv_path = out_dir / "score_table.csv"
    summary_path = out_dir / "summary.md"
    reviewer_fills = _read_csv_reviewer_fills(csv_path)

    if not summarize_only:
        log.info("[sample] rasterising %d pages from %s", len(page_indices), source_pdf.name)
        artifacts = rasterize_pages(source_pdf, page_indices, out_dir)
    else:
        artifacts = []  # no rasterise; parsers will be skipped

    if use_mlflow:
        # Pin tracking to the repo-root mlruns/ — bakeoff CLI is invoked from
        # envs/bakeoff/ so the tracking-URI default `file:./mlruns` would land
        # in envs/bakeoff/mlruns/ otherwise. We want one unified MLflow history.
        setup_mlflow(
            experiment_name=EXPERIMENT_NAME,
            tracking_uri=f"file:{(REPO_ROOT / 'mlruns').as_posix()}",
        )

    all_scores: list[PageScore] = []
    scores_by_parser: dict[str, list[PageScore]] = {}

    if not skip_marker:
        if summarize_only:
            page_to_md = _read_existing_outputs("marker", page_indices, out_dir)
            if page_to_md is None:
                log.warning("[marker] no existing outputs found in --summarize-only mode")
                page_to_md = {}
            else:
                log.info("[marker] reusing %d existing per-page outputs", len(page_to_md))
        else:
            page_to_md = _run_marker(artifacts, out_dir)

        if page_to_md:
            marker_scores = _score_parser("marker", page_to_md, expected)
            _apply_reviewer_fills(marker_scores, reviewer_fills)
            scores_by_parser["marker"] = marker_scores
            all_scores.extend(marker_scores)
            if use_mlflow:
                from daccord.bakeoff import marker_runner

                _mlflow_log(
                    parser="marker",
                    parser_version_str=marker_runner.parser_version(),
                    extra_params={},
                    scores=marker_scores,
                    source_pdf=source_pdf,
                    page_indices=page_indices,
                )

    if not skip_typhoon:
        if summarize_only:
            page_to_md = _read_existing_outputs("typhoon", page_indices, out_dir)
            if page_to_md is None:
                log.warning("[typhoon] no existing outputs found in --summarize-only mode")
                page_to_md = {}
            else:
                log.info("[typhoon] reusing %d existing per-page outputs", len(page_to_md))
        else:
            page_to_md = _run_typhoon(artifacts, out_dir)

        if page_to_md:
            typhoon_scores = _score_parser("typhoon", page_to_md, expected)
            _apply_reviewer_fills(typhoon_scores, reviewer_fills)
            scores_by_parser["typhoon"] = typhoon_scores
            all_scores.extend(typhoon_scores)
            if use_mlflow:
                from daccord.bakeoff import typhoon_runner

                _mlflow_log(
                    parser="typhoon",
                    parser_version_str=typhoon_runner.parser_version(),
                    extra_params={"model_id": typhoon_runner.model_id()},
                    scores=typhoon_scores,
                    source_pdf=source_pdf,
                    page_indices=page_indices,
                )

    _write_score_table(csv_path, all_scores)
    _write_summary(summary_path, scores_by_parser, source_pdf, page_indices)

    log.info("[done] wrote %s and %s", csv_path.name, summary_path.name)
    log.info("[done] reviewer: fill reading_order + structure_preserved + notes in CSV")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pdf", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-marker", action="store_true")
    parser.add_argument("--skip-typhoon", action="store_true")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help=(
            "skip parser runs; re-score existing per-page markdown under "
            "data/parser_bakeoff/{marker,typhoon}/ and rebuild score_table.csv "
            "+ summary.md, preserving any reviewer-filled columns in the CSV"
        ),
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="bypass the per-parser MLflow runs (useful during the iteration loop)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return run(
        source_pdf=args.source_pdf,
        out_dir=args.out_dir,
        skip_marker=args.skip_marker,
        skip_typhoon=args.skip_typhoon,
        use_mlflow=not args.no_mlflow,
        summarize_only=args.summarize_only,
    )


if __name__ == "__main__":
    sys.exit(main())

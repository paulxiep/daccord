"""Tier-7A CLI: submit + poll Bedrock async batch jobs for the F9 ensemble.

End-to-end flow:

  submit   Build prompts + start batch jobs. Per (framework_pair, model):
             1. Build BatchPrompt list (source clauses × pinned target registry).
             2. Pre-flight cost cap via daccord.costs.preflight().
             3. Upload input JSONL to s3://{bucket}/ensemble/{in,smoke/in}/...
             4. Call bedrock:CreateModelInvocationJob.
             5. Append a row to data/ensemble/jobs.jsonl (or smoke-jobs.jsonl).

  poll     Read the jobs ledger, fetch get_model_invocation_job for each
           non-terminal row. On Completed: download outputs from S3, parse
           per-model envelopes into EnsembleCandidate rows, write
           data/ensemble/raw/{pair}__{slug}.jsonl, call record_call().

  status   Print the current ledger as a per-job status table.

Smoke mode (`--smoke`):
  - Source clauses come from data/gold/toy_v1.jsonl (already hand-built —
    bypasses the data/clauses/*.json extractor path).
  - One framework_pair only: gdpr__pdpa_sg.
  - Target = 5 source clauses × 4 F9 models = 20 invocations, ~$0.01.
  - S3 uses a `smoke/` prefix so artifacts don't collide with real-run state.
  - Ledger lives at data/ensemble/smoke-jobs.jsonl (separate file).

Idempotency:
  - Re-running `submit` skips (pair, model) tuples already present in the
    ledger with status in {Submitted, Validating, Scheduled, InProgress,
    Completed} — only Failed/Stopped get re-submitted.
  - `poll` is naturally idempotent: it only updates non-terminal rows.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3

from daccord.aws import batch as awsbatch
from daccord.aws import m2
from daccord.costs import preflight, record_call
from daccord.eval.prompts import build_ensemble_prompt
from daccord.eval.registry import load_registry
from daccord.gold import GoldSet
from daccord.registry.schema import read_clauses, read_manifest
from daccord.validation import ValidatedModel, validated

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_DIR = REPO_ROOT / "data" / "registry"
DEFAULT_CLAUSES_DIR = REPO_ROOT / "data" / "clauses"
DEFAULT_TOY_GOLD = REPO_ROOT / "data" / "gold" / "toy_v1.jsonl"
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "ensemble" / "raw"
JOBS_LEDGER = REPO_ROOT / "data" / "ensemble" / "jobs.jsonl"
SMOKE_JOBS_LEDGER = REPO_ROOT / "data" / "ensemble" / "smoke-jobs.jsonl"

SMOKE_FRAMEWORK_PAIR = "gdpr__pdpa_sg"
"""Hand-picked smoke pair: GDPR sources have the most toy_v1 entries (13) and
PDPA-SG is the most common SEA target — exercises the registry-pinning prompt
constraint with a realistic mismatch (EU article → SG section)."""

SMOKE_CLAUSE_COUNT = 5
"""20 invocations total = 5 clauses × 4 F9 seats. Per-batch <$0.005."""

DEFAULT_MAX_TOKENS = 256
"""Output cap. Per-prompt output is one JSON object — observed ~150 tokens
on toy_v1 dry runs. 256 has comfortable headroom without enabling Nova's
extended-thinking blocks."""

POLL_INTERVAL_SECONDS = 60
"""Bedrock batch SLA is 1-12h typical, 24h max. Poll every minute keeps the
operator informed without hammering the API."""

TERMINAL_STATUSES = {"Completed", "Failed", "Stopped", "Expired", "PartiallyCompleted"}
"""Statuses that mean "don't poll this row again" — either landed or won't."""

log = logging.getLogger("run_ensemble")


# ─────────────────────────────────────────────────────────────────────────────
# Job ledger — append-friendly JSONL persisting one row per (pair, model).
# ─────────────────────────────────────────────────────────────────────────────


class JobLedgerEntry(ValidatedModel):
    """One submitted batch job's state. Re-loaded by `poll` + `status`."""

    framework_pair: str
    model: str
    job_arn: str
    job_name: str
    status: str
    smoke: bool
    prompt_count: int
    input_s3_uri: str
    output_s3_uri: str
    submitted_at: str
    completed_at: str | None = None
    parsed_ok: int | None = None
    parse_errors: int | None = None


@validated
def read_ledger(path: Path) -> list[JobLedgerEntry]:
    if not path.exists():
        return []
    out: list[JobLedgerEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            out.append(JobLedgerEntry.model_validate_json(stripped))
    return out


@validated
def write_ledger(path: Path, entries: list[JobLedgerEntry]) -> None:
    """Atomic write — temp file + replace, same pattern as registry manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
        suffix=".jsonl.tmp",
    ) as tmp:
        for entry in entries:
            tmp.write(entry.model_dump_json())
            tmp.write("\n")
        tmp_name = tmp.name
    Path(tmp_name).replace(path)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ─────────────────────────────────────────────────────────────────────────────
# Framework-pair + source-clause enumeration.
# ─────────────────────────────────────────────────────────────────────────────


@validated
def enumerate_framework_pairs(registry_dir: Path) -> list[str]:
    """Cross product of frameworks in `data/registry/manifest.jsonl` minus self-pairs.

    9 frameworks → 72 ordered pairs. Stable order: sort by source then target.
    """
    manifest = read_manifest(registry_dir / "manifest.jsonl")
    frameworks = sorted({row.framework for row in manifest})
    pairs: list[str] = []
    for src, tgt in itertools.product(frameworks, repeat=2):
        if src == tgt:
            continue
        pairs.append(f"{src}__{tgt}")
    return pairs


@validated
def parse_framework_pair(framework_pair: str) -> tuple[str, str]:
    """`gdpr__pdpa_sg` → `('gdpr', 'pdpa_sg')`. Splits on the LAST `__`.

    Some framework_ids contain underscores (e.g. `pdpa_my`); split-on-last
    so `gdpr__pdpa_my` parses to `('gdpr', 'pdpa_my')`, not `('gdpr__pdpa', 'my')`.
    """
    idx = framework_pair.rfind("__")
    if idx <= 0:
        raise ValueError(f"malformed framework_pair: {framework_pair!r}")
    return (framework_pair[:idx], framework_pair[idx + 2 :])


@validated
def build_prompts_for_pair(
    *,
    framework_pair: str,
    registry_dir: Path,
    clauses_dir: Path,
    max_tokens: int,
    max_clauses: int | None,
) -> list[awsbatch.BatchPrompt]:
    """One BatchPrompt per source citation in the source framework's clauses file.

    Loads source clauses from `data/clauses/{source}.json` and the target
    framework's registry from `data/registry/{target}.json`. Source clauses
    missing a body (extractor recall < 1.0) get citation_id as a degraded
    `source_mechanism` so the prompt still works.
    """
    source_fw, target_fw = parse_framework_pair(framework_pair)
    src_clauses_path = clauses_dir / f"{source_fw}.json"
    if not src_clauses_path.exists():
        raise FileNotFoundError(
            f"no clauses file for source framework {source_fw!r}: {src_clauses_path}. "
            f"Run scripts/extract_clauses.py first."
        )
    clauses = read_clauses(src_clauses_path)
    target_registry = load_registry(target_fw, registry_dir)
    src_jurisdiction = clauses.jurisdiction
    # target_jurisdiction comes from the target framework's registry too.
    target_jurisdiction_lookup = read_manifest(registry_dir / "manifest.jsonl")
    target_jur = next(
        (r.jurisdiction for r in target_jurisdiction_lookup if r.framework == target_fw),
        target_fw,  # fallback shouldn't normally happen
    )

    # Iterate the registry's citation_ids order so the prompt order is stable.
    registry = load_registry(source_fw, registry_dir)
    prompts: list[awsbatch.BatchPrompt] = []
    for cid in registry.citation_ids:
        body = clauses.clauses.get(cid, cid)  # fallback to citation_id when body missing
        messages = build_ensemble_prompt(
            source_jurisdiction=src_jurisdiction,
            source_framework=source_fw,
            source_citation_id=cid,
            source_mechanism=body,
            target_jurisdiction=target_jur,
            target_framework=target_fw,
            target_registry=target_registry.citation_ids,
        )
        prompts.append(
            awsbatch.BatchPrompt(
                record_id=f"{source_fw}-{cid}",
                source_id=f"{source_fw}-{cid}",
                source_jurisdiction=src_jurisdiction,
                source_framework=source_fw,
                source_citation_id=cid,
                source_mechanism=body,
                target_jurisdiction=target_jur,
                target_framework=target_fw,
                system=messages.system,
                user=messages.user,
                max_tokens=max_tokens,
            )
        )
        if max_clauses is not None and len(prompts) >= max_clauses:
            break
    return prompts


@validated
def build_smoke_prompts(
    *,
    toy_gold_path: Path,
    registry_dir: Path,
    max_tokens: int,
) -> list[awsbatch.BatchPrompt]:
    """5 GDPR source clauses from toy_v1 retargeted at PDPA-SG.

    Uses `GoldPair.source_mechanism` (real hand-built clause text) so the
    smoke test exercises the prompt builder + Bedrock submit/poll loop with
    realistic input — not the extractor's possibly-noisy body output.
    """
    gold = GoldSet.from_jsonl(toy_gold_path)
    gdpr_sources = [p for p in gold.pairs if p.source_framework == "gdpr"][:SMOKE_CLAUSE_COUNT]
    if len(gdpr_sources) < SMOKE_CLAUSE_COUNT:
        raise RuntimeError(
            f"toy_v1 has only {len(gdpr_sources)} GDPR-source pairs; "
            f"need {SMOKE_CLAUSE_COUNT} for the smoke test"
        )
    target_registry = load_registry("pdpa_sg", registry_dir)
    prompts: list[awsbatch.BatchPrompt] = []
    for pair in gdpr_sources:
        messages = build_ensemble_prompt(
            source_jurisdiction=pair.source_jurisdiction,
            source_framework=pair.source_framework,
            source_citation_id=pair.source_citation_id,
            source_mechanism=pair.source_mechanism,
            target_jurisdiction="sg",
            target_framework="pdpa_sg",
            target_registry=target_registry.citation_ids,
        )
        prompts.append(
            awsbatch.BatchPrompt(
                record_id=pair.id,
                source_id=pair.id,
                source_jurisdiction=pair.source_jurisdiction,
                source_framework=pair.source_framework,
                source_citation_id=pair.source_citation_id,
                source_mechanism=pair.source_mechanism,
                target_jurisdiction="sg",
                target_framework="pdpa_sg",
                system=messages.system,
                user=messages.user,
                max_tokens=max_tokens,
            )
        )
    return prompts


# ─────────────────────────────────────────────────────────────────────────────
# Cost-tracker integration — preflight before submit, record_call after parse.
# ─────────────────────────────────────────────────────────────────────────────


@validated
def estimate_tokens_per_prompt(prompt: awsbatch.BatchPrompt) -> tuple[int, int]:
    """Rough token estimate per (input, output) — used for pre-flight cost check.

    Chars/4 heuristic matches the existing eval/clients.py convention. Output
    is the configured `max_tokens` cap (worst case for cost budgeting).
    """
    input_chars = len(prompt.system) + len(prompt.user)
    return (input_chars // 4, prompt.max_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Submit subcommand.
# ─────────────────────────────────────────────────────────────────────────────


def _make_session(profile: str | None) -> tuple[Any, str]:
    """Resolve profile, build boto3.Session, return (session, account_id)."""
    resolved_profile = m2.resolve_profile(profile)
    session = boto3.Session(profile_name=resolved_profile, region_name=m2.REGION)
    sts = session.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    log.info("using AWS profile=%s account=%s region=%s", resolved_profile, account_id, m2.REGION)
    return (session, account_id)


def _resolve_pair_set(args: argparse.Namespace, registry_dir: Path) -> list[str]:
    if args.smoke:
        return [SMOKE_FRAMEWORK_PAIR]
    if args.framework_pair:
        return [args.framework_pair]
    return enumerate_framework_pairs(registry_dir)


def _ledger_key(entry: JobLedgerEntry) -> tuple[str, str]:
    return (entry.framework_pair, entry.model)


def _already_submitted(
    ledger: list[JobLedgerEntry], framework_pair: str, model: str
) -> JobLedgerEntry | None:
    for e in ledger:
        if e.framework_pair == framework_pair and e.model == model:
            return e
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Poll subcommand.
# ─────────────────────────────────────────────────────────────────────────────


def _save_prompts_cache(
    framework_pair: str, prompts: list[awsbatch.BatchPrompt], smoke: bool
) -> None:
    """Persist prompts for `parse_model_output` to recover source_* context.

    The output parser needs the original BatchPrompt to attach source_id /
    source_mechanism / etc. We cache prompts at submit time under
    `data/ensemble/{smoke_,}prompts/{pair}.jsonl` so a separate `poll`
    invocation (possibly a different shell) can recover them.
    """
    cache_dir = REPO_ROOT / "data" / "ensemble" / ("smoke-prompts" if smoke else "prompts")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{framework_pair}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(p.model_dump_json())
            f.write("\n")


def _load_prompts_cache(framework_pair: str, smoke: bool) -> list[awsbatch.BatchPrompt]:
    cache_dir = REPO_ROOT / "data" / "ensemble" / ("smoke-prompts" if smoke else "prompts")
    path = cache_dir / f"{framework_pair}.jsonl"
    if not path.exists():
        return []
    out: list[awsbatch.BatchPrompt] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            out.append(awsbatch.BatchPrompt.model_validate_json(stripped))
    return out


def cmd_submit(args: argparse.Namespace) -> int:
    """Build prompts per pair, cache them on disk (for poll to recover the
    source-side context later), pre-flight the cost cap, then submit batch jobs."""
    session, account_id = _make_session(args.profile)
    registry_dir = args.registry_dir
    clauses_dir = args.clauses_dir
    ledger_path = SMOKE_JOBS_LEDGER if args.smoke else JOBS_LEDGER
    ledger = read_ledger(ledger_path)

    pairs = _resolve_pair_set(args, registry_dir)
    log.info(
        "submit: %d framework-pair(s) × %d model(s) (smoke=%s)",
        len(pairs),
        len(m2.F9_BEDROCK_MODELS),
        args.smoke,
    )

    submitted = 0
    skipped = 0
    for framework_pair in pairs:
        if args.smoke:
            prompts = build_smoke_prompts(
                toy_gold_path=args.toy_gold,
                registry_dir=registry_dir,
                max_tokens=args.max_tokens,
            )
        else:
            prompts = build_prompts_for_pair(
                framework_pair=framework_pair,
                registry_dir=registry_dir,
                clauses_dir=clauses_dir,
                max_tokens=args.max_tokens,
                max_clauses=args.max_clauses_per_pair,
            )
        if not prompts:
            log.warning("[submit] %s no prompts — skip", framework_pair)
            continue

        # Cache prompts so a separate `poll` shell can parse outputs.
        if not args.dry_run:
            _save_prompts_cache(framework_pair, prompts, smoke=args.smoke)

        for model in m2.F9_BEDROCK_MODELS:
            existing = _already_submitted(ledger, framework_pair, model)
            if existing is not None and existing.status not in {"Failed", "Stopped", "Expired"}:
                log.info(
                    "[submit] %s / %s already %s — skip",
                    framework_pair,
                    model,
                    existing.status,
                )
                skipped += 1
                continue

            est_in = sum(estimate_tokens_per_prompt(p)[0] for p in prompts)
            est_out = sum(estimate_tokens_per_prompt(p)[1] for p in prompts)
            try:
                preflight(
                    provider="bedrock_batch",
                    model=model,
                    est_input_tokens=est_in,
                    est_output_tokens=est_out,
                )
            except Exception as exc:
                log.error("[submit] %s / %s preflight rejected: %s", framework_pair, model, exc)
                return 3

            if args.dry_run:
                log.info(
                    "[submit][dry-run] would submit %s / %s "
                    "(%d prompts, est %d in + %d out tokens)",
                    framework_pair,
                    model,
                    len(prompts),
                    est_in,
                    est_out,
                )
                continue

            jsonl_content = awsbatch.build_batch_jsonl(prompts, model=model)
            job_arn = awsbatch.submit_job(
                session=session,
                account_id=account_id,
                framework_pair=framework_pair,
                model=model,
                jsonl_content=jsonl_content,
                smoke=args.smoke,
            )
            in_uri, out_uri = awsbatch.s3_prefix_for(
                account_id, framework_pair, model, smoke=args.smoke
            )
            slug = awsbatch.model_slug(model)
            job_name = f"daccord-{framework_pair}-{slug}".replace("_", "-").lower()[:63].rstrip("-")
            entry = JobLedgerEntry(
                framework_pair=framework_pair,
                model=model,
                job_arn=job_arn,
                job_name=job_name,
                status="Submitted",
                smoke=args.smoke,
                prompt_count=len(prompts),
                input_s3_uri=in_uri,
                output_s3_uri=out_uri,
                submitted_at=_now_iso(),
            )
            ledger = [e for e in ledger if _ledger_key(e) != _ledger_key(entry)] + [entry]
            write_ledger(ledger_path, ledger)
            submitted += 1
            log.info(
                "[submit] %s / %s OK (%d prompts) job=%s",
                framework_pair,
                model,
                len(prompts),
                job_arn,
            )

    log.info("[done] submit: submitted=%d skipped=%d", submitted, skipped)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Sync subcommand — synchronous InvokeModel path. Used when Bedrock batch
# isn't enabled on the account. Two-times cost vs batch, ~9h for the full
# 72-pair run parallelized across 4 model-workers.
# ─────────────────────────────────────────────────────────────────────────────


def _run_one_model_sync(
    *,
    session: Any,
    model: str,
    framework_pair: str,
    prompts: list[awsbatch.BatchPrompt],
    raw_dir: Path,
) -> dict[str, int | str]:
    """Worker fn: serially InvokeModel for each prompt under one (pair, model).

    Writes `data/ensemble/raw/{pair}__{slug}.jsonl` atomically + returns
    stats dict for cost accounting. Per-prompt exceptions are caught and
    surface as `EnsembleCandidate.parse_error` rows so one bad prompt doesn't
    tank the whole worker.
    """
    candidates = []
    total_input_chars = 0
    total_output_chars = 0
    errors = 0
    for prompt in prompts:
        try:
            response_body = awsbatch.invoke_sync(
                session=session,
                model=model,
                system=prompt.system,
                user=prompt.user,
                max_tokens=prompt.max_tokens,
            )
            cand = awsbatch.candidate_from_invoke_response(
                prompt=prompt, response_body=response_body, model=model
            )
            candidates.append(cand)
            total_input_chars += len(prompt.system) + len(prompt.user)
            total_output_chars += len(cand.target_mechanism) + len(cand.mapping_justification)
            if cand.parse_error is not None:
                errors += 1
        except Exception as exc:
            log.error(
                "[sync] %s / %s record=%s failed: %s",
                framework_pair,
                model,
                prompt.record_id,
                exc,
            )
            errors += 1
            candidates.append(
                awsbatch.EnsembleCandidate(
                    source_id=prompt.source_id,
                    source_jurisdiction=prompt.source_jurisdiction,
                    source_framework=prompt.source_framework,
                    source_citation_id=prompt.source_citation_id,
                    source_mechanism=prompt.source_mechanism,
                    target_jurisdiction=prompt.target_jurisdiction,
                    target_framework=prompt.target_framework,
                    model=model,
                    citation_id="",
                    target_mechanism="",
                    mapping_justification="",
                    parse_error=f"InvokeModel exception: {exc!r}",
                )
            )

    slug = awsbatch.model_slug(model)
    raw_path = raw_dir / f"{framework_pair}__{slug}.jsonl"
    awsbatch.write_candidates_jsonl(raw_path, candidates)
    return {
        "framework_pair": framework_pair,
        "model": model,
        "candidate_count": len(candidates),
        "errors": errors,
        "total_input_chars": total_input_chars,
        "total_output_chars": total_output_chars,
        "raw_path": str(raw_path),
    }


def cmd_run_sync(args: argparse.Namespace) -> int:
    """Synchronous InvokeModel path — 4 workers (one per F9 seat).

    For each (framework_pair, model), the assigned worker calls
    `bedrock-runtime:InvokeModel` per prompt sequentially. Workers run in
    parallel across the 4 models — naturally respecting per-model RPM quotas
    (each model has its own quota; one thread per model never overlaps).

    Smoke: 1 pair × 4 models × 5 prompts = 20 invocations, ~$0.02, ~1-2 min.
    Full: 72 pairs × 4 models × ~110 prompts = ~32K invocations, ~$21, ~9h.
    """
    session, _account_id = _make_session(args.profile)
    registry_dir = args.registry_dir
    clauses_dir = args.clauses_dir
    raw_dir = args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    pairs = _resolve_pair_set(args, registry_dir)
    log.info(
        "run-sync: %d framework-pair(s) × %d model(s) (smoke=%s)",
        len(pairs),
        len(m2.F9_BEDROCK_MODELS),
        args.smoke,
    )

    # Build (pair → prompts) up front so we can pre-flight cost across the run.
    prompts_by_pair: dict[str, list[awsbatch.BatchPrompt]] = {}
    for framework_pair in pairs:
        if args.smoke:
            prompts = build_smoke_prompts(
                toy_gold_path=args.toy_gold,
                registry_dir=registry_dir,
                max_tokens=args.max_tokens,
            )
        else:
            prompts = build_prompts_for_pair(
                framework_pair=framework_pair,
                registry_dir=registry_dir,
                clauses_dir=clauses_dir,
                max_tokens=args.max_tokens,
                max_clauses=args.max_clauses_per_pair,
            )
        if prompts:
            prompts_by_pair[framework_pair] = prompts

    # Pre-flight: aggregate token estimate per model across all pairs.
    for model in m2.F9_BEDROCK_MODELS:
        est_in = sum(
            estimate_tokens_per_prompt(p)[0]
            for prompts in prompts_by_pair.values()
            for p in prompts
        )
        est_out = sum(
            estimate_tokens_per_prompt(p)[1]
            for prompts in prompts_by_pair.values()
            for p in prompts
        )
        try:
            preflight(
                provider="bedrock_batch",  # use same provider key; sync is 2× batch cost
                model=model,
                est_input_tokens=est_in,
                est_output_tokens=est_out,
            )
        except Exception as exc:
            log.error("[run-sync] preflight rejected for %s: %s", model, exc)
            return 3

    if args.dry_run:
        for framework_pair, prompts in prompts_by_pair.items():
            log.info(
                "[run-sync][dry-run] %s: %d prompts × %d models = %d invocations",
                framework_pair,
                len(prompts),
                len(m2.F9_BEDROCK_MODELS),
                len(prompts) * len(m2.F9_BEDROCK_MODELS),
            )
        return 0

    # Build a flat work list: one (pair, model) tuple per worker job.
    work_items = [
        (fp, model, prompts_by_pair[fp]) for fp in prompts_by_pair for model in m2.F9_BEDROCK_MODELS
    ]
    log.info(
        "[run-sync] dispatching %d (pair, model) jobs across %d workers",
        len(work_items),
        args.workers,
    )

    completed = 0
    total_errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _run_one_model_sync,
                session=session,
                model=model,
                framework_pair=fp,
                prompts=prompts,
                raw_dir=raw_dir,
            ): (fp, model)
            for fp, model, prompts in work_items
        }
        for fut in as_completed(futures):
            fp, model = futures[fut]
            try:
                stats = fut.result()
            except Exception as exc:
                log.error("[run-sync] worker %s / %s crashed: %s", fp, model, exc)
                total_errors += 1
                continue
            completed += 1
            errors = int(stats["errors"])
            total_errors += errors

            # Record cost (actual char counts → /4 token estimate).
            input_tokens = int(stats["total_input_chars"]) // 4
            output_tokens = int(stats["total_output_chars"]) // 4
            record_call(
                provider="bedrock_batch",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                batch_id=f"sync-{fp}-{awsbatch.model_slug(model)}",
            )
            log.info(
                "[run-sync] %s / %s done (%d candidates, %d errors) → %s",
                fp,
                model,
                stats["candidate_count"],
                errors,
                stats["raw_path"],
            )

    log.info(
        "[done] run-sync: completed=%d/%d total_errors=%d", completed, len(work_items), total_errors
    )
    return 0 if total_errors == 0 else 1


def cmd_poll(args: argparse.Namespace) -> int:
    session, _account_id = _make_session(args.profile)
    ledger_path = SMOKE_JOBS_LEDGER if args.smoke else JOBS_LEDGER
    raw_dir = args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ledger = read_ledger(ledger_path)
        if not ledger:
            log.info("[poll] no jobs in ledger %s", ledger_path)
            return 0

        pending = [e for e in ledger if e.status not in TERMINAL_STATUSES]
        if not pending:
            log.info("[poll] all %d job(s) terminal", len(ledger))
            return 0

        log.info("[poll] %d/%d non-terminal", len(pending), len(ledger))
        any_updated = False
        for entry in list(pending):
            resp = awsbatch.get_job_status(session, entry.job_arn)
            status = resp.get("status", "Unknown")
            if status == entry.status and status not in TERMINAL_STATUSES:
                log.info("  %s / %s: %s", entry.framework_pair, entry.model, status)
                continue

            log.info("  %s / %s: %s → %s", entry.framework_pair, entry.model, entry.status, status)
            entry_updated = entry.model_copy(update={"status": status})

            if status == "Completed":
                prompts = _load_prompts_cache(entry.framework_pair, smoke=entry.smoke)
                if not prompts:
                    log.warning(
                        "  no cached prompts for %s — skipping parse "
                        "(re-submit lost the prompts cache)",
                        entry.framework_pair,
                    )
                else:
                    candidates, stats = awsbatch.download_and_parse_outputs(
                        session=session,
                        output_s3_uri=entry.output_s3_uri,
                        prompts=prompts,
                        model=entry.model,
                    )
                    slug = awsbatch.model_slug(entry.model)
                    raw_path = raw_dir / f"{entry.framework_pair}__{slug}.jsonl"
                    awsbatch.write_candidates_jsonl(raw_path, candidates)
                    log.info(
                        "  wrote %d candidate(s) to %s (parsed_ok=%d errors=%d missing=%d)",
                        len(candidates),
                        raw_path,
                        stats["parsed_ok"],
                        stats["parse_errors"],
                        stats["missing"],
                    )

                    # Cost record using actual tokens from Bedrock response if available.
                    in_tok = (
                        resp.get("inputTokenCount")
                        or sum(len(p.system) + len(p.user) for p in prompts) // 4
                    )
                    out_tok = resp.get("outputTokenCount") or stats["parsed_ok"] * 150
                    record_call(
                        provider="bedrock_batch",
                        model=entry.model,
                        input_tokens=int(in_tok),
                        output_tokens=int(out_tok),
                        batch_id=entry.job_arn,
                    )

                    entry_updated = entry_updated.model_copy(
                        update={
                            "completed_at": _now_iso(),
                            "parsed_ok": stats["parsed_ok"],
                            "parse_errors": stats["parse_errors"],
                        }
                    )

            ledger = [e for e in ledger if _ledger_key(e) != _ledger_key(entry)] + [entry_updated]
            write_ledger(ledger_path, ledger)
            any_updated = True

        # Re-check after this sweep.
        non_terminal_remaining = [e for e in ledger if e.status not in TERMINAL_STATUSES]
        if not non_terminal_remaining or args.once:
            if non_terminal_remaining:
                log.info(
                    "[poll] --once: %d job(s) still non-terminal, exiting",
                    len(non_terminal_remaining),
                )
            else:
                log.info("[poll] all jobs terminal")
            return 0

        log.info("[poll] sleeping %ds before next sweep…", args.poll_interval)
        time.sleep(args.poll_interval)
        if not any_updated:
            # Long-poll: avoid logging every minute if nothing changed.
            log.debug("[poll] no status changes in this sweep")


# ─────────────────────────────────────────────────────────────────────────────
# Path-2 subcommand — `run-paid`: paid direct API ensemble (Anthropic +
# OpenAI + Gemini + Together). Reuses build_prompts_for_pair /
# build_smoke_prompts from the Bedrock paths so prompt construction is
# identical across all strategies. See docs/7a_path.md §"Path 2".
# ─────────────────────────────────────────────────────────────────────────────


@validated
def parse_shard(shard_str: str | None) -> tuple[int, int] | None:
    """Parse `--shard k/N` into `(k, N)`. Returns None for unset.

    Raises `SystemExit` on malformed input so the CLI dies with a clear
    message before any prompts are built. `0 <= k < N` required.
    """
    if shard_str is None:
        return None
    try:
        k_str, n_str = shard_str.split("/", 1)
        k, n = int(k_str), int(n_str)
    except (ValueError, AttributeError) as exc:
        raise SystemExit(
            f"--shard must be 'k/N' (e.g. '0/2', '1/2'); got: {shard_str!r}"
        ) from exc
    if n <= 0 or k < 0 or k >= n:
        raise SystemExit(f"--shard k/N requires 0 <= k < N (got {k}/{n})")
    return (k, n)


def cmd_run_paid(args: argparse.Namespace) -> int:
    """Path-2 entry: PaidAPIStrategy across the 4-seat 2025+ ensemble.

    Per-call JSONL append + resume-by-source_id contract from
    `daccord.ensemble.strategy` means crashes lose at most one in-flight
    call. Re-invoking the same command picks up exactly where the prior
    run left off.

    `--shard k/N` enables data-parallel runs: shard k processes pairs[k::N]
    only. Run N shards in parallel (separate terminals / containers) to
    multiply throughput. Per-provider RPM caps are the limiting factor:
    Anthropic Tier 1 50 RPM × ~3.3s/call = 18 RPM sequential, so safe N=2
    (36 RPM total) or borderline N=3 (54 RPM with 429s auto-retried via
    --retry-errors). Other seats have much higher headroom.
    """
    from daccord.ensemble.strategies.paid_api import PaidAPIStrategy

    registry_dir = args.registry_dir
    clauses_dir = args.clauses_dir
    raw_dir = args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    pairs = _resolve_pair_set(args, registry_dir)
    shard = parse_shard(args.shard)
    if shard is not None:
        k, n = shard
        total = len(pairs)
        pairs = pairs[k::n]
        log.info(
            "[run-paid] shard %d/%d: processing %d of %d total pairs",
            k,
            n,
            len(pairs),
            total,
        )

    # Build prompts per pair (same path as Bedrock).
    prompts_by_pair: dict[str, list[awsbatch.BatchPrompt]] = {}
    for framework_pair in pairs:
        if args.smoke:
            prompts = build_smoke_prompts(
                toy_gold_path=args.toy_gold,
                registry_dir=registry_dir,
                max_tokens=args.max_tokens,
            )
        else:
            prompts = build_prompts_for_pair(
                framework_pair=framework_pair,
                registry_dir=registry_dir,
                clauses_dir=clauses_dir,
                max_tokens=args.max_tokens,
                max_clauses=args.max_clauses_per_pair,
            )
        if prompts:
            prompts_by_pair[framework_pair] = prompts

    if not prompts_by_pair:
        log.warning("[run-paid] no prompts to dispatch")
        return 0

    # Dry-run path skips strategy construction — the 4 ModelClient
    # __init__s assert their API keys, which a dry-run shouldn't require.
    # We just compute the workload size against the default 4 seats.
    if args.dry_run:
        for framework_pair, prompts in prompts_by_pair.items():
            log.info(
                "[run-paid][dry-run] %s: %d prompts × 4 seats = %d invocations",
                framework_pair,
                len(prompts),
                len(prompts) * 4,
            )
        return 0

    # Strategy construction — uses the default 4-seat lineup unless the
    # operator overrode env vars. Each client lazy-imports its SDK so we
    # only crash on missing API keys when an actual call is attempted.
    try:
        strategy = PaidAPIStrategy()
    except Exception as exc:
        log.error("[run-paid] strategy construction failed: %s", exc)
        log.error(
            "[run-paid] hint: set ANTHROPIC_API_KEY / OPENAI_API_KEY / "
            "TOGETHER_API_KEY / GOOGLE_API_KEY"
        )
        return 3

    log.info(
        "[run-paid] %d pair(s) × %d seat(s): %s",
        len(prompts_by_pair),
        len(strategy.models),
        ", ".join(strategy.models),
    )

    total_processed = 0
    total_errors = 0
    total_resumed = 0
    for framework_pair, prompts in prompts_by_pair.items():
        log.info(
            "[run-paid] %s: dispatching %d prompts across %d seats%s",
            framework_pair,
            len(prompts),
            len(strategy.models),
            " (retrying parse_errors)" if args.retry_errors else "",
        )
        results = strategy.run_pair(
            framework_pair,
            prompts,
            raw_dir,
            smoke=args.smoke,
            retry_errors=args.retry_errors,
        )
        for model, rr in results.items():
            log.info(
                "[run-paid]   %s: processed=%d ok=%d errors=%d resumed=%d (%.1fs)",
                model,
                rr.total_processed,
                rr.parse_ok,
                rr.parse_errors,
                rr.resumed_from_disk,
                rr.seconds_elapsed,
            )
            total_processed += rr.total_processed
            total_errors += rr.parse_errors
            total_resumed += rr.resumed_from_disk

    log.info(
        "[run-paid] done: processed=%d errors=%d resumed=%d",
        total_processed,
        total_errors,
        total_resumed,
    )
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Status subcommand.
# ─────────────────────────────────────────────────────────────────────────────


def cmd_status(args: argparse.Namespace) -> int:
    ledger_path = SMOKE_JOBS_LEDGER if args.smoke else JOBS_LEDGER
    ledger = read_ledger(ledger_path)
    if not ledger:
        log.info("ledger %s is empty", ledger_path)
        return 0
    for e in sorted(ledger, key=lambda r: (r.framework_pair, r.model)):
        log.info(
            "%-30s %-50s %-12s prompts=%d parsed_ok=%s",
            e.framework_pair,
            awsbatch.model_slug(e.model),
            e.status,
            e.prompt_count,
            e.parsed_ok if e.parsed_ok is not None else "?",
        )
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry.
# ─────────────────────────────────────────────────────────────────────────────


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--smoke",
        action="store_true",
        help="20-invocation smoke test using toy_v1 GDPR sources → PDPA-SG.",
    )
    p.add_argument(
        "--profile",
        type=str,
        default=None,
        help="AWS profile name (default: env > aws-account.yaml > caravan-poc)",
    )
    p.add_argument("--verbose", action="store_true")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    subparsers = p.add_subparsers(dest="cmd", required=True)

    p_submit = subparsers.add_parser("submit", help="Submit Bedrock batch jobs.")
    _add_common_args(p_submit)
    p_submit.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    p_submit.add_argument("--clauses-dir", type=Path, default=DEFAULT_CLAUSES_DIR)
    p_submit.add_argument("--toy-gold", type=Path, default=DEFAULT_TOY_GOLD)
    p_submit.add_argument(
        "--framework-pair",
        type=str,
        default=None,
        help="Restrict to one pair (e.g. gdpr__pdpa_sg).",
    )
    p_submit.add_argument(
        "--max-clauses-per-pair",
        type=int,
        default=None,
        help="Cap source clauses per pair (cost-aware staging).",
    )
    p_submit.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p_submit.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts + run preflight but don't actually submit.",
    )
    p_submit.set_defaults(func=cmd_submit)

    p_poll = subparsers.add_parser("poll", help="Poll + download completed batch jobs.")
    _add_common_args(p_poll)
    p_poll.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p_poll.add_argument("--poll-interval", type=int, default=POLL_INTERVAL_SECONDS)
    p_poll.add_argument(
        "--once", action="store_true", help="Single sweep instead of sleep-and-retry loop."
    )
    p_poll.set_defaults(func=cmd_poll)

    p_status = subparsers.add_parser("status", help="Print job ledger.")
    _add_common_args(p_status)
    p_status.set_defaults(func=cmd_status)

    p_sync = subparsers.add_parser(
        "run-sync",
        help="Synchronous InvokeModel path (4 workers, one per model). "
        "2× batch cost but works without Bedrock-batch account enablement.",
    )
    _add_common_args(p_sync)
    p_sync.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    p_sync.add_argument("--clauses-dir", type=Path, default=DEFAULT_CLAUSES_DIR)
    p_sync.add_argument("--toy-gold", type=Path, default=DEFAULT_TOY_GOLD)
    p_sync.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p_sync.add_argument(
        "--framework-pair",
        type=str,
        default=None,
        help="Restrict to one pair (e.g. gdpr__pdpa_sg).",
    )
    p_sync.add_argument("--max-clauses-per-pair", type=int, default=None)
    p_sync.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p_sync.add_argument(
        "--workers",
        type=int,
        default=len(m2.F9_BEDROCK_MODELS),
        help="Thread pool size. Default = 4 (one per F9 model).",
    )
    p_sync.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts + preflight but don't actually invoke models.",
    )
    p_sync.set_defaults(func=cmd_run_sync)

    p_paid = subparsers.add_parser(
        "run-paid",
        help="Path 2 — paid direct API ensemble (Anthropic Haiku 4.5 + "
        "GPT-5-mini + Gemini 3.1 Flash Lite + Together Llama 4 Maverick). "
        "Per-call resumable; see docs/7a_path.md.",
    )
    _add_common_args(p_paid)
    p_paid.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    p_paid.add_argument("--clauses-dir", type=Path, default=DEFAULT_CLAUSES_DIR)
    p_paid.add_argument("--toy-gold", type=Path, default=DEFAULT_TOY_GOLD)
    p_paid.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p_paid.add_argument(
        "--framework-pair",
        type=str,
        default=None,
        help="Restrict to one pair (e.g. gdpr__pdpa_sg).",
    )
    p_paid.add_argument(
        "--max-clauses-per-pair",
        type=int,
        default=None,
        help="Cap source clauses per pair (cost-aware staging).",
    )
    p_paid.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p_paid.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts but don't actually invoke models.",
    )
    p_paid.add_argument(
        "--retry-errors",
        action="store_true",
        help=(
            "Scrub parse_error rows from each seat's output JSONL before "
            "resume so previously-failed (seat, source_id) pairs get "
            "re-called. Successful rows are untouched — no work duplication."
        ),
    )
    p_paid.add_argument(
        "--shard",
        type=str,
        default=None,
        metavar="k/N",
        help=(
            "Data-parallel run: this shard processes pairs[k::N] (sorted, "
            "stable). Launch N shards in parallel terminals to ~N-x throughput. "
            "Safe N depends on Anthropic Tier-1 50 RPM ceiling: N=2 is safe, "
            "N=3 expects some 429s (recover via --retry-errors)."
        ),
    )
    p_paid.set_defaults(func=cmd_run_paid)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

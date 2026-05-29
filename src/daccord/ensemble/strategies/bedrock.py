"""Path 1 — Bedrock batch + Bedrock sync `EnsembleStrategy` implementations.

Wraps the existing `daccord.aws.batch` boto3 code behind the
`EnsembleStrategy` Protocol from `daccord.ensemble.strategy`. The wrappers
add nothing semantic — they exist so `scripts/run_ensemble.py` can pick
strategies by `--strategy` flag and tier-6B sees identical
`data/ensemble/raw/*.jsonl` regardless of backend.

See [docs/7a_path.md] §"Path 1" for the lineup + cost + wall-clock.

Two strategies live here because Bedrock has two ways to run the same
job and we want both behind the same Protocol:

  - **`BedrockBatchStrategy`** — `bedrock:CreateModelInvocationJob`
    + S3 staging. Cloud-side overnight (1–24 h SLA). 50% off OD pricing.
    The original tier-7A design.
  - **`BedrockSyncStrategy`** — `bedrock-runtime:InvokeModel` via
    ThreadPoolExecutor. Used when batch isn't enabled on the account
    or when an operator wants immediate feedback. 2× the cost of batch.

Account-side state (boto3 Session, AWS account ID) is captured at
construction time so `run_pair` has the same call surface as Path 2's
local-API strategy.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from daccord.aws import batch as awsbatch
from daccord.aws import m2
from daccord.ensemble.prompt import BatchPrompt
from daccord.ensemble.strategy import (
    RunResult,
    output_path_for,
    write_candidates_atomic,
)
from daccord.validation import validated

log = logging.getLogger("daccord.ensemble.strategies.bedrock")


class BedrockBatchStrategy:
    """`bedrock:CreateModelInvocationJob` + S3 path (Path 1, batch flavor).

    `run_pair` is a blocking call here: submit → poll until terminal →
    download → parse → write JSONL. For long-overnight Bedrock batches
    the operator typically uses `scripts/run_ensemble.py submit` + a
    later `poll` instead of this all-in-one wrapper — but the wrapper
    exists so a single CLI command (`run --strategy bedrock-batch`)
    works for smoke tests + the simple case.
    """

    name = "bedrock-batch"

    @validated
    def __init__(
        self,
        session: Any,
        account_id: str,
        *,
        models: list[str] | None = None,
        poll_interval_seconds: int = 60,
    ) -> None:
        self.session = session
        self.account_id = account_id
        self.models = models or list(m2.F9_BEDROCK_MODELS)
        self.poll_interval_seconds = poll_interval_seconds

    def run_pair(
        self,
        framework_pair: str,
        prompts: list[BatchPrompt],
        out_dir: Path,
        *,
        smoke: bool,
    ) -> dict[str, RunResult]:
        results: dict[str, RunResult] = {}
        for model in self.models:
            results[model] = self._run_one_model(
                framework_pair, prompts, out_dir, model, smoke=smoke
            )
        return results

    def _run_one_model(
        self,
        framework_pair: str,
        prompts: list[BatchPrompt],
        out_dir: Path,
        model: str,
        *,
        smoke: bool,
    ) -> RunResult:
        start = time.monotonic()
        output_path = output_path_for(out_dir, framework_pair, model)

        # Submit
        jsonl_content = awsbatch.build_batch_jsonl(prompts, model=model)
        job_arn = awsbatch.submit_job(
            session=self.session,
            account_id=self.account_id,
            framework_pair=framework_pair,
            model=model,
            jsonl_content=jsonl_content,
            smoke=smoke,
        )
        log.info(
            "[bedrock-batch] %s / %s submitted job_arn=%s",
            framework_pair,
            model,
            job_arn,
        )

        # Poll until terminal.
        while True:
            status_response = awsbatch.get_job_status(self.session, job_arn)
            status = status_response.get("status", "Unknown")
            log.info(
                "[bedrock-batch] %s / %s job status=%s",
                framework_pair,
                model,
                status,
            )
            if status in {"Completed", "PartiallyCompleted"}:
                break
            if status in {"Failed", "Stopped", "Expired"}:
                log.error(
                    "[bedrock-batch] %s / %s job ended with status=%s",
                    framework_pair,
                    model,
                    status,
                )
                # Write empty result file so resume doesn't re-submit.
                write_candidates_atomic(output_path, [])
                return RunResult(
                    framework_pair=framework_pair,
                    model=model,
                    output_path=str(output_path),
                    total_processed=0,
                    parse_ok=0,
                    parse_errors=0,
                    resumed_from_disk=0,
                    seconds_elapsed=time.monotonic() - start,
                )
            time.sleep(self.poll_interval_seconds)

        # Download + parse + write
        _, out_uri = awsbatch.s3_prefix_for(self.account_id, framework_pair, model, smoke=smoke)
        candidates, stats = awsbatch.download_and_parse_outputs(
            session=self.session,
            output_s3_uri=out_uri,
            prompts=prompts,
            model=model,
        )
        write_candidates_atomic(output_path, candidates)
        log.info(
            "[bedrock-batch] %s / %s parsed_ok=%d parse_errors=%d missing=%d",
            framework_pair,
            model,
            stats["parsed_ok"],
            stats["parse_errors"],
            stats["missing"],
        )
        return RunResult(
            framework_pair=framework_pair,
            model=model,
            output_path=str(output_path),
            total_processed=int(stats["parsed_ok"]) + int(stats["parse_errors"]),
            parse_ok=int(stats["parsed_ok"]),
            parse_errors=int(stats["parse_errors"]),
            resumed_from_disk=0,
            seconds_elapsed=time.monotonic() - start,
        )


class BedrockSyncStrategy:
    """`bedrock-runtime:InvokeModel` synchronous path (Path 1, sync flavor).

    ThreadPoolExecutor with one worker per model. Per-prompt invocation
    via `awsbatch.invoke_sync` which has its own retry layer for
    `ThrottlingException` / `ModelTimeoutException` / etc.

    Unlike batch, sync writes can be made resilient via per-call append.
    This implementation uses `awsbatch.candidate_from_invoke_response`
    + the atomic-write pattern at the end. To match the Path-2 resume
    contract, a future iteration can append-per-call; today the existing
    invoke_sync semantics (5 retries + exponential backoff) make
    per-prompt loss vanishingly unlikely.
    """

    name = "bedrock-sync"

    @validated
    def __init__(
        self,
        session: Any,
        account_id: str,
        *,
        models: list[str] | None = None,
        workers: int = 4,
    ) -> None:
        self.session = session
        self.account_id = account_id
        self.models = models or list(m2.F9_BEDROCK_MODELS)
        self.workers = workers

    def run_pair(
        self,
        framework_pair: str,
        prompts: list[BatchPrompt],
        out_dir: Path,
        *,
        smoke: bool,
    ) -> dict[str, RunResult]:
        _ = smoke  # sync path doesn't need to distinguish (no S3 prefixing)
        results: dict[str, RunResult] = {}
        # One worker per model — same pattern as the original cmd_run_sync.
        with ThreadPoolExecutor(max_workers=min(self.workers, len(self.models))) as pool:
            futures = {
                pool.submit(self._run_one_model, framework_pair, prompts, out_dir, model): model
                for model in self.models
            }
            for fut in as_completed(futures):
                model = futures[fut]
                try:
                    results[model] = fut.result()
                except Exception as exc:
                    log.error(
                        "[bedrock-sync] worker %s / %s crashed: %s",
                        framework_pair,
                        model,
                        exc,
                    )
        return results

    def _run_one_model(
        self,
        framework_pair: str,
        prompts: list[BatchPrompt],
        out_dir: Path,
        model: str,
    ) -> RunResult:
        start = time.monotonic()
        output_path = output_path_for(out_dir, framework_pair, model)
        candidates: list[awsbatch.EnsembleCandidate] = []
        errors = 0
        for prompt in prompts:
            try:
                response_body = awsbatch.invoke_sync(
                    session=self.session,
                    model=model,
                    system=prompt.system,
                    user=prompt.user,
                    max_tokens=prompt.max_tokens,
                )
                cand = awsbatch.candidate_from_invoke_response(
                    prompt=prompt, response_body=response_body, model=model
                )
                candidates.append(cand)
                if cand.parse_error is not None:
                    errors += 1
            except Exception as exc:
                log.error(
                    "[bedrock-sync] %s / %s record=%s failed: %s",
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

        write_candidates_atomic(output_path, candidates)
        return RunResult(
            framework_pair=framework_pair,
            model=model,
            output_path=str(output_path),
            total_processed=len(candidates),
            parse_ok=len(candidates) - errors,
            parse_errors=errors,
            resumed_from_disk=0,
            seconds_elapsed=time.monotonic() - start,
        )

"""MLflow tracking plumbing for D'accord.

Phase 1 uses a local file backend (`file:./mlruns`) gitignored in [.gitignore].
Phase 2 will flip `MLFLOW_TRACKING_URI` to a remote server; nothing else changes.

See [docs/MLFLOW.md] for conventions and the tier-10A wiring snippet.
"""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
from pathlib import Path

import mlflow

from daccord.validation import validated

DEFAULT_TRACKING_URI = "file:./mlruns"
DEFAULT_EXPERIMENT_NAME = "daccord-qlora"
PROJECT_TAG_KEY = "project"
PROJECT_TAG_VALUE = "d-accord"


@validated
def setup_mlflow(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: str | None = None,
) -> None:
    """Set the MLflow tracking URI + experiment for the current process.

    Resolution order for the tracking URI:
        1. Explicit `tracking_uri` arg.
        2. `MLFLOW_TRACKING_URI` env var.
        3. `file:./mlruns` (Phase 1 default; gitignored).
    """
    resolved_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(resolved_uri)
    mlflow.set_experiment(experiment_name)


@validated
def get_git_commit(short: int = 12) -> str:
    """Return the current git HEAD commit SHA (short by default).

    Returns `"unknown"` if git is unavailable (e.g. SageMaker training container
    without a `.git` directory) so logging never blocks on environment quirks.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", f"--short={short}", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except Exception:
        # SubprocessError + FileNotFoundError + OSError all subclass Exception;
        # this codepath logs "unknown" rather than crashing the caller when
        # git is unavailable (SageMaker container, no .git, etc).
        return "unknown"
    return result.stdout.strip() or "unknown"


@validated
def compute_file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream `path` in `chunk_size` chunks and return the hex SHA256.

    Used by `log_adapter_sha256` and externally for the gold-set dataset hash.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


@validated
def log_standard_params(
    run_name: str,
    seed: int,
    dataset_hash: str | None = None,
    extra: dict[str, str] | None = None,
) -> None:
    """Log the project's reproducibility-contract params under the active run.

    Must be called inside an active `mlflow.start_run()` context. Logs:
    `run_name`, `git_commit`, `seed`, `dataset_hash` (if not None), and any
    additional string params in `extra`. Also tags the run `project=d-accord`
    so cost-attribution stays aligned with the AWS resource-tag convention.
    """
    mlflow.set_tag(PROJECT_TAG_KEY, PROJECT_TAG_VALUE)
    params: dict[str, str | int] = {
        "run_name": run_name,
        "git_commit": get_git_commit(),
        "seed": seed,
    }
    if dataset_hash is not None:
        params["dataset_hash"] = dataset_hash
    if extra is not None:
        params.update(extra)
    mlflow.log_params(params)


@validated
def log_adapter_sha256(adapter_path: Path) -> str:
    """Hash the saved adapter file, log it as param `adapter_sha256`, return it.

    Call at end of training, after `trainer.save_model(...)`. Closes M3's
    "adapter SHA logged" requirement and M4's adapter/dataset-hash linkage.
    """
    digest = compute_file_sha256(adapter_path)
    mlflow.log_param("adapter_sha256", digest)
    return digest


@validated
def set_all_seeds(seed: int) -> None:
    """Seed every RNG the training pipeline can touch.

    `torch` and `transformers` are lazy-imported so this module is usable from
    the toy/smoke run before the training-deps stack lands (tier 10A).
    Missing optional imports are silently skipped — by design.
    """
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch  # pyright: ignore[reportMissingImports]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        import transformers  # pyright: ignore[reportMissingImports]

        transformers.set_seed(seed)
    except ImportError:
        pass

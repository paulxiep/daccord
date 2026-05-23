"""Smoke test for the 1B MLflow plumbing.

Emits one dummy run to the configured tracking backend, no model needed.
Verifies that `setup_mlflow`, `log_standard_params`, and the env-var flow all
work end-to-end. Pass criterion: after running, `mlflow ui` shows the run
with `git_commit`, `seed=42`, `phase=1B-plumbing` and metric `dummy_loss`.

Run: `uv run python scripts/mlflow_smoke_test.py`
"""

from __future__ import annotations

import mlflow

from daccord.tracking import (
    log_standard_params,
    set_all_seeds,
    setup_mlflow,
)


def main() -> None:
    setup_mlflow(experiment_name="daccord-smoke")
    set_all_seeds(42)
    with mlflow.start_run(run_name="smoke-1B"):
        log_standard_params(
            run_name="smoke-1B",
            seed=42,
            dataset_hash=None,
            extra={"phase": "1B-plumbing"},
        )
        mlflow.log_metric("dummy_loss", 0.5, step=0)
    print(
        "Run logged. View with:\n"
        "  uv run mlflow ui --backend-store-uri file:./mlruns\n"
        "Then open http://127.0.0.1:5000 -> experiment 'daccord-smoke'."
    )


if __name__ == "__main__":
    main()

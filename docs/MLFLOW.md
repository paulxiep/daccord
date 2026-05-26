# MLflow conventions

Tier-1B deliverable. Wires MLflow once so every run from the toy training
forward — including failed/aborted ones — is tracked without retrofitting.
The `daccord.tracking` module is the single touch-point for training, eval,
and publishing scripts.

## Backend

- **Phase 1**: local file store at `file:./mlruns` (gitignored in [.gitignore](../.gitignore)).
- **Phase 2 / cloud**: change `MLFLOW_TRACKING_URI` in `.env.local` to a remote tracking server (e.g. AWS-managed MLflow). No code edits needed.

Resolution order in [setup_mlflow](../src/daccord/tracking.py): explicit arg → `MLFLOW_TRACKING_URI` env → `file:./mlruns` default.

## Standard params logged (project reproducibility contract)

| Param | Source | Logged by |
|---|---|---|
| `run_name` | call site | `log_standard_params` |
| `git_commit` | `git rev-parse --short=12 HEAD` | `log_standard_params` (auto) |
| `seed` | call site | `log_standard_params` |
| `dataset_hash` | SHA256 of gold JSONL | `log_standard_params` (None-safe; populated from M2 onward) |
| `adapter_sha256` | SHA256 of saved adapter | `log_adapter_sha256` at end of training |
| HF hyperparams / metrics | `TrainingArguments` + `Trainer` | HF's MLflow callback via `report_to=["mlflow"]` |

Tag `project=daccord` is set on every run, mirroring the AWS resource-tag convention from [development_plan.md §6](development_plan.md).

## Tier-10A wiring snippet

```python
from pathlib import Path

import mlflow
from transformers import Trainer, TrainingArguments

from daccord.tracking import (
    log_adapter_sha256,
    log_standard_params,
    set_all_seeds,
    setup_mlflow,
)

setup_mlflow()
set_all_seeds(42)
with mlflow.start_run(run_name="qlora-toy-200"):
    log_standard_params("qlora-toy-200", seed=42, dataset_hash=GOLD_SHA)
    trainer = Trainer(
        ...,
        args=TrainingArguments(
            ...,
            report_to=["mlflow"],  # HF callback auto-logs params + metrics
            run_name="qlora-toy-200",
        ),
    )
    trainer.train()
    trainer.save_model("training/runs/qlora-toy-200/adapter")
    log_adapter_sha256(Path("training/runs/qlora-toy-200/adapter/adapter_model.safetensors"))
```

We deliberately do **not** call `mlflow.autolog()` or `mlflow.transformers.autolog()` globally. HF's `Trainer` MLflow callback covers the training params/metrics; `mlflow.transformers.autolog()` would also dump the model weights (50MB+ per run) into MLflow's artifact store, which we don't want — the adapter lives on disk and is referenced by SHA.

## Viewing runs

```bash
uv run mlflow ui --backend-store-uri file:./mlruns
# then open http://127.0.0.1:5000
```

## Phase 2 transition

When SageMaker hosting (Phase 2) is triggered, optionally switch to a remote tracking server:

```bash
# .env.local (gitignored)
MLFLOW_TRACKING_URI=https://your-mlflow-server.example
```

No code change. The runs already in `./mlruns/` can be migrated via `mlflow runs migrate` or simply left as the local Phase-1 record.

## `.env.example` convention

`.env.example` is a multi-tier shared file:
- **1B owns**: `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`
- **1C will append**: per-provider daily caps (`ANTHROPIC_DAILY_CAP_USD`, `OPENAI_DAILY_CAP_USD`, `TOGETHER_DAILY_CAP_USD`) and API keys

Parallel tier-1 tasks append rather than rewrite; the file is small and merge-friendly.

## Smoke test

`scripts/mlflow_smoke_test.py` emits one dummy run end-to-end (no model). Run it once after `uv sync` to verify the plumbing:

```bash
uv run python scripts/mlflow_smoke_test.py
uv run mlflow ui --backend-store-uri file:./mlruns
# experiment "daccord-smoke" should show one run "smoke-1B"
```

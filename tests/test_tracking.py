"""Unit tests for daccord.tracking — the 1B MLflow plumbing.

End-to-end UI verification is left to scripts/mlflow_smoke_test.py; these tests
cover the pure-Python helpers and the param-logging contract.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import mlflow
import pytest
from pydantic import ValidationError

from daccord.tracking import (
    DEFAULT_EXPERIMENT_NAME,
    PROJECT_TAG_KEY,
    PROJECT_TAG_VALUE,
    compute_file_sha256,
    get_git_commit,
    log_adapter_sha256,
    log_standard_params,
    set_all_seeds,
    setup_mlflow,
)


@pytest.fixture
def isolated_tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point MLflow at a per-test tmpdir so runs don't leak into ./mlruns."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    uri = f"file:{tmp_path.as_posix()}"
    setup_mlflow(experiment_name="test-experiment", tracking_uri=uri)
    return tmp_path


def test_setup_mlflow_uses_explicit_uri(isolated_tracking: Path) -> None:
    assert mlflow.get_tracking_uri() == f"file:{isolated_tracking.as_posix()}"
    assert mlflow.get_experiment_by_name("test-experiment") is not None


def test_setup_mlflow_falls_back_to_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path.as_posix()}")
    setup_mlflow(experiment_name="env-experiment")
    assert mlflow.get_tracking_uri() == f"file:{tmp_path.as_posix()}"


def test_get_git_commit_returns_hex_or_unknown() -> None:
    commit = get_git_commit()
    assert commit == "unknown" or re.fullmatch(r"[0-9a-f]{12}", commit)


def test_get_git_commit_respects_short_arg() -> None:
    commit = get_git_commit(short=7)
    assert commit == "unknown" or re.fullmatch(r"[0-9a-f]{7}", commit)


def test_compute_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    payload = b"d-accord reproducibility contract" * 1000
    f = tmp_path / "adapter.bin"
    f.write_bytes(payload)
    assert compute_file_sha256(f) == hashlib.sha256(payload).hexdigest()


def test_compute_file_sha256_handles_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "empty.bin"
    f.write_bytes(b"")
    assert compute_file_sha256(f) == hashlib.sha256(b"").hexdigest()


def test_log_standard_params_writes_expected_keys(isolated_tracking: Path) -> None:
    with mlflow.start_run(run_name="unit-1") as run:
        log_standard_params(
            run_name="unit-1",
            seed=7,
            dataset_hash="abc123",
            extra={"phase": "test"},
        )

    fetched = mlflow.get_run(run.info.run_id)
    assert fetched.data.params["run_name"] == "unit-1"
    assert fetched.data.params["seed"] == "7"
    assert fetched.data.params["dataset_hash"] == "abc123"
    assert fetched.data.params["phase"] == "test"
    assert "git_commit" in fetched.data.params
    assert fetched.data.tags[PROJECT_TAG_KEY] == PROJECT_TAG_VALUE


def test_log_standard_params_omits_dataset_hash_when_none(
    isolated_tracking: Path,
) -> None:
    with mlflow.start_run(run_name="unit-2") as run:
        log_standard_params(run_name="unit-2", seed=1)

    fetched = mlflow.get_run(run.info.run_id)
    assert "dataset_hash" not in fetched.data.params


def test_log_adapter_sha256_logs_and_returns_digest(
    isolated_tracking: Path, tmp_path: Path
) -> None:
    payload = b"fake-adapter-bytes"
    adapter = tmp_path / "adapter.safetensors"
    adapter.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    with mlflow.start_run(run_name="unit-3") as run:
        returned = log_adapter_sha256(adapter)

    assert returned == expected
    fetched = mlflow.get_run(run.info.run_id)
    assert fetched.data.params["adapter_sha256"] == expected


def test_set_all_seeds_is_deterministic_for_stdlib_random() -> None:
    import random as stdlib_random

    set_all_seeds(123)
    a = [stdlib_random.random() for _ in range(5)]
    set_all_seeds(123)
    b = [stdlib_random.random() for _ in range(5)]
    assert a == b


def test_validated_rejects_wrong_arg_types() -> None:
    with pytest.raises(ValidationError):
        log_standard_params(run_name=123, seed=1)  # type: ignore[arg-type]


def test_default_experiment_name_constant() -> None:
    assert DEFAULT_EXPERIMENT_NAME == "daccord-qlora"

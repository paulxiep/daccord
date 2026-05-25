from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from daccord.costs.config import Provider, inflight_path
from daccord.validation import ValidatedModel, validated

_SCHEMA = """
CREATE TABLE IF NOT EXISTS inflight (
  ts_utc        TEXT    NOT NULL,
  provider      TEXT    NOT NULL,
  model         TEXT    NOT NULL,
  input_tokens  INTEGER NOT NULL,
  output_tokens INTEGER NOT NULL,
  cost_usd      REAL    NOT NULL,
  run_id        TEXT,
  batch_id      TEXT
);
CREATE INDEX IF NOT EXISTS idx_inflight_date_provider
  ON inflight(substr(ts_utc, 1, 10), provider);
"""

# WAL-mode switch needs exclusive access; without serialization, concurrent first-time
# openers race and one fails with "database is locked". Init each db path exactly once.
_init_lock = threading.Lock()
_initialized: set[str] = set()


def _ensure_initialized(target: Path) -> None:
    key = str(target.resolve())
    if key in _initialized:
        return
    with _init_lock:
        if key in _initialized:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(target, timeout=30, isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(_SCHEMA)
        finally:
            conn.close()
        _initialized.add(key)


class CallRow(ValidatedModel):
    ts_utc: str
    provider: Provider
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    run_id: str | None = None
    batch_id: str | None = None


class DailyRow(ValidatedModel):
    date: str
    provider: Provider
    model: str
    input_tokens: int
    output_tokens: int
    n_calls: int
    cost_usd: float


@contextmanager
def _connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    target = path if path is not None else inflight_path()
    _ensure_initialized(target)
    # timeout=30 lets concurrent writers wait out the busy window instead of erroring.
    conn = sqlite3.connect(target, timeout=30, isolation_level=None)
    try:
        yield conn
    finally:
        conn.close()


@validated
def append_call(row: CallRow, db_path: Path | None = None) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO inflight "
            "(ts_utc, provider, model, input_tokens, output_tokens, cost_usd, run_id, batch_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row.ts_utc,
                row.provider,
                row.model,
                row.input_tokens,
                row.output_tokens,
                row.cost_usd,
                row.run_id,
                row.batch_id,
            ),
        )


@validated
def sum_today(provider: Provider, db_path: Path | None = None) -> float:
    today = datetime.now(UTC).date().isoformat()
    with _connect(db_path) as conn:
        cur = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) FROM inflight "
            "WHERE provider = ? AND substr(ts_utc, 1, 10) = ?",
            (provider, today),
        )
        return float(cur.fetchone()[0])


@validated
def count_today(provider: Provider, db_path: Path | None = None) -> int:
    today = datetime.now(UTC).date().isoformat()
    with _connect(db_path) as conn:
        cur = conn.execute(
            "SELECT COUNT(*) FROM inflight WHERE provider = ? AND substr(ts_utc, 1, 10) = ?",
            (provider, today),
        )
        return int(cur.fetchone()[0])


@validated
def daily_rows(db_path: Path | None = None) -> list[DailyRow]:
    with _connect(db_path) as conn:
        cur = conn.execute(
            "SELECT substr(ts_utc, 1, 10) AS date, provider, model, "
            "       SUM(input_tokens), SUM(output_tokens), COUNT(*), "
            "       ROUND(SUM(cost_usd), 4) "
            "FROM inflight "
            "GROUP BY date, provider, model "
            "ORDER BY date, provider, model"
        )
        return [
            DailyRow(
                date=r[0],
                provider=r[1],
                model=r[2],
                input_tokens=int(r[3]),
                output_tokens=int(r[4]),
                n_calls=int(r[5]),
                cost_usd=float(r[6]),
            )
            for r in cur.fetchall()
        ]


@validated
def daily_provider_totals(db_path: Path | None = None) -> dict[str, dict[Provider, float]]:
    """{date_iso: {provider: cost_usd_sum}} aggregated across models. For R7 streak check."""
    with _connect(db_path) as conn:
        cur = conn.execute(
            "SELECT substr(ts_utc, 1, 10) AS date, provider, ROUND(SUM(cost_usd), 4) "
            "FROM inflight GROUP BY date, provider ORDER BY date, provider"
        )
        out: dict[str, dict[Provider, float]] = {}
        for date, provider, total in cur.fetchall():
            out.setdefault(date, {})[provider] = float(total)
        return out

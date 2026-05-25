import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from daccord.corpus.manifest import ManifestEntry
from daccord.corpus.sources import Source
from daccord.validation import ValidatedModel, validated

USER_AGENT = "d-accord-corpus/0.0.1 (regulatory corpus download; contact: paulxiep@outlook.com)"
RETRY_STATUSES = frozenset({500, 502, 503, 504, 522, 524})
MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 1.5


class DownloadResult(ValidatedModel):
    entry: ManifestEntry
    skipped_existing: bool


@validated
def expected_path(source: Source, raw_root: Path) -> Path:
    return raw_root / source.jurisdiction / source.framework / source.filename


@validated
def _sha256_of(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


@validated
def hash_existing(source: Source, raw_root: Path) -> ManifestEntry | None:
    path = expected_path(source, raw_root)
    if not path.exists():
        return None
    sha, size = _sha256_of(path)
    return ManifestEntry(
        framework=source.framework,
        jurisdiction=source.jurisdiction,
        filename=source.filename,
        source_url=str(source.url) if source.url else None,
        local_path=str(path.as_posix()),
        sha256=sha,
        content_length=size,
        retrieved_at=datetime.now(UTC),
        manual=source.manual,
    )


@validated
def fetch(source: Source, raw_root: Path, client: httpx.Client) -> DownloadResult:
    if source.manual or source.url is None:
        raise ValueError(f"fetch() called for manual source {source.framework}/{source.filename}")

    out_path = expected_path(source, raw_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    last_exc: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with client.stream("GET", str(source.url), follow_redirects=True) as resp:
                if resp.status_code in RETRY_STATUSES:
                    raise httpx.HTTPStatusError(
                        f"retryable status {resp.status_code}", request=resp.request, response=resp
                    )
                resp.raise_for_status()
                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                hasher = hashlib.sha256()
                size = 0
                with tmp_path.open("wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=1 << 16):
                        f.write(chunk)
                        hasher.update(chunk)
                        size += len(chunk)
                if size == 0:
                    tmp_path.unlink(missing_ok=True)
                    raise ValueError(
                        f"empty body from {source.url} "
                        f"(status {resp.status_code}; UA-blocked or content negotiation)"
                    )
                tmp_path.replace(out_path)
                entry = ManifestEntry(
                    framework=source.framework,
                    jurisdiction=source.jurisdiction,
                    filename=source.filename,
                    source_url=str(source.url),
                    local_path=str(out_path.as_posix()),
                    sha256=hasher.hexdigest(),
                    content_length=size,
                    retrieved_at=datetime.now(UTC),
                    manual=False,
                )
                return DownloadResult(entry=entry, skipped_existing=False)
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            last_exc = exc
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
                continue
            raise
    assert last_exc is not None
    raise last_exc


@validated
def make_client(timeout_seconds: float = 60.0) -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(timeout_seconds),
        headers={"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*;q=0.8"},
    )

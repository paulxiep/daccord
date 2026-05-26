import hashlib
from pathlib import Path

import httpx
import pytest

from daccord.corpus.downloader import expected_path, fetch, hash_existing, make_client
from daccord.corpus.sources import Source

PDF_BYTES = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\n%%EOF\n" * 1024


def _url_source(filename: str = "test.pdf") -> Source:
    return Source(
        framework="testfw",
        jurisdiction="xx",
        filename=filename,
        description="t",
        url="https://example.com/" + filename,  # type: ignore[arg-type]
    )


def _client_with(handler) -> httpx.Client:
    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport, follow_redirects=True)


def test_expected_path_structure(tmp_path: Path) -> None:
    s = _url_source("doc.pdf")
    assert expected_path(s, tmp_path) == tmp_path / "xx" / "testfw" / "doc.pdf"


def test_fetch_writes_file_and_records_hash(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=PDF_BYTES)

    client = _client_with(handler)
    result = fetch(_url_source(), tmp_path, client)
    assert result.entry.sha256 == hashlib.sha256(PDF_BYTES).hexdigest()
    assert result.entry.content_length == len(PDF_BYTES)
    out_path = tmp_path / "xx" / "testfw" / "test.pdf"
    assert out_path.exists()
    assert out_path.read_bytes() == PDF_BYTES


def test_fetch_retries_on_503_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("daccord.corpus.downloader.BACKOFF_BASE_SECONDS", 0.0)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, content=b"busy")
        return httpx.Response(200, content=PDF_BYTES)

    client = _client_with(handler)
    result = fetch(_url_source("retry.pdf"), tmp_path, client)
    assert calls["n"] == 3
    assert result.entry.content_length == len(PDF_BYTES)


def test_fetch_raises_after_max_attempts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("daccord.corpus.downloader.BACKOFF_BASE_SECONDS", 0.0)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"busy")

    client = _client_with(handler)
    with pytest.raises(httpx.HTTPStatusError):
        fetch(_url_source("fail.pdf"), tmp_path, client)
    assert not (tmp_path / "xx" / "testfw" / "fail.pdf").exists()


def test_fetch_raises_on_404(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"nope")

    client = _client_with(handler)
    with pytest.raises(httpx.HTTPStatusError):
        fetch(_url_source("missing.pdf"), tmp_path, client)


def test_fetch_raises_on_empty_200_body(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    client = _client_with(handler)
    with pytest.raises(ValueError, match="empty body"):
        fetch(_url_source("empty.pdf"), tmp_path, client)
    assert not (tmp_path / "xx" / "testfw" / "empty.pdf").exists()


def test_fetch_rejects_manual_source(tmp_path: Path) -> None:
    s = Source(
        framework="testfw",
        jurisdiction="xx",
        filename="m.pdf",
        description="manual",
        manual=True,
    )
    client = _client_with(lambda req: httpx.Response(200))
    with pytest.raises(ValueError, match="manual source"):
        fetch(s, tmp_path, client)


def test_hash_existing_returns_none_when_absent(tmp_path: Path) -> None:
    assert hash_existing(_url_source("absent.pdf"), tmp_path) is None


def test_hash_existing_returns_entry_for_present_file(tmp_path: Path) -> None:
    s = _url_source("present.pdf")
    out_path = expected_path(s, tmp_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(PDF_BYTES)
    entry = hash_existing(s, tmp_path)
    assert entry is not None
    assert entry.sha256 == hashlib.sha256(PDF_BYTES).hexdigest()
    assert entry.content_length == len(PDF_BYTES)


def test_make_client_sets_user_agent() -> None:
    with make_client(timeout_seconds=5.0) as c:
        assert "daccord-corpus" in c.headers["User-Agent"]

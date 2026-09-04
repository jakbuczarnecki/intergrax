"""Focused tests for HTTP proof-data transport resume and failure hardening."""

from __future__ import annotations

import hashlib
import socket
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import httpx
import pytest

from intergrax.proof_data.errors import DataPackageTransportError
from intergrax.proof_data.transport.http import (
    HttpDataPackageTransport,
    _reject_unsafe_redirect_scheme,
)

StreamScenario = Callable[[dict[str, str]], "_MockResponse"]


class _MockResponse:
    def __init__(
        self,
        status_code: int,
        *,
        headers: dict[str, str] | None = None,
        chunks: tuple[bytes, ...] = (),
        fail_after_chunks: int | None = None,
        request_url: str = "http://example.test/file",
    ) -> None:
        self.status_code = status_code
        self.headers = httpx.Headers(headers or {})
        self.url = request_url
        self._chunks = chunks
        self._fail_after_chunks = fail_after_chunks
        self._chunks_delivered = 0

    def iter_bytes(self, chunk_size: int) -> Iterator[bytes]:
        for chunk in self._chunks:
            yield chunk
            self._chunks_delivered += 1
            if (
                self._fail_after_chunks is not None
                and self._chunks_delivered >= self._fail_after_chunks
            ):
                raise httpx.ReadError(
                    "simulated mid-stream disconnect",
                    request=httpx.Request("GET", str(self.url)),
                )


class _MockStreamContext:
    def __init__(self, response: _MockResponse) -> None:
        self._response = response

    def __enter__(self) -> _MockResponse:
        return self._response

    def __exit__(self, *args: object) -> None:
        return None


def _install_stream_mock(
    monkeypatch: pytest.MonkeyPatch,
    scenarios: list[StreamScenario],
) -> list[dict[str, str]]:
    requested_headers: list[dict[str, str]] = []
    call_index = 0

    class _MockClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            hooks = kwargs.get("event_hooks")
            self._response_hooks = (
                hooks.get("response", []) if isinstance(hooks, dict) else []
            )

        def __enter__(self) -> _MockClient:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def stream(
            self,
            method: str,
            url: str,
            headers: dict[str, str] | None = None,
        ) -> _MockStreamContext:
            nonlocal call_index
            normalized_headers = dict(headers or {})
            requested_headers.append(normalized_headers)
            if call_index >= len(scenarios):
                raise AssertionError(f"unexpected HTTP attempt {call_index + 1}")
            response = scenarios[call_index](normalized_headers)
            response.url = url
            for hook in self._response_hooks:
                hook(response)
            call_index += 1
            return _MockStreamContext(response)

    monkeypatch.setattr(httpx, "Client", _MockClient)
    return requested_headers


def _range_offset(headers: dict[str, str]) -> int:
    range_header = headers.get("Range", "")
    if not range_header.startswith("bytes="):
        return 0
    return int(range_header.split("=")[1].split("-")[0])


def _range_response(
    payload: bytes,
    headers: dict[str, str],
    *,
    write_bytes: int | None = None,
    fail_after_chunk: bool = False,
) -> _MockResponse:
    offset = _range_offset(headers)
    body = payload[offset:]
    if write_bytes is not None:
        body = body[:write_bytes]
    return _MockResponse(
        206,
        headers={
            "Content-Range": f"bytes {offset}-{len(payload) - 1}/{len(payload)}",
        },
        chunks=(body,),
        fail_after_chunks=1 if fail_after_chunk else None,
    )


def _full_response(
    payload: bytes,
    *,
    write_bytes: int | None = None,
    fail_after_chunk: bool = False,
) -> _MockResponse:
    body = payload if write_bytes is None else payload[:write_bytes]
    return _MockResponse(
        200,
        chunks=(body,),
        fail_after_chunks=1 if fail_after_chunk else None,
    )


def _immediate_read_error(
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    url: str = "http://example.test/file",
) -> _MockResponse:
    response = _MockResponse(status_code, headers=headers, chunks=())
    response.url = url

    def _iter_bytes(chunk_size: int) -> Iterator[bytes]:
        raise httpx.ReadError(
            "simulated immediate disconnect",
            request=httpx.Request("GET", url),
        )
        yield b""

    response.iter_bytes = _iter_bytes  # type: ignore[method-assign]
    return response


class _ControllableHandler(BaseHTTPRequestHandler):
    payload: ClassVar[bytes] = b""
    requested_ranges: ClassVar[list[str | None]] = []
    attempt_rules: ClassVar[list[dict[str, object]]] = []
    _attempt_counter: ClassVar[int] = 0
    lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def reset(cls, payload: bytes, rules: list[dict[str, object]]) -> None:
        with cls.lock:
            cls.payload = payload
            cls.requested_ranges = []
            cls.attempt_rules = list(rules)
            cls._attempt_counter = 0

    def do_GET(self) -> None:
        with self.lock:
            attempt_index = _ControllableHandler._attempt_counter
            _ControllableHandler._attempt_counter += 1
            rule = (
                _ControllableHandler.attempt_rules[attempt_index]
                if attempt_index < len(_ControllableHandler.attempt_rules)
                else {}
            )

        range_header = self.headers.get("Range")
        with self.lock:
            self.requested_ranges.append(range_header)

        if rule.get("status") == 416:
            self.send_response(416)
            self.end_headers()
            return

        if rule.get("ignore_range"):
            body = self.payload
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if rule.get("bad_content_range"):
            start = 0
            if range_header and range_header.startswith("bytes="):
                start = int(range_header.split("=")[1].split("-")[0])
            chunk = self.payload[start:]
            self.send_response(206)
            self.send_header(
                "Content-Range",
                f"bytes {max(0, start - 500)}-{len(self.payload) - 1}/{len(self.payload)}",
            )
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            return

        if range_header and range_header.startswith("bytes="):
            start = int(range_header.split("=")[1].split("-")[0])
            body = self.payload[start:]
            self.send_response(206)
            self.send_header(
                "Content-Range",
                f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}",
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self._write_with_optional_disconnect(body, rule)
            return

        body = self.payload
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self._write_with_optional_disconnect(body, rule)

    def _write_with_optional_disconnect(
        self,
        body: bytes,
        rule: dict[str, object],
    ) -> None:
        disconnect_after = rule.get("disconnect_after_bytes")
        if disconnect_after is not None:
            limit = int(disconnect_after)
            self.wfile.write(body[:limit])
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_RDWR)
            return
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def http_server() -> tuple[str, ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ControllableHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_uri = f"http://127.0.0.1:{server.server_address[1]}/file"
    yield base_uri, server, thread
    server.shutdown()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_clean_download_bytes_written_equals_payload_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"z" * 5_000
    partial = tmp_path / "object.part"
    _install_stream_mock(monkeypatch, [lambda headers: _full_response(payload)])
    transport = HttpDataPackageTransport(max_retries=1, retry_backoff_seconds=0)

    result = transport.download_file("http://example.test/file", partial, resume_from_byte=0)

    assert partial.read_bytes() == payload
    assert result.bytes_written == 5_000


def test_normal_resume_bytes_written_excludes_existing_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"w" * 5_000
    partial = tmp_path / "object.part"
    partial.write_bytes(payload[:2_000])
    _install_stream_mock(monkeypatch, [lambda headers: _range_response(payload, headers)])
    transport = HttpDataPackageTransport(max_retries=1, retry_backoff_seconds=0)

    result = transport.download_file("http://example.test/file", partial, resume_from_byte=2_000)

    assert partial.read_bytes() == payload
    assert result.bytes_written == 3_000


def test_range_ignored_after_mid_stream_failure_counts_only_surviving_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"x" * 5_000
    partial = tmp_path / "object.part"
    requested_headers = _install_stream_mock(
        monkeypatch,
        [
            lambda headers: _full_response(
                payload,
                write_bytes=1_000,
                fail_after_chunk=True,
            ),
            lambda headers: _full_response(payload),
        ],
    )
    transport = HttpDataPackageTransport(max_retries=3, retry_backoff_seconds=0)

    result = transport.download_file("http://example.test/file", partial, resume_from_byte=0)

    assert partial.read_bytes() == payload
    assert [headers.get("Range") for headers in requested_headers] == [
        None,
        "bytes=1000-",
    ]
    assert result.bytes_written == 5_000


def test_pre_existing_partial_reset_counts_only_post_reset_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"y" * 5_000
    partial = tmp_path / "object.part"
    partial.write_bytes(payload[:2_000])
    requested_headers = _install_stream_mock(
        monkeypatch,
        [
            lambda headers: _range_response(
                payload,
                headers,
                write_bytes=1_000,
                fail_after_chunk=True,
            ),
            lambda headers: _full_response(payload),
        ],
    )
    transport = HttpDataPackageTransport(max_retries=3, retry_backoff_seconds=0)

    result = transport.download_file("http://example.test/file", partial, resume_from_byte=2_000)

    assert partial.read_bytes() == payload
    assert [headers.get("Range") for headers in requested_headers] == [
        "bytes=2000-",
        "bytes=3000-",
    ]
    assert result.bytes_written == 5_000


def test_416_reset_discards_prior_invocation_bytes_from_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"m" * 5_000
    partial = tmp_path / "object.part"
    requested_headers = _install_stream_mock(
        monkeypatch,
        [
            lambda headers: _range_response(
                payload,
                headers,
                write_bytes=1_000,
                fail_after_chunk=True,
            ),
            lambda headers: _MockResponse(416, chunks=()),
            lambda headers: _full_response(payload),
        ],
    )
    transport = HttpDataPackageTransport(max_retries=4, retry_backoff_seconds=0)

    result = transport.download_file("http://example.test/file", partial, resume_from_byte=0)

    assert partial.read_bytes() == payload
    assert [headers.get("Range") for headers in requested_headers] == [
        None,
        "bytes=1000-",
        None,
    ]
    assert result.bytes_written == 5_000


def test_single_mid_stream_failure_resumes_from_current_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"a" * 10_240
    partial = tmp_path / "object.part"
    partial.write_bytes(payload[:2_048])
    requested_headers = _install_stream_mock(
        monkeypatch,
        [
            lambda headers: _range_response(
                payload,
                headers,
                write_bytes=3_072,
                fail_after_chunk=True,
            ),
            lambda headers: _range_response(payload, headers),
        ],
    )
    transport = HttpDataPackageTransport(max_retries=3, retry_backoff_seconds=0)

    result = transport.download_file("http://example.test/file", partial, resume_from_byte=2_048)

    assert partial.read_bytes() == payload
    assert _sha256(partial.read_bytes()) == _sha256(payload)
    assert [headers.get("Range") for headers in requested_headers] == [
        "bytes=2048-",
        "bytes=5120-",
    ]
    assert result.resumed is True
    assert result.supports_range is True
    assert result.bytes_written == 8_192


def test_multiple_mid_stream_failures_resume_monotonically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"b" * 4_096
    partial = tmp_path / "object.part"
    requested_headers = _install_stream_mock(
        monkeypatch,
        [
            lambda headers: _full_response(
                payload,
                write_bytes=1_000,
                fail_after_chunk=True,
            ),
            lambda headers: _range_response(
                payload,
                headers,
                write_bytes=1_000,
                fail_after_chunk=True,
            ),
            lambda headers: _range_response(payload, headers),
        ],
    )
    transport = HttpDataPackageTransport(max_retries=5, retry_backoff_seconds=0)

    result = transport.download_file("http://example.test/file", partial, resume_from_byte=0)

    assert partial.read_bytes() == payload
    assert [headers.get("Range") for headers in requested_headers] == [
        None,
        "bytes=1000-",
        "bytes=2000-",
    ]
    assert result.bytes_written == 4_096


def test_server_ignores_range_restarts_only_partial_file(
    tmp_path: Path,
    http_server: tuple[str, ThreadingHTTPServer, threading.Thread],
) -> None:
    payload = b"c" * 2_000
    _ControllableHandler.reset(payload, [{"ignore_range": True}])
    partial = tmp_path / "object.part"
    partial.write_bytes(payload[:1_000])
    transport = HttpDataPackageTransport(max_retries=1)

    result = transport.download_file(http_server[0], partial, resume_from_byte=1_000)

    assert partial.read_bytes() == payload
    assert _ControllableHandler.requested_ranges == ["bytes=1000-"]
    assert result.resumed is False
    assert result.bytes_written == 2_000


def test_invalid_content_range_fails_closed(
    tmp_path: Path,
    http_server: tuple[str, ThreadingHTTPServer, threading.Thread],
) -> None:
    payload = b"d" * 2_000
    _ControllableHandler.reset(payload, [{"bad_content_range": True}])
    partial = tmp_path / "object.part"
    partial.write_bytes(payload[:1_000])
    transport = HttpDataPackageTransport(max_retries=1)

    with pytest.raises(DataPackageTransportError, match="Content-Range"):
        transport.download_file(http_server[0], partial, resume_from_byte=1_000)

    assert partial.stat().st_size == 1_000


def test_httpx_transport_error_maps_to_data_package_transport_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = tmp_path / "object.part"

    class _BrokenClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> _BrokenClient:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def stream(self, *args: object, **kwargs: object) -> None:
            raise httpx.ReadError(
                "simulated read failure",
                request=httpx.Request("GET", "http://example.test/file"),
            )

    monkeypatch.setattr(httpx, "Client", _BrokenClient)
    transport = HttpDataPackageTransport(max_retries=1)
    with pytest.raises(DataPackageTransportError) as exc_info:
        transport.download_file(
            "http://example.test/file",
            partial,
            resume_from_byte=0,
        )
    assert not isinstance(exc_info.value, httpx.ReadError)
    assert not isinstance(exc_info.value, httpx.TransportError)


def test_retry_limit_is_exactly_max_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"e" * 512
    partial = tmp_path / "object.part"
    requested_headers = _install_stream_mock(
        monkeypatch,
        [
            lambda headers: _full_response(
                payload,
                write_bytes=100,
                fail_after_chunk=True,
            ),
            lambda headers: _range_response(
                payload,
                headers,
                write_bytes=100,
                fail_after_chunk=True,
            ),
            lambda headers: _range_response(
                payload,
                headers,
                write_bytes=100,
                fail_after_chunk=True,
            ),
            lambda headers: _range_response(
                payload,
                headers,
                write_bytes=100,
                fail_after_chunk=True,
            ),
        ],
    )
    transport = HttpDataPackageTransport(max_retries=3, retry_backoff_seconds=0)

    with pytest.raises(DataPackageTransportError, match="after 3 attempts"):
        transport.download_file("http://example.test/file", partial, resume_from_byte=0)

    assert len(requested_headers) == 3
    assert partial.stat().st_size == 300


def test_partial_preserved_after_exhausted_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"f" * 1_024
    partial = tmp_path / "object.part"
    _install_stream_mock(
        monkeypatch,
        [
            lambda headers: _full_response(
                payload,
                write_bytes=400,
                fail_after_chunk=True,
            ),
            lambda headers: _immediate_read_error(
                status_code=206,
                headers={
                    "Content-Range": (
                        f"bytes {_range_offset(headers)}-{len(payload) - 1}/{len(payload)}"
                    ),
                },
            ),
        ],
    )
    transport = HttpDataPackageTransport(max_retries=2, retry_backoff_seconds=0)

    with pytest.raises(DataPackageTransportError):
        transport.download_file("http://example.test/file", partial, resume_from_byte=0)

    assert partial.is_file()
    assert partial.stat().st_size == 400


def test_fresh_process_resume_from_existing_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"g" * 3_000
    partial = tmp_path / "object.part"
    first_transport = HttpDataPackageTransport(max_retries=1, retry_backoff_seconds=0)
    _install_stream_mock(
        monkeypatch,
        [
            lambda headers: _full_response(
                payload,
                write_bytes=1_200,
                fail_after_chunk=True,
            ),
        ],
    )

    with pytest.raises(DataPackageTransportError):
        first_transport.download_file("http://example.test/file", partial, resume_from_byte=0)

    assert partial.stat().st_size == 1_200

    requested_headers = _install_stream_mock(
        monkeypatch,
        [lambda headers: _range_response(payload, headers)],
    )
    second_transport = HttpDataPackageTransport(max_retries=2, retry_backoff_seconds=0)
    result = second_transport.download_file(
        "http://example.test/file",
        partial,
        resume_from_byte=0,
    )

    assert partial.read_bytes() == payload
    assert [headers.get("Range") for headers in requested_headers] == ["bytes=1200-"]
    assert result.bytes_written == 1_800


def test_stale_resume_hint_uses_authoritative_partial_size(
    tmp_path: Path,
    http_server: tuple[str, ThreadingHTTPServer, threading.Thread],
) -> None:
    payload = b"h" * 5_000
    _ControllableHandler.reset(payload, [{}])
    partial = tmp_path / "object.part"
    partial.write_bytes(payload[:3_000])
    transport = HttpDataPackageTransport(max_retries=1)

    transport.download_file(http_server[0], partial, resume_from_byte=2_000)

    assert partial.read_bytes() == payload
    assert _ControllableHandler.requested_ranges == ["bytes=3000-"]


def test_missing_partial_with_resume_hint_starts_from_zero(
    tmp_path: Path,
    http_server: tuple[str, ThreadingHTTPServer, threading.Thread],
) -> None:
    payload = b"i" * 256
    _ControllableHandler.reset(payload, [{}])
    partial = tmp_path / "object.part"
    transport = HttpDataPackageTransport(max_retries=1)

    transport.download_file(http_server[0], partial, resume_from_byte=128)

    assert partial.read_bytes() == payload
    assert _ControllableHandler.requested_ranges == [None]


def test_unsafe_redirect_scheme_is_rejected() -> None:
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "file:///etc/passwd"),
    )
    with pytest.raises(DataPackageTransportError, match="unsupported scheme"):
        _reject_unsafe_redirect_scheme(response)


def test_416_restarts_once_without_recursion(
    tmp_path: Path,
    http_server: tuple[str, ThreadingHTTPServer, threading.Thread],
) -> None:
    payload = b"j" * 128
    _ControllableHandler.reset(payload, [{"status": 416}, {}])
    partial = tmp_path / "object.part"
    partial.write_bytes(payload[:64])
    transport = HttpDataPackageTransport(max_retries=2, retry_backoff_seconds=0)

    result = transport.download_file(http_server[0], partial, resume_from_byte=64)

    assert partial.read_bytes() == payload
    assert _ControllableHandler.requested_ranges == ["bytes=64-", None]
    assert result.bytes_written == 128


def test_416_loop_is_bounded(
    tmp_path: Path,
    http_server: tuple[str, ThreadingHTTPServer, threading.Thread],
) -> None:
    payload = b"k" * 64
    _ControllableHandler.reset(payload, [{"status": 416}, {"status": 416}])
    partial = tmp_path / "object.part"
    partial.write_bytes(payload[:32])
    transport = HttpDataPackageTransport(max_retries=3, retry_backoff_seconds=0)

    with pytest.raises(DataPackageTransportError, match="416"):
        transport.download_file(http_server[0], partial, resume_from_byte=32)

    assert len(_ControllableHandler.requested_ranges) == 2

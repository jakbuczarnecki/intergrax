# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import socket
import ssl
import time
from unittest.mock import patch

import pytest

from intergrax.websearch.capture.contracts import (
    WebContentCaptureError,
    WebContentCaptureErrorCode,
)
from intergrax.websearch.capture.http_transport import (
    ApprovedHttpsRequest,
    PinnedHttpsTransport,
    _build_request_bytes,
    _sync_pinned_fetch,
)

pytestmark = pytest.mark.unit


def _deadline(seconds: float = 30.0) -> float:
    return time.monotonic() + seconds


def _request(
    *,
    approved_ips: tuple[str, ...] = ("93.184.216.34",),
    max_response_bytes: int = 1024,
    request_target: str = "/",
    port: int = 443,
) -> ApprovedHttpsRequest:
    return ApprovedHttpsRequest(
        hostname="example.com",
        port=port,
        request_target=request_target,
        approved_ips=approved_ips,
        deadline=_deadline(),
        max_response_bytes=max_response_bytes,
    )


def _http_bytes(status: int, headers: dict[str, str], body: bytes) -> bytes:
    lines = [f"HTTP/1.1 {status} OK"]
    for key, value in headers.items():
        lines.append(f"{key}: {value}")
    header_block = "\r\n".join(lines) + "\r\n\r\n"
    return header_block.encode("iso-8859-1") + body


def _chunked_http_bytes(status: int, chunks: list[bytes], *, trailers: str = "") -> bytes:
  lines = [f"HTTP/1.1 {status} OK", "Transfer-Encoding: chunked"]
  header_block = "\r\n".join(lines) + "\r\n\r\n"
  payload = bytearray(header_block.encode("iso-8859-1"))
  for chunk in chunks:
      payload.extend(f"{len(chunk):x}\r\n".encode("ascii"))
      payload.extend(chunk)
      payload.extend(b"\r\n")
  payload.extend(b"0\r\n")
  if trailers:
      payload.extend(trailers.encode("ascii"))
      if not trailers.endswith("\r\n"):
          payload.extend(b"\r\n")
  payload.extend(b"\r\n")
  return bytes(payload)


class FakeStreamSocket:
    def __init__(self, payload: bytes, *, recv_sizes: list[int] | None = None) -> None:
        self._payload = payload
        self._pos = 0
        self.closed = False
        self.connect_calls: list[tuple[str, int]] = []
        self.recv_sizes = list(recv_sizes or [])
        self.recv_calls = 0
        self.timeouts: list[float] = []

    def connect(self, address: tuple[str, int]) -> None:
        self.connect_calls.append(address)

    def sendall(self, data: bytes) -> None:
        self.sent = data

    def recv(self, size: int) -> bytes:
        if self.recv_sizes:
            size = self.recv_sizes.pop(0)
        chunk = self._payload[self._pos : self._pos + size]
        self._pos += len(chunk)
        self.recv_calls += 1
        return chunk

    def settimeout(self, timeout: float) -> None:
        self.timeouts.append(timeout)

    def do_handshake(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class FakeMonotonic:
    def __init__(self, start: float = 0.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, delta: float) -> None:
        self.value += delta


def _wrap_passthrough(
    sock: socket.socket,
    server_hostname: str | None = None,
    **kwargs: object,
) -> socket.socket:
    return sock


def test_build_request_bytes_uses_hostname_in_host_header() -> None:
    payload = _build_request_bytes("example.com", 443, "/path?q=1")
    text = payload.decode("ascii")
    assert "Host: example.com" in text
    assert "GET /path?q=1 HTTP/1.1" in text
    assert "Accept-Encoding: identity" in text
    assert "example.com/botinfo" not in text


def test_build_request_bytes_includes_non_default_port() -> None:
    payload = _build_request_bytes("example.com", 8443, "/")
    text = payload.decode("ascii")
    assert "Host: example.com:8443" in text


@pytest.mark.asyncio
async def test_approved_ip_used_as_connect_target() -> None:
    fake_sock = FakeStreamSocket(_http_bytes(200, {"Content-Length": "2"}, b"ok"))
    connections: list[tuple[str, int]] = []

    def connect_factory(host: str, port: int, timeout: float) -> FakeStreamSocket:
        connections.append((host, port))
        return fake_sock

    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=connect_factory)
        response = await transport.fetch(_request())

    assert connections == [("93.184.216.34", 443)]
    assert response.status_code == 200
    assert response.body == b"ok"


@pytest.mark.asyncio
async def test_tls_hostname_preserved() -> None:
    fake_sock = FakeStreamSocket(_http_bytes(200, {"Content-Length": "2"}, b"ok"))
    captured_hostnames: list[str] = []

    def connect_factory(_host: str, _port: int, _timeout: float) -> FakeStreamSocket:
        return fake_sock

    def wrap_socket(
        sock: socket.socket,
        server_hostname: str | None = None,
        **kwargs: object,
    ) -> socket.socket:
        captured_hostnames.append(server_hostname or "")
        return sock

    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = wrap_socket
        transport = PinnedHttpsTransport(connect_factory=connect_factory)
        await transport.fetch(_request())

    assert captured_hostnames == ["example.com"]


@pytest.mark.asyncio
async def test_response_read_in_bounded_chunks() -> None:
    body = b"a" * 5000
    fake_sock = FakeStreamSocket(
        _http_bytes(200, {"Content-Length": str(len(body))}, body)
    )

    def connect_factory(_host: str, _port: int, _timeout: float) -> FakeStreamSocket:
        return fake_sock

    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=connect_factory)
        response = await transport.fetch(_request(max_response_bytes=10000))

    assert len(response.body) == 5000


@pytest.mark.asyncio
async def test_content_length_preflight_enforced() -> None:
    fake_sock = FakeStreamSocket(_http_bytes(200, {"Content-Length": "9999"}, b""))

    def connect_factory(_host: str, _port: int, _timeout: float) -> FakeStreamSocket:
        return fake_sock

    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=connect_factory)
        with pytest.raises(WebContentCaptureError) as exc:
            await transport.fetch(_request(max_response_bytes=100))
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_CONTENT_TOO_LARGE


@pytest.mark.asyncio
async def test_streamed_overflow_enforced() -> None:
    body = b"x" * 200
    fake_sock = FakeStreamSocket(_http_bytes(200, {}, body))

    def connect_factory(_host: str, _port: int, _timeout: float) -> FakeStreamSocket:
        return fake_sock

    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=connect_factory)
        with pytest.raises(WebContentCaptureError) as exc:
            await transport.fetch(_request(max_response_bytes=50))
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_CONTENT_TOO_LARGE


@pytest.mark.asyncio
async def test_socket_closes_after_success() -> None:
    fake_sock = FakeStreamSocket(_http_bytes(200, {"Content-Length": "2"}, b"ok"))

    def connect_factory(_host: str, _port: int, _timeout: float) -> FakeStreamSocket:
        return fake_sock

    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=connect_factory)
        await transport.fetch(_request())

    assert fake_sock.closed


@pytest.mark.asyncio
async def test_socket_closes_after_failure() -> None:
    fake_sock = FakeStreamSocket(_http_bytes(200, {"Content-Length": "9999"}, b""))

    def connect_factory(_host: str, _port: int, _timeout: float) -> FakeStreamSocket:
        return fake_sock

    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=connect_factory)
        with pytest.raises(WebContentCaptureError):
            await transport.fetch(_request(max_response_bytes=10))

    assert fake_sock.closed


def test_sync_pinned_fetch_tls_failure_maps_to_safe_error() -> None:
    fake_sock = FakeStreamSocket(b"")

    def connect_factory(_host: str, _port: int, _timeout: float) -> FakeStreamSocket:
        return fake_sock

    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = ssl.SSLError("tls boom")
        with pytest.raises(WebContentCaptureError) as exc:
            _sync_pinned_fetch(
                _request(),
                "93.184.216.34",
                connect_factory,
                time.monotonic,
            )
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_TLS_FAILED
    assert "boom" not in str(exc.value)


@pytest.mark.asyncio
async def test_content_length_body_split_across_recv() -> None:
    body = b"hello-world"
    payload = _http_bytes(200, {"Content-Length": str(len(body))}, body)
    fake_sock = FakeStreamSocket(payload, recv_sizes=[20, 20, 200])

    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(
            connect_factory=lambda *_args: fake_sock,
        )
        response = await transport.fetch(_request())
    assert response.body == body


@pytest.mark.asyncio
async def test_transport_stops_exactly_at_content_length() -> None:
    body = b"12345"
    extra = b"SHOULD-NOT-READ"
    payload = _http_bytes(200, {"Content-Length": str(len(body))}, body + extra)
    fake_sock = FakeStreamSocket(payload)

    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(
            connect_factory=lambda *_args: fake_sock,
        )
        response = await transport.fetch(_request())
    assert response.body == body


@pytest.mark.asyncio
async def test_premature_eof_rejected() -> None:
    header = _http_bytes(200, {"Content-Length": "10"}, b"short")
    fake_sock = FakeStreamSocket(header)

    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=lambda *_args: fake_sock)
        with pytest.raises(WebContentCaptureError) as exc:
            await transport.fetch(_request())
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID


@pytest.mark.parametrize(
    "content_length",
    ["-1", "abc", "1 2"],
)
@pytest.mark.asyncio
async def test_invalid_content_length_rejected(content_length: str) -> None:
    fake_sock = FakeStreamSocket(_http_bytes(200, {"Content-Length": content_length}, b""))
    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=lambda *_args: fake_sock)
        with pytest.raises(WebContentCaptureError) as exc:
            await transport.fetch(_request())
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID


@pytest.mark.asyncio
async def test_conflicting_duplicate_content_length_rejected() -> None:
    raw = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Length: 5\r\n"
        b"Content-Length: 6\r\n"
        b"\r\n"
        b"12345"
    )
    fake_sock = FakeStreamSocket(raw)
    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=lambda *_args: fake_sock)
        with pytest.raises(WebContentCaptureError) as exc:
            await transport.fetch(_request())
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID


@pytest.mark.asyncio
async def test_te_plus_cl_rejected() -> None:
    raw = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Length: 5\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"0\r\n\r\n"
    )
    fake_sock = FakeStreamSocket(raw)
    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=lambda *_args: fake_sock)
        with pytest.raises(WebContentCaptureError) as exc:
            await transport.fetch(_request())
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID


@pytest.mark.asyncio
async def test_valid_chunked_response_decoded() -> None:
    fake_sock = FakeStreamSocket(_chunked_http_bytes(200, [b"hello", b" world"]))
    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=lambda *_args: fake_sock)
        response = await transport.fetch(_request())
    assert response.body == b"hello world"


@pytest.mark.asyncio
async def test_chunk_extensions_supported() -> None:
    raw = (
        b"HTTP/1.1 200 OK\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"5;ext=1\r\n"
        b"hello\r\n"
        b"0\r\n\r\n"
    )
    fake_sock = FakeStreamSocket(raw)
    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=lambda *_args: fake_sock)
        response = await transport.fetch(_request())
    assert response.body == b"hello"


@pytest.mark.asyncio
async def test_chunked_body_split_across_recv_boundaries() -> None:
    payload = _chunked_http_bytes(200, [b"abc", b"def"])
    fake_sock = FakeStreamSocket(payload, recv_sizes=[1, 1, 1, 200])
    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=lambda *_args: fake_sock)
        response = await transport.fetch(_request())
    assert response.body == b"abcdef"


@pytest.mark.asyncio
async def test_malformed_chunk_size_rejected() -> None:
    raw = (
        b"HTTP/1.1 200 OK\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"zz\r\n"
        b"0\r\n\r\n"
    )
    fake_sock = FakeStreamSocket(raw)
    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=lambda *_args: fake_sock)
        with pytest.raises(WebContentCaptureError) as exc:
            await transport.fetch(_request())
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID


@pytest.mark.asyncio
async def test_missing_chunk_crlf_rejected() -> None:
    raw = (
        b"HTTP/1.1 200 OK\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"5\r\n"
        b"hello\n"
        b"0\r\n\r\n"
    )
    fake_sock = FakeStreamSocket(raw)
    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=lambda *_args: fake_sock)
        with pytest.raises(WebContentCaptureError) as exc:
            await transport.fetch(_request())
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID


@pytest.mark.asyncio
async def test_missing_zero_chunk_rejected() -> None:
    raw = (
        b"HTTP/1.1 200 OK\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"5\r\n"
        b"hello\r\n"
    )
    fake_sock = FakeStreamSocket(raw)
    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=lambda *_args: fake_sock)
        with pytest.raises(WebContentCaptureError) as exc:
            await transport.fetch(_request())
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID


@pytest.mark.asyncio
async def test_chunked_decoded_overflow_rejected() -> None:
    fake_sock = FakeStreamSocket(_chunked_http_bytes(200, [b"x" * 100]))
    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=lambda *_args: fake_sock)
        with pytest.raises(WebContentCaptureError) as exc:
            await transport.fetch(_request(max_response_bytes=50))
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_CONTENT_TOO_LARGE


@pytest.mark.asyncio
async def test_oversized_trailers_rejected() -> None:
    trailers = "X-Trailer: " + ("a" * 5000) + "\r\n"
    fake_sock = FakeStreamSocket(_chunked_http_bytes(200, [b"ok"], trailers=trailers))
    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=lambda *_args: fake_sock)
        with pytest.raises(WebContentCaptureError) as exc:
            await transport.fetch(_request())
    assert exc.value.code in {
        WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID,
        WebContentCaptureErrorCode.WEB_URL_CONTENT_TOO_LARGE,
    }


@pytest.mark.asyncio
async def test_unsupported_transfer_encoding_rejected() -> None:
    raw = (
        b"HTTP/1.1 200 OK\r\n"
        b"Transfer-Encoding: gzip\r\n"
        b"\r\n"
        b"data"
    )
    fake_sock = FakeStreamSocket(raw)
    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=lambda *_args: fake_sock)
        with pytest.raises(WebContentCaptureError) as exc:
            await transport.fetch(_request())
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID


@pytest.mark.asyncio
async def test_first_connect_failure_second_ip_success() -> None:
    good_sock = FakeStreamSocket(_http_bytes(200, {"Content-Length": "2"}, b"ok"))
    connections: list[str] = []

    def connect_factory(host: str, port: int, timeout: float) -> FakeStreamSocket:
        connections.append(host)
        if host == "1.1.1.1":
            raise OSError("connect failed")
        return good_sock

    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=connect_factory)
        response = await transport.fetch(
            _request(approved_ips=("1.1.1.1", "93.184.216.34")),
        )
    assert connections == ["1.1.1.1", "93.184.216.34"]
    assert response.body == b"ok"


@pytest.mark.asyncio
async def test_first_tls_failure_second_ip_success() -> None:
    good_sock = FakeStreamSocket(_http_bytes(200, {"Content-Length": "2"}, b"ok"))
    connections: list[str] = []

    def connect_factory(host: str, port: int, timeout: float) -> FakeStreamSocket:
        connections.append(host)
        return good_sock

    def wrap_socket(
        sock: socket.socket,
        server_hostname: str | None = None,
        **kwargs: object,
    ) -> socket.socket:
        if connections[-1] == "1.1.1.1":
            raise ssl.SSLError("tls failed")
        return sock

    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = wrap_socket
        transport = PinnedHttpsTransport(connect_factory=connect_factory)
        response = await transport.fetch(
            _request(approved_ips=("1.1.1.1", "93.184.216.34")),
        )
    assert connections == ["1.1.1.1", "93.184.216.34"]
    assert response.body == b"ok"


@pytest.mark.asyncio
async def test_deadline_exhausted_skips_next_ip() -> None:
    clock = FakeMonotonic(0.0)
    request = ApprovedHttpsRequest(
        hostname="example.com",
        port=443,
        request_target="/",
        approved_ips=("1.1.1.1", "93.184.216.34"),
        deadline=1.0,
        max_response_bytes=1024,
    )

    def connect_factory(host: str, port: int, timeout: float) -> FakeStreamSocket:
        clock.advance(1.1)
        raise OSError("slow connect")

    transport = PinnedHttpsTransport(connect_factory=connect_factory, monotonic=clock)
    with pytest.raises(WebContentCaptureError) as exc:
        await transport.fetch(request)
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_TIMEOUT


@pytest.mark.asyncio
async def test_non_retryable_framing_error_skips_next_ip() -> None:
    bad_sock = FakeStreamSocket(_http_bytes(200, {"Content-Length": "abc"}, b""))
    connections: list[str] = []

    def connect_factory(host: str, port: int, timeout: float) -> FakeStreamSocket:
        connections.append(host)
        return bad_sock

    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=connect_factory)
        with pytest.raises(WebContentCaptureError) as exc:
            await transport.fetch(
                _request(approved_ips=("1.1.1.1", "93.184.216.34")),
            )
    assert connections == ["1.1.1.1"]
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID


@pytest.mark.parametrize(
    "encoding",
    ["gzip", "br", "deflate"],
)
@pytest.mark.asyncio
async def test_content_encoding_rejected(encoding: str) -> None:
    fake_sock = FakeStreamSocket(
        _http_bytes(200, {"Content-Encoding": encoding, "Content-Length": "2"}, b"ok"),
    )
    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=lambda *_args: fake_sock)
        with pytest.raises(WebContentCaptureError) as exc:
            await transport.fetch(_request())
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_CONTENT_ENCODING_UNSUPPORTED


@pytest.mark.asyncio
async def test_content_encoding_identity_accepted() -> None:
    fake_sock = FakeStreamSocket(
        _http_bytes(200, {"Content-Encoding": "identity", "Content-Length": "2"}, b"ok"),
    )
    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=lambda *_args: fake_sock)
        response = await transport.fetch(_request())
    assert response.body == b"ok"


def test_connect_time_reduces_tls_handshake_timeout() -> None:
    clock = FakeMonotonic(0.0)
    fake_sock = FakeStreamSocket(_http_bytes(200, {"Content-Length": "2"}, b"ok"))
    request = ApprovedHttpsRequest(
        hostname="example.com",
        port=443,
        request_target="/",
        approved_ips=("93.184.216.34",),
        deadline=10.0,
        max_response_bytes=1024,
    )

    def connect_factory(_host: str, _port: int, timeout: float) -> FakeStreamSocket:
        assert timeout == pytest.approx(10.0)
        clock.advance(9.0)
        return fake_sock

    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        _sync_pinned_fetch(request, "93.184.216.34", connect_factory, clock)

    assert fake_sock.timeouts[0] == pytest.approx(1.0)


def test_deadline_expires_after_connect_before_handshake() -> None:
    clock = FakeMonotonic(0.0)
    fake_sock = FakeStreamSocket(_http_bytes(200, {"Content-Length": "2"}, b"ok"))
    request = ApprovedHttpsRequest(
        hostname="example.com",
        port=443,
        request_target="/",
        approved_ips=("93.184.216.34",),
        deadline=1.0,
        max_response_bytes=1024,
    )

    def connect_factory(_host: str, _port: int, _timeout: float) -> FakeStreamSocket:
        clock.advance(1.5)
        return fake_sock

    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        with pytest.raises(WebContentCaptureError) as exc:
            _sync_pinned_fetch(request, "93.184.216.34", connect_factory, clock)
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_TIMEOUT
    assert exc.value.retryable is True


@pytest.mark.asyncio
async def test_cumulative_trailer_limit_exceeded() -> None:
    trailer_lines = "".join(f"T-{index:02d}: {'a' * 90}\r\n" for index in range(60))
    fake_sock = FakeStreamSocket(_chunked_http_bytes(200, [b"ok"], trailers=trailer_lines))
    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=lambda *_args: fake_sock)
        with pytest.raises(WebContentCaptureError) as exc:
            await transport.fetch(_request())
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID


@pytest.mark.asyncio
async def test_malformed_trailer_name_rejected() -> None:
    trailers = " bad-name: value\r\n"
    fake_sock = FakeStreamSocket(_chunked_http_bytes(200, [b"ok"], trailers=trailers))
    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=lambda *_args: fake_sock)
        with pytest.raises(WebContentCaptureError) as exc:
            await transport.fetch(_request())
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID


@pytest.mark.asyncio
async def test_trailer_obs_fold_rejected() -> None:
    trailers = " X-Trailer: folded\r\n"
    fake_sock = FakeStreamSocket(_chunked_http_bytes(200, [b"ok"], trailers=trailers))
    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=lambda *_args: fake_sock)
        with pytest.raises(WebContentCaptureError) as exc:
            await transport.fetch(_request())
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID


@pytest.mark.asyncio
async def test_non_ascii_chunk_size_rejected() -> None:
    raw = (
        b"HTTP/1.1 200 OK\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"\xff5\r\n"
        b"hello\r\n"
        b"0\r\n\r\n"
    )
    fake_sock = FakeStreamSocket(raw)
    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=lambda *_args: fake_sock)
        with pytest.raises(WebContentCaptureError) as exc:
            await transport.fetch(_request())
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID


@pytest.mark.asyncio
async def test_duplicate_transfer_encoding_rejected() -> None:
    raw = (
        b"HTTP/1.1 200 OK\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"0\r\n\r\n"
    )
    fake_sock = FakeStreamSocket(raw)
    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=lambda *_args: fake_sock)
        with pytest.raises(WebContentCaptureError) as exc:
            await transport.fetch(_request())
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID


@pytest.mark.asyncio
async def test_duplicate_content_encoding_rejected() -> None:
    raw = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Encoding: identity\r\n"
        b"Content-Encoding: identity\r\n"
        b"Content-Length: 2\r\n"
        b"\r\n"
        b"ok"
    )
    fake_sock = FakeStreamSocket(raw)
    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=lambda *_args: fake_sock)
        with pytest.raises(WebContentCaptureError) as exc:
            await transport.fetch(_request())
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_CONTENT_ENCODING_UNSUPPORTED


@pytest.mark.parametrize(
    "raw_headers",
    [
        b"Content-Length : 5\r\n",
        b"Content Length: 5\r\n",
        b": value\r\n",
        b"name\t: value\r\n",
    ],
)
@pytest.mark.asyncio
async def test_invalid_header_name_whitespace_rejected(raw_headers: bytes) -> None:
    raw = b"HTTP/1.1 200 OK\r\n" + raw_headers + b"\r\n" + b"12345"
    fake_sock = FakeStreamSocket(raw)
    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=lambda *_args: fake_sock)
        with pytest.raises(WebContentCaptureError) as exc:
            await transport.fetch(_request())
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID


@pytest.mark.parametrize(
    "status_line",
    [
        "XYZ 200 OK",
        "HTTP/2 200",
        "HTTP/1.1 OK",
        "HTTP/1.1 20",
        "HTTP/1.1 2000",
        "HTTP/1.1 999",
    ],
)
@pytest.mark.asyncio
async def test_invalid_status_line_rejected(status_line: str) -> None:
    raw = f"{status_line}\r\nContent-Length: 2\r\n\r\nok".encode("ascii")
    fake_sock = FakeStreamSocket(raw)
    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = _wrap_passthrough
        transport = PinnedHttpsTransport(connect_factory=lambda *_args: fake_sock)
        with pytest.raises(WebContentCaptureError) as exc:
            await transport.fetch(_request())
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID

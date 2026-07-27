# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import socket
import ssl
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


def _http_bytes(status: int, headers: dict[str, str], body: bytes) -> bytes:
    lines = [f"HTTP/1.1 {status} OK"]
    for key, value in headers.items():
        lines.append(f"{key}: {value}")
    header_block = "\r\n".join(lines) + "\r\n\r\n"
    return header_block.encode("iso-8859-1") + body


class FakeStreamSocket:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._pos = 0
        self.closed = False
        self.connect_calls: list[tuple[str, int]] = []

    def connect(self, address: tuple[str, int]) -> None:
        self.connect_calls.append(address)

    def sendall(self, data: bytes) -> None:
        self.sent = data

    def recv(self, size: int) -> bytes:
        chunk = self._payload[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk

    def settimeout(self, _timeout: float) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def test_build_request_bytes_uses_hostname_in_host_header() -> None:
    payload = _build_request_bytes("example.com", "/path?q=1")
    text = payload.decode("ascii")
    assert "Host: example.com" in text
    assert "GET /path?q=1 HTTP/1.1" in text
    assert "Accept-Encoding: identity" in text
    assert "example.com/botinfo" not in text


@pytest.mark.asyncio
async def test_approved_ip_used_as_connect_target() -> None:
    fake_sock = FakeStreamSocket(_http_bytes(200, {"Content-Length": "2"}, b"ok"))
    connections: list[tuple[str, int]] = []

    def connect_factory(host: str, port: int, timeout: float) -> FakeStreamSocket:
        connections.append((host, port))
        return fake_sock

    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = lambda sock, server_hostname=None: sock
        transport = PinnedHttpsTransport(connect_factory=connect_factory)
        response = await transport.fetch(
            ApprovedHttpsRequest(
                hostname="example.com",
                port=443,
                request_target="/",
                approved_ips=("93.184.216.34",),
                timeout_seconds=5.0,
                max_response_bytes=1024,
            )
        )

    assert connections == [("93.184.216.34", 443)]
    assert response.status_code == 200
    assert response.body == b"ok"


@pytest.mark.asyncio
async def test_tls_hostname_preserved() -> None:
    fake_sock = FakeStreamSocket(_http_bytes(200, {"Content-Length": "2"}, b"ok"))
    captured_hostnames: list[str] = []

    def connect_factory(_host: str, _port: int, _timeout: float) -> FakeStreamSocket:
        return fake_sock

    def wrap_socket(sock: socket.socket, server_hostname: str | None = None) -> socket.socket:
        captured_hostnames.append(server_hostname or "")
        return sock

    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = wrap_socket
        transport = PinnedHttpsTransport(connect_factory=connect_factory)
        await transport.fetch(
            ApprovedHttpsRequest(
                hostname="example.com",
                port=443,
                request_target="/",
                approved_ips=("93.184.216.34",),
                timeout_seconds=5.0,
                max_response_bytes=1024,
            )
        )

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
        mock_ctx.return_value.wrap_socket.side_effect = lambda sock, server_hostname=None: sock
        transport = PinnedHttpsTransport(connect_factory=connect_factory)
        response = await transport.fetch(
            ApprovedHttpsRequest(
                hostname="example.com",
                port=443,
                request_target="/",
                approved_ips=("93.184.216.34",),
                timeout_seconds=5.0,
                max_response_bytes=10000,
            )
        )

    assert len(response.body) == 5000


@pytest.mark.asyncio
async def test_content_length_preflight_enforced() -> None:
    fake_sock = FakeStreamSocket(_http_bytes(200, {"Content-Length": "9999"}, b""))

    def connect_factory(_host: str, _port: int, _timeout: float) -> FakeStreamSocket:
        return fake_sock

    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = lambda sock, server_hostname=None: sock
        transport = PinnedHttpsTransport(connect_factory=connect_factory)
        with pytest.raises(WebContentCaptureError) as exc:
            await transport.fetch(
                ApprovedHttpsRequest(
                    hostname="example.com",
                    port=443,
                    request_target="/",
                    approved_ips=("93.184.216.34",),
                    timeout_seconds=5.0,
                    max_response_bytes=100,
                )
            )
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_CONTENT_TOO_LARGE


@pytest.mark.asyncio
async def test_streamed_overflow_enforced() -> None:
    body = b"x" * 200
    fake_sock = FakeStreamSocket(_http_bytes(200, {}, body))

    def connect_factory(_host: str, _port: int, _timeout: float) -> FakeStreamSocket:
        return fake_sock

    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = lambda sock, server_hostname=None: sock
        transport = PinnedHttpsTransport(connect_factory=connect_factory)
        with pytest.raises(WebContentCaptureError) as exc:
            await transport.fetch(
                ApprovedHttpsRequest(
                    hostname="example.com",
                    port=443,
                    request_target="/",
                    approved_ips=("93.184.216.34",),
                    timeout_seconds=5.0,
                    max_response_bytes=50,
                )
            )
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_CONTENT_TOO_LARGE


@pytest.mark.asyncio
async def test_socket_closes_after_success() -> None:
    fake_sock = FakeStreamSocket(_http_bytes(200, {"Content-Length": "2"}, b"ok"))

    def connect_factory(_host: str, _port: int, _timeout: float) -> FakeStreamSocket:
        return fake_sock

    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = lambda sock, server_hostname=None: sock
        transport = PinnedHttpsTransport(connect_factory=connect_factory)
        await transport.fetch(
            ApprovedHttpsRequest(
                hostname="example.com",
                port=443,
                request_target="/",
                approved_ips=("93.184.216.34",),
                timeout_seconds=5.0,
                max_response_bytes=1024,
            )
        )

    assert fake_sock.closed


@pytest.mark.asyncio
async def test_socket_closes_after_failure() -> None:
    fake_sock = FakeStreamSocket(_http_bytes(200, {"Content-Length": "9999"}, b""))

    def connect_factory(_host: str, _port: int, _timeout: float) -> FakeStreamSocket:
        return fake_sock

    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = lambda sock, server_hostname=None: sock
        transport = PinnedHttpsTransport(connect_factory=connect_factory)
        with pytest.raises(WebContentCaptureError):
            await transport.fetch(
                ApprovedHttpsRequest(
                    hostname="example.com",
                    port=443,
                    request_target="/",
                    approved_ips=("93.184.216.34",),
                    timeout_seconds=5.0,
                    max_response_bytes=10,
                )
            )

    assert fake_sock.closed


def test_sync_pinned_fetch_tls_failure_maps_to_safe_error() -> None:
    fake_sock = FakeStreamSocket(b"")

    def connect_factory(_host: str, _port: int, _timeout: float) -> FakeStreamSocket:
        return fake_sock

    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = ssl.SSLError("tls boom")
        with pytest.raises(WebContentCaptureError) as exc:
            _sync_pinned_fetch(
                ApprovedHttpsRequest(
                    hostname="example.com",
                    port=443,
                    request_target="/",
                    approved_ips=("93.184.216.34",),
                    timeout_seconds=5.0,
                    max_response_bytes=1024,
                ),
                "93.184.216.34",
                connect_factory,
            )
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_TLS_FAILED
    assert "boom" not in str(exc.value)

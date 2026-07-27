# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import asyncio
import socket
import ssl
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from intergrax.websearch.capture.contracts import (
    WebContentCaptureError,
    WebContentCaptureErrorCode,
)

_USER_AGENT = "IntergraxWebContentCapture/1.0"
_ACCEPT = "text/html,application/xhtml+xml,text/plain"
_MAX_HEADER_BYTES = 65536
_READ_CHUNK_SIZE = 65536


@dataclass(frozen=True, slots=True)
class ApprovedHttpsRequest:
    hostname: str
    port: int
    request_target: str
    approved_ips: tuple[str, ...]
    timeout_seconds: float
    max_response_bytes: int


@dataclass(frozen=True, slots=True)
class RawHttpsResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes
    content_bytes: int


ConnectFactory = Callable[
    [str, int, float],
    socket.socket,
]


@runtime_checkable
class HttpsTransport(Protocol):
    async def fetch(self, request: ApprovedHttpsRequest) -> RawHttpsResponse: ...


def _normalize_headers(raw_headers: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in raw_headers.items():
        normalized[key.lower()] = value
    return normalized


def _parse_status_line(line: str) -> int:
    parts = line.split(" ", 2)
    if len(parts) < 2:
        raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_FETCH_FAILED)
    try:
        return int(parts[1])
    except ValueError:
        raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_FETCH_FAILED)


def _read_headers(sock: socket.socket, timeout: float) -> tuple[int, dict[str, str], bytes]:
    sock.settimeout(timeout)
    buffer = bytearray()
    while b"\r\n\r\n" not in buffer:
        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_TIMEOUT,
                retryable=True,
            )
        except OSError:
            raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_FETCH_FAILED)
        if not chunk:
            raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_FETCH_FAILED)
        buffer.extend(chunk)
        if len(buffer) > _MAX_HEADER_BYTES:
            raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_FETCH_FAILED)

    header_bytes, remainder = buffer.split(b"\r\n\r\n", 1)
    lines = header_bytes.decode("iso-8859-1").split("\r\n")
    status_code = _parse_status_line(lines[0])
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        headers[name.strip().lower()] = value.strip()
    return status_code, headers, remainder


def _read_body_bounded(
    sock: socket.socket,
    *,
    timeout: float,
    max_response_bytes: int,
    headers: Mapping[str, str],
    initial_body: bytes = b"",
) -> bytes:
    sock.settimeout(timeout)
    content_length_raw = headers.get("content-length")
    if content_length_raw is not None:
        try:
            content_length = int(content_length_raw)
        except ValueError:
            content_length = None
        else:
            if content_length > max_response_bytes:
                raise WebContentCaptureError(
                    WebContentCaptureErrorCode.WEB_URL_CONTENT_TOO_LARGE,
                )

    body = bytearray(initial_body)
    if len(body) > max_response_bytes:
        raise WebContentCaptureError(
            WebContentCaptureErrorCode.WEB_URL_CONTENT_TOO_LARGE,
        )
    if content_length_raw is not None:
        try:
            expected = int(content_length_raw)
            if len(body) >= expected:
                if len(body) > max_response_bytes:
                    raise WebContentCaptureError(
                        WebContentCaptureErrorCode.WEB_URL_CONTENT_TOO_LARGE,
                    )
                return bytes(body[:expected])
        except ValueError:
            pass

    while True:
        try:
            chunk = sock.recv(_READ_CHUNK_SIZE)
        except socket.timeout:
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_TIMEOUT,
                retryable=True,
            )
        except OSError:
            raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_FETCH_FAILED)
        if not chunk:
            break
        body.extend(chunk)
        if len(body) > max_response_bytes:
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_CONTENT_TOO_LARGE,
            )
    return bytes(body)


def _build_request_bytes(hostname: str, request_target: str) -> bytes:
    if not request_target.startswith("/"):
        request_target = f"/{request_target}"
    lines = [
        f"GET {request_target} HTTP/1.1",
        f"Host: {hostname}",
        f"User-Agent: {_USER_AGENT}",
        f"Accept: {_ACCEPT}",
        "Accept-Encoding: identity",
        "Connection: close",
        "",
        "",
    ]
    return "\r\n".join(lines).encode("ascii")


def _sync_pinned_fetch(
    request: ApprovedHttpsRequest,
    connect_ip: str,
    connect_factory: ConnectFactory,
) -> RawHttpsResponse:
    sock: socket.socket | None = None
    ssl_sock: ssl.SSLSocket | None = None
    try:
        sock = connect_factory(connect_ip, request.port, request.timeout_seconds)
        context = ssl.create_default_context()
        ssl_sock = context.wrap_socket(sock, server_hostname=request.hostname)
        request_bytes = _build_request_bytes(request.hostname, request.request_target)
        ssl_sock.sendall(request_bytes)
        status_code, headers, remainder = _read_headers(ssl_sock, request.timeout_seconds)
        body = _read_body_bounded(
            ssl_sock,
            timeout=request.timeout_seconds,
            max_response_bytes=request.max_response_bytes,
            headers=headers,
            initial_body=remainder,
        )
        return RawHttpsResponse(
            status_code=status_code,
            headers=_normalize_headers(headers),
            body=body,
            content_bytes=len(body),
        )
    except WebContentCaptureError:
        raise
    except ssl.SSLError:
        raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_TLS_FAILED)
    except OSError:
        raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_FETCH_FAILED)
    finally:
        if ssl_sock is not None:
            try:
                ssl_sock.close()
            except OSError:
                pass
        elif sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def _default_connect_factory(host: str, port: int, timeout: float) -> socket.socket:
    return socket.create_connection((host, port), timeout=timeout)


class PinnedHttpsTransport:
    def __init__(
        self,
        *,
        connect_factory: ConnectFactory | None = None,
        fetch_count: list[int] | None = None,
    ) -> None:
        self._connect_factory = connect_factory or _default_connect_factory
        self._fetch_count = fetch_count

    async def fetch(self, request: ApprovedHttpsRequest) -> RawHttpsResponse:
        if self._fetch_count is not None:
            self._fetch_count.append(1)
        last_error: WebContentCaptureError | None = None
        for connect_ip in request.approved_ips:
            try:
                return await asyncio.to_thread(
                    _sync_pinned_fetch,
                    request,
                    connect_ip,
                    self._connect_factory,
                )
            except WebContentCaptureError as exc:
                if exc.code == WebContentCaptureErrorCode.WEB_URL_TLS_FAILED:
                    last_error = exc
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_FETCH_FAILED)

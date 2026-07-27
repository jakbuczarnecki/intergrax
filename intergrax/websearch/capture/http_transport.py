# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import asyncio
import re
import socket
import ssl
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.websearch.capture.contracts import (
    WebContentCaptureError,
    WebContentCaptureErrorCode,
)

_USER_AGENT = "IntergraxWebContentCapture/1.0"
_ACCEPT = "text/html,application/xhtml+xml,text/plain"
_MAX_HEADER_BYTES = 65536
_READ_CHUNK_SIZE = 65536
_MAX_TRAILER_BYTES = 4096
_RETRYABLE_IP_ERRORS = frozenset(
    {
        WebContentCaptureErrorCode.WEB_URL_TLS_FAILED,
        WebContentCaptureErrorCode.WEB_URL_FETCH_FAILED,
        WebContentCaptureErrorCode.WEB_URL_TIMEOUT,
    },
)
_HEADER_TOKEN_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_ALLOWED_HTTP_VERSIONS = frozenset({"HTTP/1.0", "HTTP/1.1"})


@dataclass(frozen=True, slots=True)
class ApprovedHttpsRequest:
    hostname: str
    port: int
    request_target: str
    approved_ips: tuple[str, ...]
    deadline: float
    max_response_bytes: int

    def __repr__(self) -> str:
        return (
            f"ApprovedHttpsRequest(hostname={self.hostname!r}, port={self.port}, "
            f"approved_ip_count={len(self.approved_ips)}, "
            f"max_response_bytes={self.max_response_bytes})"
        )


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
MonotonicClock = Callable[[], float]


@runtime_checkable
class HttpsTransport(Protocol):
    async def fetch(self, request: ApprovedHttpsRequest) -> RawHttpsResponse: ...


def _remaining_seconds(deadline: float, monotonic: MonotonicClock) -> float:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise WebContentCaptureError(
            WebContentCaptureErrorCode.WEB_URL_TIMEOUT,
            retryable=True,
        )
    return remaining


def _host_header(hostname: str, port: int) -> str:
    if port == 443:
        return hostname
    return f"{hostname}:{port}"


def _build_request_bytes(hostname: str, port: int, request_target: str) -> bytes:
    if not request_target.startswith("/"):
        request_target = f"/{request_target}"
    host_value = _host_header(hostname, port)
    lines = [
        f"GET {request_target} HTTP/1.1",
        f"Host: {host_value}",
        f"User-Agent: {_USER_AGENT}",
        f"Accept: {_ACCEPT}",
        "Accept-Encoding: identity",
        "Connection: close",
        "",
        "",
    ]
    return "\r\n".join(lines).encode("ascii")


def _framing_invalid() -> WebContentCaptureError:
    return WebContentCaptureError(
        WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID,
    )


def _parse_status_line(line: str) -> int:
    parts = line.split(" ", 2)
    if len(parts) < 2:
        raise _framing_invalid()
    version = parts[0]
    if version not in _ALLOWED_HTTP_VERSIONS:
        raise _framing_invalid()
    status_str = parts[1]
    if len(status_str) != 3 or not status_str.isdigit():
        raise _framing_invalid()
    status_code = int(status_str)
    if status_code < 100 or status_code > 599:
        raise _framing_invalid()
    return status_code


def _validate_header_name(name: str) -> None:
    if not name or name != name.strip():
        raise _framing_invalid()
    if not _HEADER_TOKEN_RE.match(name):
        raise _framing_invalid()


def _decode_chunk_size_line(line_bytes: bytes) -> str:
    try:
        return line_bytes[:-2].decode("ascii", errors="strict")
    except UnicodeDecodeError:
        raise _framing_invalid() from None


def _normalize_header_value(value: str) -> str:
    return " ".join(value.split())


def _validate_content_length_values(values: list[str]) -> str:
    normalized_values: list[str] = []
    for value in values:
        stripped = value.strip()
        if not stripped.isdigit():
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID,
            )
        parsed = int(stripped)
        if parsed < 0:
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID,
            )
        normalized_values.append(str(parsed))
    unique = set(normalized_values)
    if len(unique) > 1:
        raise WebContentCaptureError(
            WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID,
        )
    return normalized_values[0]


def _validate_content_encoding_values(values: list[str]) -> str:
    if len(values) > 1:
        raise WebContentCaptureError(
            WebContentCaptureErrorCode.WEB_URL_CONTENT_ENCODING_UNSUPPORTED,
        )
    normalized = [_normalize_header_value(value).lower() for value in values]
    encoding = normalized[0]
    if encoding and encoding != "identity":
        raise WebContentCaptureError(
            WebContentCaptureErrorCode.WEB_URL_CONTENT_ENCODING_UNSUPPORTED,
        )
    return encoding


def _validate_transfer_encoding_values(values: list[str]) -> str | None:
    if len(values) > 1:
        raise _framing_invalid()
    if not values:
        return None
    encoding = _normalize_header_value(values[0]).lower()
    if encoding != "chunked":
        raise _framing_invalid()
    return encoding


def _finalize_headers(header_entries: list[tuple[str, str]]) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for name, value in header_entries:
        grouped.setdefault(name.lower(), []).append(value)

    if "content-length" in grouped and "transfer-encoding" in grouped:
        raise WebContentCaptureError(
            WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID,
        )

    headers: dict[str, str] = {}
    for name, values in grouped.items():
        if name == "content-length":
            headers[name] = _validate_content_length_values(values)
        elif name == "location":
            if len(values) > 1:
                raise WebContentCaptureError(
                    WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID,
                )
            headers[name] = _normalize_header_value(values[0])
        elif name == "transfer-encoding":
            headers[name] = _validate_transfer_encoding_values(values) or ""
        elif name == "content-encoding":
            headers[name] = _validate_content_encoding_values(values)
        elif name == "content-type":
            if len(values) > 1:
                raise WebContentCaptureError(
                    WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID,
                )
            headers[name] = _normalize_header_value(values[0])
        else:
            headers[name] = _normalize_header_value(values[-1])
    return headers


def _recv_with_deadline(
    sock: socket.socket,
    size: int,
    *,
    deadline: float,
    monotonic: MonotonicClock,
    retryable: bool,
) -> bytes:
    sock.settimeout(_remaining_seconds(deadline, monotonic))
    try:
        return sock.recv(size)
    except socket.timeout:
        raise WebContentCaptureError(
            WebContentCaptureErrorCode.WEB_URL_TIMEOUT,
            retryable=retryable,
        )
    except OSError:
        raise WebContentCaptureError(
            WebContentCaptureErrorCode.WEB_URL_FETCH_FAILED,
            retryable=retryable,
        )


def _read_headers(
    sock: socket.socket,
    *,
    deadline: float,
    monotonic: MonotonicClock,
    retryable: bool,
) -> tuple[int, dict[str, str], bytes]:
    buffer = bytearray()
    while b"\r\n\r\n" not in buffer:
        chunk = _recv_with_deadline(
            sock,
            4096,
            deadline=deadline,
            monotonic=monotonic,
            retryable=retryable,
        )
        if not chunk:
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID,
            )
        buffer.extend(chunk)
        if len(buffer) > _MAX_HEADER_BYTES:
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID,
            )

    header_bytes, remainder = buffer.split(b"\r\n\r\n", 1)
    try:
        header_text = header_bytes.decode("iso-8859-1")
    except UnicodeDecodeError:
        raise WebContentCaptureError(
            WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID,
        )

    lines = header_text.split("\r\n")
    status_code = _parse_status_line(lines[0])
    header_entries: list[tuple[str, str]] = []
    for line in lines[1:]:
        if not line:
            continue
        if line[0] in " \t":
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID,
            )
        if ":" not in line:
            raise _framing_invalid()
        name, value = line.split(":", 1)
        _validate_header_name(name)
        header_entries.append((name.lower(), value))

    headers = _finalize_headers(header_entries)
    return status_code, headers, remainder


def _read_exact(
    sock: socket.socket,
    *,
    count: int,
    initial: bytes,
    deadline: float,
    monotonic: MonotonicClock,
    retryable: bool,
) -> bytes:
    body = bytearray(initial)
    while len(body) < count:
        chunk = _recv_with_deadline(
            sock,
            min(_READ_CHUNK_SIZE, count - len(body)),
            deadline=deadline,
            monotonic=monotonic,
            retryable=retryable,
        )
        if not chunk:
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID,
            )
        body.extend(chunk)
    return bytes(body[:count])


def _read_until_close(
    sock: socket.socket,
    *,
    initial: bytes,
    max_response_bytes: int,
    deadline: float,
    monotonic: MonotonicClock,
    retryable: bool,
) -> bytes:
    body = bytearray(initial)
    if len(body) > max_response_bytes:
        raise WebContentCaptureError(
            WebContentCaptureErrorCode.WEB_URL_CONTENT_TOO_LARGE,
        )
    while True:
        chunk = _recv_with_deadline(
            sock,
            _READ_CHUNK_SIZE,
            deadline=deadline,
            monotonic=monotonic,
            retryable=retryable,
        )
        if not chunk:
            break
        body.extend(chunk)
        if len(body) > max_response_bytes:
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_CONTENT_TOO_LARGE,
            )
    return bytes(body)


def _read_chunked_body(
    sock: socket.socket,
    *,
    initial: bytes,
    max_response_bytes: int,
    deadline: float,
    monotonic: MonotonicClock,
    retryable: bool,
) -> bytes:
    buffer = bytearray(initial)
    body = bytearray()

    def _ensure_bytes(count: int) -> bytes:
        nonlocal buffer
        while len(buffer) < count:
            chunk = _recv_with_deadline(
                sock,
                _READ_CHUNK_SIZE,
                deadline=deadline,
                monotonic=monotonic,
                retryable=retryable,
            )
            if not chunk:
                raise WebContentCaptureError(
                    WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID,
                )
            buffer.extend(chunk)
        extracted = bytes(buffer[:count])
        del buffer[:count]
        return extracted

    while True:
        line_bytes = bytearray()
        while True:
            if buffer:
                byte = buffer[0:1]
                del buffer[0:1]
            else:
                byte = _recv_with_deadline(
                    sock,
                    1,
                    deadline=deadline,
                    monotonic=monotonic,
                    retryable=retryable,
                )
                if not byte:
                    raise WebContentCaptureError(
                        WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID,
                    )
            line_bytes.extend(byte)
            if line_bytes.endswith(b"\r\n"):
                break
            if len(line_bytes) > 128:
                raise WebContentCaptureError(
                    WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID,
                )

        line = _decode_chunk_size_line(line_bytes)
        size_part = line.split(";", 1)[0].strip()
        if not size_part:
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID,
            )
        try:
            chunk_size = int(size_part, 16)
        except ValueError:
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID,
            )
        if chunk_size < 0:
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID,
            )

        if chunk_size == 0:
            trailer_total_bytes = 0
            while True:
                line_bytes = bytearray()
                while True:
                    if buffer:
                        byte = buffer[0:1]
                        del buffer[0:1]
                    else:
                        byte = _recv_with_deadline(
                            sock,
                            1,
                            deadline=deadline,
                            monotonic=monotonic,
                            retryable=retryable,
                        )
                        if not byte:
                            raise _framing_invalid()
                    line_bytes.extend(byte)
                    if line_bytes.endswith(b"\r\n"):
                        break
                    if len(line_bytes) > _MAX_TRAILER_BYTES:
                        raise _framing_invalid()
                trailer_total_bytes += len(line_bytes)
                if trailer_total_bytes > _MAX_TRAILER_BYTES:
                    raise _framing_invalid()
                if line_bytes == b"\r\n":
                    return bytes(body)
                trailer_line = line_bytes[:-2].decode("ascii", errors="strict")
                if trailer_line[0] in " \t":
                    raise _framing_invalid()
                if ":" not in trailer_line:
                    raise _framing_invalid()
                trailer_name, _trailer_value = trailer_line.split(":", 1)
                _validate_header_name(trailer_name)

        if chunk_size > max_response_bytes - len(body):
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_CONTENT_TOO_LARGE,
            )
        chunk_data = _ensure_bytes(chunk_size)
        crlf = _ensure_bytes(2)
        if crlf != b"\r\n":
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_RESPONSE_FRAMING_INVALID,
            )
        body.extend(chunk_data)
        if len(body) > max_response_bytes:
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_CONTENT_TOO_LARGE,
            )


def _read_body_bounded(
    sock: socket.socket,
    *,
    max_response_bytes: int,
    headers: Mapping[str, str],
    initial_body: bytes,
    deadline: float,
    monotonic: MonotonicClock,
    retryable: bool,
) -> bytes:
    transfer_encoding = headers.get("transfer-encoding")
    if transfer_encoding:
        return _read_chunked_body(
            sock,
            initial=initial_body,
            max_response_bytes=max_response_bytes,
            deadline=deadline,
            monotonic=monotonic,
            retryable=retryable,
        )

    if len(initial_body) > max_response_bytes:
        raise WebContentCaptureError(
            WebContentCaptureErrorCode.WEB_URL_CONTENT_TOO_LARGE,
        )

    content_length_raw = headers.get("content-length")
    if content_length_raw is not None:
        content_length = int(content_length_raw)
        if content_length > max_response_bytes:
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_CONTENT_TOO_LARGE,
            )
        return _read_exact(
            sock,
            count=content_length,
            initial=initial_body,
            deadline=deadline,
            monotonic=monotonic,
            retryable=retryable,
        )

    return _read_until_close(
        sock,
        initial=initial_body,
        max_response_bytes=max_response_bytes,
        deadline=deadline,
        monotonic=monotonic,
        retryable=retryable,
    )


def _sync_pinned_fetch(
    request: ApprovedHttpsRequest,
    connect_ip: str,
    connect_factory: ConnectFactory,
    monotonic: MonotonicClock,
) -> RawHttpsResponse:
    sock: socket.socket | None = None
    ssl_sock: ssl.SSLSocket | None = None
    headers_received = False
    try:
        timeout = _remaining_seconds(request.deadline, monotonic)
        sock = connect_factory(connect_ip, request.port, timeout)
        sock.settimeout(_remaining_seconds(request.deadline, monotonic))
        context = ssl.create_default_context()
        ssl_sock = context.wrap_socket(
            sock,
            server_hostname=request.hostname,
            do_handshake_on_connect=False,
        )
        ssl_sock.settimeout(_remaining_seconds(request.deadline, monotonic))
        ssl_sock.do_handshake()
        ssl_sock.settimeout(_remaining_seconds(request.deadline, monotonic))
        request_bytes = _build_request_bytes(
            request.hostname,
            request.port,
            request.request_target,
        )
        ssl_sock.sendall(request_bytes)
        status_code, headers, remainder = _read_headers(
            ssl_sock,
            deadline=request.deadline,
            monotonic=monotonic,
            retryable=True,
        )
        headers_received = True
        body = _read_body_bounded(
            ssl_sock,
            max_response_bytes=request.max_response_bytes,
            headers=headers,
            initial_body=remainder,
            deadline=request.deadline,
            monotonic=monotonic,
            retryable=False,
        )
        return RawHttpsResponse(
            status_code=status_code,
            headers=headers,
            body=body,
            content_bytes=len(body),
        )
    except WebContentCaptureError as exc:
        if headers_received and exc.retryable:
            raise WebContentCaptureError(exc.code, retryable=False) from None
        raise
    except socket.timeout:
        raise WebContentCaptureError(
            WebContentCaptureErrorCode.WEB_URL_TIMEOUT,
            retryable=not headers_received,
        )
    except ssl.SSLError:
        raise WebContentCaptureError(
            WebContentCaptureErrorCode.WEB_URL_TLS_FAILED,
            retryable=not headers_received,
        )
    except OSError:
        raise WebContentCaptureError(
            WebContentCaptureErrorCode.WEB_URL_FETCH_FAILED,
            retryable=not headers_received,
        )
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
        monotonic: MonotonicClock | None = None,
    ) -> None:
        self._connect_factory = connect_factory or _default_connect_factory
        self._fetch_count = fetch_count
        self._monotonic = monotonic or time.monotonic

    async def fetch(self, request: ApprovedHttpsRequest) -> RawHttpsResponse:
        last_error: WebContentCaptureError | None = None
        for connect_ip in request.approved_ips:
            if self._fetch_count is not None:
                self._fetch_count.append(connect_ip)
            remaining = request.deadline - self._monotonic()
            if remaining <= 0:
                raise WebContentCaptureError(
                    WebContentCaptureErrorCode.WEB_URL_TIMEOUT,
                    retryable=True,
                )
            try:
                return await asyncio.to_thread(
                    _sync_pinned_fetch,
                    request,
                    connect_ip,
                    self._connect_factory,
                    self._monotonic,
                )
            except WebContentCaptureError as exc:
                if exc.retryable and exc.code in _RETRYABLE_IP_ERRORS:
                    last_error = exc
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_FETCH_FAILED)

"""HTTP/HTTPS transport for proof data packages."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx

from intergrax.proof_data.errors import DataPackageTransportError
from intergrax.proof_data.transport.port import TransportDownloadResult

DEFAULT_CHUNK_SIZE_BYTES = 1024 * 1024
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 0.25
ALLOWED_SCHEMES = frozenset({"https", "http"})
NON_RETRYABLE_HTTP_STATUSES = frozenset({400, 401, 403, 404, 405, 410, 451})
_CONTENT_RANGE_PATTERN = re.compile(
    r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$",
    re.IGNORECASE,
)


class _PermanentTransportFailure(Exception):
    """Internal signal for deterministic HTTP failures that must not retry."""


@dataclass(frozen=True, slots=True)
class _AttemptPlan:
    offset: int
    mode: Literal["wb", "ab"]
    resumed: bool


class HttpDataPackageTransport:
    """Reference transport using streaming HTTP(S) with optional Range resume.

    Resume offset is derived from the current partial file size before every HTTP
    attempt. ``resume_from_byte`` is an initial contract hint only; the on-disk
    partial is authoritative and is never truncated to honor stale caller state.
    """

    def __init__(
        self,
        *,
        chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
        timeout_seconds: float = 60.0,
    ) -> None:
        if chunk_size_bytes <= 0:
            raise ValueError("chunk_size_bytes must be > 0")
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self._chunk_size_bytes = chunk_size_bytes
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._timeout_seconds = timeout_seconds

    def download_file(
        self,
        source_uri: str,
        destination_partial: Path,
        *,
        resume_from_byte: int = 0,
    ) -> TransportDownloadResult:
        parsed = urlparse(source_uri)
        if parsed.scheme not in ALLOWED_SCHEMES:
            raise DataPackageTransportError(
                "unsupported URI scheme for HTTP transport: "
                f"{parsed.scheme or '<missing>'}"
            )
        if not parsed.netloc:
            raise DataPackageTransportError(
                f"URI missing host: {_sanitize_uri_for_diagnostics(source_uri)}"
            )
        if resume_from_byte < 0:
            raise DataPackageTransportError("resume_from_byte must be >= 0")

        destination_partial.parent.mkdir(parents=True, exist_ok=True)

        total_bytes_written = 0
        resumed = False
        supports_range = False
        final_uri = source_uri
        reset_on_416_used = False
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            plan = _build_attempt_plan(destination_partial)
            size_before_attempt = _current_partial_size(destination_partial)

            try:
                attempt_bytes, attempt_resumed, attempt_supports_range, attempt_final_uri = (
                    self._stream_attempt(
                        source_uri,
                        destination_partial,
                        plan=plan,
                        reset_on_416_used=reset_on_416_used,
                    )
                )
            except _PermanentTransportFailure as exc:
                raise DataPackageTransportError(str(exc)) from exc
            except httpx.TransportError as exc:
                persisted_bytes = max(
                    0,
                    _current_partial_size(destination_partial) - size_before_attempt,
                )
                total_bytes_written += persisted_bytes
                last_error = exc
                if attempt >= self._max_retries:
                    break
                time.sleep(self._retry_backoff_seconds * attempt)
                continue
            except DataPackageTransportError:
                raise

            if attempt_bytes is None:
                reset_on_416_used = True
                if attempt >= self._max_retries:
                    last_error = DataPackageTransportError(
                        "range restart required but retry budget exhausted"
                    )
                    break
                time.sleep(self._retry_backoff_seconds * attempt)
                continue

            total_bytes_written += attempt_bytes
            if attempt_resumed:
                resumed = True
            if attempt_supports_range:
                supports_range = True
            final_uri = attempt_final_uri
            return TransportDownloadResult(
                bytes_written=total_bytes_written,
                resumed=resumed,
                supports_range=supports_range,
                final_uri=final_uri,
            )

        sanitized = _sanitize_uri_for_diagnostics(source_uri)
        raise DataPackageTransportError(
            f"download failed after {self._max_retries} attempts for {sanitized}: "
            f"{last_error}"
        )

    def _stream_attempt(
        self,
        source_uri: str,
        destination_partial: Path,
        *,
        plan: _AttemptPlan,
        reset_on_416_used: bool,
    ) -> tuple[int | None, bool, bool, str]:
        """Execute one HTTP attempt.

        Returns ``(None, False, False, uri)`` when a bounded 416 restart is required.
        """
        headers = _build_range_headers(plan.offset)
        supports_range = False
        attempt_resumed = plan.resumed

        with httpx.Client(
            timeout=self._timeout_seconds,
            follow_redirects=True,
            event_hooks={"response": [_reject_unsafe_redirect_scheme]},
        ) as client:
            try:
                with client.stream("GET", source_uri, headers=headers) as response:
                    final_uri = str(response.url)
                    sanitized = _sanitize_uri_for_diagnostics(final_uri)

                    if response.status_code == 416:
                        if plan.offset > 0:
                            if reset_on_416_used:
                                raise _PermanentTransportFailure(
                                    f"HTTP 416 for {sanitized} after partial reset"
                                )
                            destination_partial.unlink(missing_ok=True)
                            return None, False, False, final_uri
                        raise _PermanentTransportFailure(
                            f"HTTP 416 for {sanitized}"
                        )

                    if (
                        plan.offset > 0
                        and response.status_code not in {206, 200}
                    ):
                        if response.status_code in NON_RETRYABLE_HTTP_STATUSES:
                            raise _PermanentTransportFailure(
                                f"HTTP {response.status_code} for {sanitized}"
                            )
                        raise DataPackageTransportError(
                            f"HTTP {response.status_code} for {sanitized}"
                        )

                    write_mode = plan.mode
                    if plan.offset > 0 and response.status_code == 206:
                        supports_range = True
                        _validate_range_response(
                            plan.offset,
                            response.headers.get("Content-Range"),
                            sanitized,
                        )
                    elif plan.offset > 0 and response.status_code == 200:
                        destination_partial.unlink(missing_ok=True)
                        write_mode = "wb"
                        attempt_resumed = False

                    if response.status_code >= 400:
                        if response.status_code in NON_RETRYABLE_HTTP_STATUSES:
                            raise _PermanentTransportFailure(
                                f"HTTP {response.status_code} for {sanitized}"
                            )
                        raise DataPackageTransportError(
                            f"HTTP {response.status_code} for {sanitized}"
                        )

                    bytes_written = 0
                    with destination_partial.open(write_mode) as handle:
                        try:
                            for chunk in response.iter_bytes(self._chunk_size_bytes):
                                if not chunk:
                                    continue
                                handle.write(chunk)
                                bytes_written += len(chunk)
                        finally:
                            handle.flush()

                    return bytes_written, attempt_resumed, supports_range, final_uri
            except httpx.TransportError:
                raise
            except _PermanentTransportFailure:
                raise
            except DataPackageTransportError:
                raise

        raise DataPackageTransportError("download attempt ended without response")


def _current_partial_size(destination_partial: Path) -> int:
    if destination_partial.is_file():
        return destination_partial.stat().st_size
    return 0


def _build_attempt_plan(destination_partial: Path) -> _AttemptPlan:
    offset = _current_partial_size(destination_partial)
    if offset > 0:
        return _AttemptPlan(offset=offset, mode="ab", resumed=True)
    return _AttemptPlan(offset=0, mode="wb", resumed=False)


def _build_range_headers(offset: int) -> dict[str, str]:
    if offset <= 0:
        return {}
    return {"Range": f"bytes={offset}-"}


def _validate_range_response(
    requested_offset: int,
    content_range: str | None,
    sanitized_uri: str,
) -> None:
    if content_range is None:
        raise _PermanentTransportFailure(
            f"HTTP 206 missing Content-Range for {sanitized_uri}"
        )
    parsed = _parse_content_range(content_range)
    if parsed is None:
        raise _PermanentTransportFailure(
            f"malformed Content-Range for {sanitized_uri}: {content_range}"
        )
    start, _end, _total = parsed
    if start != requested_offset:
        raise _PermanentTransportFailure(
            f"Content-Range start {start} does not match requested offset "
            f"{requested_offset} for {sanitized_uri}"
        )


def _parse_content_range(value: str) -> tuple[int, int, int | None] | None:
    match = _CONTENT_RANGE_PATTERN.match(value.strip())
    if match is None:
        return None
    start = int(match.group(1))
    end = int(match.group(2))
    total_raw = match.group(3)
    total = None if total_raw == "*" else int(total_raw)
    return start, end, total


def _sanitize_uri_for_diagnostics(uri: str) -> str:
    parsed = urlparse(uri)
    path = parsed.path or "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _reject_unsafe_redirect_scheme(response: httpx.Response) -> None:
    parsed = urlparse(str(response.url))
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise DataPackageTransportError(
            f"redirect target uses unsupported scheme: {parsed.scheme or '<missing>'}"
        )

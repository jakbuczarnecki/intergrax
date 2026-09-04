"""HTTP/HTTPS transport for proof data packages."""

from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from intergrax.proof_data.errors import DataPackageTransportError
from intergrax.proof_data.transport.port import TransportDownloadResult

DEFAULT_CHUNK_SIZE_BYTES = 1024 * 1024
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 0.25
ALLOWED_SCHEMES = frozenset({"https", "http"})


class HttpDataPackageTransport:
    """Reference transport using streaming HTTP(S) with optional Range resume."""

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
                f"unsupported URI scheme for HTTP transport: {parsed.scheme or '<missing>'}"
            )
        if not parsed.netloc:
            raise DataPackageTransportError(f"URI missing host: {source_uri}")
        if resume_from_byte < 0:
            raise DataPackageTransportError("resume_from_byte must be >= 0")

        destination_partial.parent.mkdir(parents=True, exist_ok=True)
        attempt = 0
        last_error: Exception | None = None
        while attempt < self._max_retries:
            attempt += 1
            try:
                return self._download_once(
                    source_uri,
                    destination_partial,
                    resume_from_byte=resume_from_byte,
                )
            except DataPackageTransportError as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    break
                time.sleep(self._retry_backoff_seconds * attempt)
        raise DataPackageTransportError(
            f"download failed after {self._max_retries} attempts for {source_uri}: {last_error}"
        )

    def _download_once(
        self,
        source_uri: str,
        destination_partial: Path,
        *,
        resume_from_byte: int,
    ) -> TransportDownloadResult:
        headers: dict[str, str] = {}
        mode = "wb"
        resumed = False
        supports_range = False
        if resume_from_byte > 0:
            headers["Range"] = f"bytes={resume_from_byte}-"
            mode = "ab"
            resumed = True

        with httpx.Client(
            timeout=self._timeout_seconds,
            follow_redirects=True,
            event_hooks={"response": [_reject_unsafe_redirect_scheme]},
        ) as client:
            with client.stream("GET", source_uri, headers=headers) as response:
                final_uri = str(response.url)
                if resume_from_byte > 0 and response.status_code == 416:
                    destination_partial.unlink(missing_ok=True)
                    return self._download_once(
                        source_uri,
                        destination_partial,
                        resume_from_byte=0,
                    )
                if resume_from_byte > 0 and response.status_code not in {206, 200}:
                    destination_partial.unlink(missing_ok=True)
                    return self._download_once(
                        source_uri,
                        destination_partial,
                        resume_from_byte=0,
                    )
                if resume_from_byte > 0 and response.status_code == 206:
                    supports_range = True
                elif resume_from_byte > 0 and response.status_code == 200:
                    destination_partial.unlink(missing_ok=True)
                    mode = "wb"
                    resumed = False

                if response.status_code >= 400:
                    raise DataPackageTransportError(
                        f"HTTP {response.status_code} for {source_uri}"
                    )

                bytes_written = 0
                with destination_partial.open(mode) as handle:
                    for chunk in response.iter_bytes(self._chunk_size_bytes):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        bytes_written += len(chunk)

        return TransportDownloadResult(
            bytes_written=bytes_written,
            resumed=resumed,
            supports_range=supports_range,
            final_uri=final_uri,
        )


def _reject_unsafe_redirect_scheme(response: httpx.Response) -> None:
    parsed = urlparse(str(response.url))
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise DataPackageTransportError(
            f"redirect target uses unsupported scheme: {parsed.scheme or '<missing>'}"
        )

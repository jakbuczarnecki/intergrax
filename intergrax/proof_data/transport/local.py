"""Local filesystem transport for tests and trusted local mirrors."""

from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import unquote, urlparse

from intergrax.proof_data.errors import DataPackageTransportError
from intergrax.proof_data.transport.port import TransportDownloadResult

DEFAULT_CHUNK_SIZE_BYTES = 1024 * 1024


class LocalFileDataPackageTransport:
    """Copy bytes from ``file://`` URIs or absolute local paths."""

    def __init__(self, *, chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES) -> None:
        if chunk_size_bytes <= 0:
            raise ValueError("chunk_size_bytes must be > 0")
        self._chunk_size_bytes = chunk_size_bytes

    def download_file(
        self,
        source_uri: str,
        destination_partial: Path,
        *,
        resume_from_byte: int = 0,
    ) -> TransportDownloadResult:
        source_path = _resolve_local_source(source_uri)
        if not source_path.is_file():
            raise DataPackageTransportError(f"local source file missing: {source_path}")

        destination_partial.parent.mkdir(parents=True, exist_ok=True)
        source_size = source_path.stat().st_size
        if resume_from_byte > source_size:
            resume_from_byte = 0

        bytes_written = 0
        mode = "wb" if resume_from_byte == 0 else "ab"
        with source_path.open("rb") as source_handle, destination_partial.open(mode) as dest_handle:
            if resume_from_byte > 0:
                source_handle.seek(resume_from_byte)
            while True:
                chunk = source_handle.read(self._chunk_size_bytes)
                if not chunk:
                    break
                dest_handle.write(chunk)
                bytes_written += len(chunk)

        return TransportDownloadResult(
            bytes_written=bytes_written,
            resumed=resume_from_byte > 0,
            supports_range=True,
            final_uri=str(source_path),
        )


def _resolve_local_source(source_uri: str) -> Path:
    parsed = urlparse(source_uri)
    if parsed.scheme == "file":
        raw_path = unquote(parsed.path)
        if raw_path.startswith("/") and len(raw_path) > 2 and raw_path[2] == ":":
            raw_path = raw_path.lstrip("/")
        return Path(raw_path)
    if parsed.scheme in {"", None}:
        return Path(source_uri)
    if parsed.scheme in {"https", "http"}:
        raise DataPackageTransportError(
            "LocalFileDataPackageTransport does not support remote HTTP URIs"
        )
    raise DataPackageTransportError(f"unsupported local transport URI: {source_uri}")

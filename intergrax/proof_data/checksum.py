"""Streaming SHA256 verification for proof data packages."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from intergrax.proof_data.errors import DataPackageIntegrityError

SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_CHUNK_SIZE_BYTES = 1024 * 1024


def normalize_sha256_hex(value: str) -> str:
    normalized = value.strip().lower()
    if not SHA256_HEX_PATTERN.match(normalized):
        msg = "sha256 must be exactly 64 lowercase hexadecimal characters"
        raise DataPackageIntegrityError(msg)
    return normalized


def sha256_file(path: Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE_BYTES) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_integrity(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE_BYTES,
) -> None:
    expected = normalize_sha256_hex(expected_sha256)
    if not path.is_file():
        raise DataPackageIntegrityError(f"expected file is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected_size_bytes:
        raise DataPackageIntegrityError(
            f"size mismatch for {path.name}: expected {expected_size_bytes}, got {actual_size}"
        )
    actual = sha256_file(path, chunk_size=chunk_size)
    if actual != expected:
        raise DataPackageIntegrityError(
            f"checksum mismatch for {path.name}: expected {expected}, got {actual}"
        )

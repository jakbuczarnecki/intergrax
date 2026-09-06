"""SHA-256 checksum helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackIntegrityError,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256sums(path: Path, entries: tuple[tuple[str, Path], ...]) -> None:
    lines: list[str] = []
    for relative_name, file_path in entries:
        if not file_path.is_file():
            raise VpiDataPackIntegrityError(f"checksum target missing: {file_path}")
        checksum = sha256_file(file_path)
        lines.append(f"{checksum}  {relative_name}")
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temp_path.replace(path)


def verify_sha256sums(path: Path, root: Path) -> None:
    if not path.is_file():
        raise VpiDataPackIntegrityError(f"checksum file missing: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise VpiDataPackIntegrityError(f"invalid checksum line: {line}")
        expected_checksum, relative_name = parts[0], parts[1].strip()
        target = root / relative_name
        actual = sha256_file(target)
        if actual != expected_checksum:
            raise VpiDataPackIntegrityError(
                f"checksum mismatch for {relative_name}: expected {expected_checksum}, got {actual}"
            )

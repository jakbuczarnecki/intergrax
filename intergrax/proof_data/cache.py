"""Content-addressable cache for proof data package files."""

from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from intergrax.proof_data.checksum import normalize_sha256_hex, verify_file_integrity
from intergrax.proof_data.errors import DataPackageIntegrityError

PARTIAL_SUFFIX = ".part"


@dataclass(frozen=True, slots=True)
class CacheLookupResult:
    cache_path: Path
    hit: bool


class DataPackageCache:
    """Checksum-addressed local object cache."""

    def __init__(self, root_dir: Path) -> None:
        self._root_dir = root_dir.resolve()
        self._objects_dir = self._root_dir / "objects"

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    def object_path(self, sha256_hex: str) -> Path:
        normalized = normalize_sha256_hex(sha256_hex)
        prefix = normalized[:2]
        return self._objects_dir / prefix / normalized

    def partial_path(self, sha256_hex: str) -> Path:
        return self.object_path(sha256_hex).with_suffix(self.object_path(sha256_hex).suffix + PARTIAL_SUFFIX)

    def lookup(self, sha256_hex: str, *, expected_size_bytes: int) -> CacheLookupResult:
        cache_path = self.object_path(sha256_hex)
        if not cache_path.is_file():
            return CacheLookupResult(cache_path=cache_path, hit=False)
        try:
            verify_file_integrity(
                cache_path,
                expected_sha256=sha256_hex,
                expected_size_bytes=expected_size_bytes,
            )
        except DataPackageIntegrityError:
            self.quarantine(cache_path)
            return CacheLookupResult(cache_path=cache_path, hit=False)
        return CacheLookupResult(cache_path=cache_path, hit=True)

    def publish_partial(
        self,
        partial_path: Path,
        *,
        sha256_hex: str,
        expected_size_bytes: int,
    ) -> Path:
        cache_path = self.object_path(sha256_hex)
        verify_file_integrity(
            partial_path,
            expected_sha256=sha256_hex,
            expected_size_bytes=expected_size_bytes,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_target = cache_path.with_name(f"{cache_path.name}.{uuid.uuid4().hex}.tmp")
        shutil.copyfile(partial_path, temp_target)
        os.replace(temp_target, cache_path)
        return cache_path

    def materialize_to(
        self,
        *,
        sha256_hex: str,
        expected_size_bytes: int,
        destination: Path,
    ) -> Path:
        lookup = self.lookup(sha256_hex, expected_size_bytes=expected_size_bytes)
        if not lookup.hit:
            raise DataPackageIntegrityError(
                f"cache miss for sha256={sha256_hex}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_target = destination.with_name(f"{destination.name}.{uuid.uuid4().hex}.tmp")
        shutil.copyfile(lookup.cache_path, temp_target)
        os.replace(temp_target, destination)
        return destination

    def quarantine(self, path: Path) -> None:
        if not path.exists():
            return
        invalid_path = path.with_suffix(path.suffix + ".invalid")
        if invalid_path.exists():
            invalid_path.unlink()
        os.replace(path, invalid_path)

    def remove_partial(self, sha256_hex: str) -> None:
        partial_path = self.partial_path(sha256_hex)
        partial_path.unlink(missing_ok=True)

    def existing_partial_size(self, sha256_hex: str) -> int:
        partial_path = self.partial_path(sha256_hex)
        if not partial_path.is_file():
            return 0
        return partial_path.stat().st_size

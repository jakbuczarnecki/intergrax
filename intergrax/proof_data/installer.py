"""Proof data package installation orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

from intergrax.proof_data.cache import DataPackageCache
from intergrax.proof_data.checksum import verify_file_integrity
from intergrax.proof_data.descriptor import DataPackageFileDescriptor, ProofDataPackageDescriptor
from intergrax.proof_data.errors import DataPackageInstallError, DataPackageIntegrityError
from intergrax.proof_data.paths import resolve_under_root
from intergrax.proof_data.report import DataPackageInstallReport
from intergrax.proof_data.transport.port import DataPackageTransportPort


@dataclass(frozen=True, slots=True)
class DataPackageInstallRequest:
    descriptor: ProofDataPackageDescriptor
    install_root: Path
    cache: DataPackageCache
    transport: DataPackageTransportPort
    base_uri: str | None = None


class DataPackageInstaller:
    """Validate descriptors, resolve files, cache, download, and verify."""

    def install(self, request: DataPackageInstallRequest) -> DataPackageInstallReport:
        started = time.perf_counter()
        install_root = request.install_root.resolve()
        install_root.mkdir(parents=True, exist_ok=True)

        files_downloaded = 0
        files_reused_from_cache = 0
        files_installed_from_existing = 0
        bytes_downloaded = 0
        bytes_reused = 0

        for file_descriptor in request.descriptor.files:
            destination = resolve_under_root(install_root, file_descriptor.relative_path)
            if self._installed_file_is_valid(destination, file_descriptor):
                files_installed_from_existing += 1
                bytes_reused += file_descriptor.size_bytes
                continue

            cache_lookup = request.cache.lookup(
                file_descriptor.sha256,
                expected_size_bytes=file_descriptor.size_bytes,
            )
            if cache_lookup.hit:
                request.cache.materialize_to(
                    sha256_hex=file_descriptor.sha256,
                    expected_size_bytes=file_descriptor.size_bytes,
                    destination=destination,
                )
                files_reused_from_cache += 1
                bytes_reused += file_descriptor.size_bytes
                continue

            source_uri = self._resolve_source_uri(
                request.base_uri,
                file_descriptor.relative_path,
            )
            partial_path = request.cache.partial_path(file_descriptor.sha256)
            resume_from = request.cache.existing_partial_size(file_descriptor.sha256)
            result = request.transport.download_file(
                source_uri,
                partial_path,
                resume_from_byte=resume_from,
            )
            bytes_downloaded += result.bytes_written
            files_downloaded += 1
            try:
                request.cache.publish_partial(
                    partial_path,
                    sha256_hex=file_descriptor.sha256,
                    expected_size_bytes=file_descriptor.size_bytes,
                )
                request.cache.materialize_to(
                    sha256_hex=file_descriptor.sha256,
                    expected_size_bytes=file_descriptor.size_bytes,
                    destination=destination,
                )
            finally:
                request.cache.remove_partial(file_descriptor.sha256)

        elapsed = time.perf_counter() - started
        return DataPackageInstallReport(
            package_id=request.descriptor.package_id,
            package_version=request.descriptor.package_version,
            files_total=len(request.descriptor.files),
            files_downloaded=files_downloaded,
            files_reused_from_cache=files_reused_from_cache,
            files_installed_from_existing=files_installed_from_existing,
            bytes_downloaded=bytes_downloaded,
            bytes_reused=bytes_reused,
            verification_passed=True,
            install_location=install_root,
            elapsed_seconds=elapsed,
        )

    def _installed_file_is_valid(
        self,
        destination: Path,
        file_descriptor: DataPackageFileDescriptor,
    ) -> bool:
        if not destination.is_file():
            return False
        try:
            verify_file_integrity(
                destination,
                expected_sha256=file_descriptor.sha256,
                expected_size_bytes=file_descriptor.size_bytes,
            )
        except DataPackageIntegrityError:
            destination.unlink(missing_ok=True)
            return False
        return True

    def _resolve_source_uri(self, base_uri: str | None, relative_path: str) -> str:
        if base_uri is None or not base_uri.strip():
            raise DataPackageInstallError(
                "base_uri is required to download package files; "
                "install from a local trusted mirror or configure distribution base URI"
            )
        normalized_base = base_uri.strip()
        if not normalized_base.endswith("/"):
            normalized_base = f"{normalized_base}/"
        return urljoin(normalized_base, relative_path)

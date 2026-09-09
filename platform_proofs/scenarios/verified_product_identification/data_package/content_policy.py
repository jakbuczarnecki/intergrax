"""VPI Data Pack distribution content policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from intergrax.proof_data.checksum import sha256_file

from platform_proofs.scenarios.verified_product_identification.data_package.errors import (
    VpiDataPackageDescriptorBuildError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    read_manifest_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    resolve_data_pack_paths,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.status import (
    DataPackStatus,
)

DISTRIBUTION_STRATEGY = "FILE_PER_SHARD"
BUILD_STATE_EXCLUSION_REASON = (
    "build-state.json is operational resumable-build provenance; consumers require "
    "manifest, shard indexes, checksums, and proof evidence only."
)
EXCLUDED_RELATIVE_PATHS = frozenset({"state/build-state.json"})


class DistributableFileRole(StrEnum):
    MANIFEST = "MANIFEST"
    RELATIONAL_SHARD = "RELATIONAL_SHARD"
    EMBEDDING_SHARD = "EMBEDDING_SHARD"
    SHARD_INDEX = "SHARD_INDEX"
    CHECKSUMS = "CHECKSUMS"
    PROOF_REPORT = "PROOF_REPORT"


@dataclass(frozen=True, slots=True)
class DistributableFile:
    relative_path: str
    role: DistributableFileRole
    path: Path
    size_bytes: int
    sha256: str


def role_for_relative_path(relative_path: str) -> DistributableFileRole:
    if relative_path == "manifest/manifest.json":
        return DistributableFileRole.MANIFEST
    if relative_path == "indexes/shards.json":
        return DistributableFileRole.SHARD_INDEX
    if relative_path == "checksums/SHA256SUMS":
        return DistributableFileRole.CHECKSUMS
    if relative_path == "evidence/proof-report.json":
        return DistributableFileRole.PROOF_REPORT
    if relative_path.startswith("relational/") and relative_path.endswith(".parquet"):
        return DistributableFileRole.RELATIONAL_SHARD
    if relative_path.startswith("embeddings/") and relative_path.endswith(".parquet"):
        return DistributableFileRole.EMBEDDING_SHARD
    raise VpiDataPackageDescriptorBuildError(
        f"unsupported distributable relative path: {relative_path}"
    )


def _parse_sha256sums(checksums_path: Path) -> tuple[tuple[str, str], ...]:
    if not checksums_path.is_file():
        raise VpiDataPackageDescriptorBuildError(f"checksum file missing: {checksums_path}")
    entries: list[tuple[str, str]] = []
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise VpiDataPackageDescriptorBuildError(f"invalid checksum line: {line}")
        checksum, relative_path = parts[0].strip(), parts[1].strip()
        entries.append((relative_path, checksum))
    return tuple(entries)


def _file_descriptor(
    artifact_root: Path,
    relative_path: str,
    expected_sha256: str | None = None,
) -> DistributableFile:
    if relative_path in EXCLUDED_RELATIVE_PATHS:
        raise VpiDataPackageDescriptorBuildError(
            f"excluded path cannot be distributed: {relative_path}"
        )
    file_path = artifact_root / relative_path
    if not file_path.is_file():
        raise VpiDataPackageDescriptorBuildError(f"missing distributable file: {relative_path}")
    actual_sha256 = sha256_file(file_path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise VpiDataPackageDescriptorBuildError(
            f"checksum mismatch for {relative_path}: expected {expected_sha256}, got {actual_sha256}"
        )
    return DistributableFile(
        relative_path=relative_path,
        role=role_for_relative_path(relative_path),
        path=file_path,
        size_bytes=file_path.stat().st_size,
        sha256=actual_sha256,
    )


def assert_artifact_finalized(artifact_root: Path) -> None:
    paths = resolve_data_pack_paths(artifact_root)
    if not paths.manifest_file.is_file():
        raise VpiDataPackageDescriptorBuildError("manifest/manifest.json is required")
    manifest = read_manifest_file(paths.manifest_file)
    if manifest.status is not DataPackStatus.READY:
        raise VpiDataPackageDescriptorBuildError(
            f"data pack status must be READY, got {manifest.status.value}"
        )


def collect_distributable_files(artifact_root: Path) -> tuple[DistributableFile, ...]:
    """Collect immutable per-file distribution entries from a finalized Data Pack."""
    root = artifact_root.resolve()
    paths = resolve_data_pack_paths(root)
    assert_artifact_finalized(root)

    collected: dict[str, DistributableFile] = {}
    for relative_path, expected_sha256 in _parse_sha256sums(paths.checksums_file):
        if relative_path in EXCLUDED_RELATIVE_PATHS:
            continue
        collected[relative_path] = _file_descriptor(
            root,
            relative_path,
            expected_sha256=expected_sha256,
        )

    checksums_relative = "checksums/SHA256SUMS"
    if checksums_relative not in collected:
        collected[checksums_relative] = _file_descriptor(root, checksums_relative)

    proof_report_relative = "evidence/proof-report.json"
    if proof_report_relative not in collected:
        if not paths.proof_report_file.is_file():
            raise VpiDataPackageDescriptorBuildError(
                f"missing required proof report: {proof_report_relative}"
            )
        collected[proof_report_relative] = _file_descriptor(root, proof_report_relative)

    return tuple(collected[path] for path in sorted(collected))

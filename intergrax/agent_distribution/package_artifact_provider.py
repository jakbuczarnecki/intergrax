# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Lock-authoritative package artifact resolution for physical materialization (AP-8)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from intergrax.agent_distribution.errors import (
    MaterializationError,
    MaterializationLockArtifactLocationBlocked,
)
from intergrax.agent_distribution.stores import AgentArtifactMetadataStore

ArtifactKind = Literal["wheel", "source_bundle"]


@dataclass(frozen=True)
class ResolvedPackageArtifact:
    """Stageable immutable package bytes resolved from distribution authority."""

    distribution_package_id: str
    package_digest: str
    local_source_path: Path
    artifact_kind: ArtifactKind = "wheel"


class ArtifactStoreRefResolver(Protocol):
    """Resolve durable ``artifact_store_ref`` into a local readable artifact path."""

    def resolve_local_path(self, artifact_store_ref: str) -> Path:
        """Return a host-local path to immutable package artifact bytes."""


class PackageArtifactProvider(Protocol):
    """Resolve lock-identified package artifacts for physical materialization."""

    def resolve_artifact(
        self,
        distribution_package_id: str,
        package_digest: str,
    ) -> ResolvedPackageArtifact:
        """Return stageable artifact bytes for one lock-authorized package."""


def sha256_file_digest(path: Path) -> str:
    """Return canonical ``sha256:<hex>`` digest for artifact file bytes."""
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return f"sha256:{hasher.hexdigest()}"


def verify_artifact_file_digest(path: Path, expected_digest: str) -> None:
    """Fail closed when staged artifact bytes do not match lock digest."""
    actual = sha256_file_digest(path)
    if actual != expected_digest.strip().lower():
        raise MaterializationError(
            f"artifact digest mismatch for {path.name}: expected {expected_digest}, got {actual}"
        )


@dataclass(frozen=True)
class FilesystemArtifactStoreRefResolver:
    """Explicit adapter for ``file://`` artifact store references (tests/dev only)."""

    root: Path

    def resolve_local_path(self, artifact_store_ref: str) -> Path:
        ref = artifact_store_ref.strip()
        if not ref.startswith("file://"):
            raise MaterializationError(
                f"unsupported artifact_store_ref scheme for filesystem resolver: {ref}"
            )
        rel = ref.removeprefix("file://").lstrip("/")
        if not rel or ".." in Path(rel).parts:
            raise MaterializationError(f"artifact_store_ref path rejected: {ref}")
        resolved = (self.root / rel).resolve()
        try:
            resolved.relative_to(self.root.resolve())
        except ValueError as exc:
            raise MaterializationError(
                f"artifact_store_ref escapes resolver root: {ref}"
            ) from exc
        if not resolved.is_file():
            raise MaterializationLockArtifactLocationBlocked(
                MaterializationLockArtifactLocationBlocked.BLOCKER_CODE
                + f": missing artifact bytes at {ref}"
            )
        return resolved


@dataclass(frozen=True)
class MetadataBackedPackageArtifactProvider:
    """Resolve artifacts via ``AgentArtifactMetadata`` + explicit ref resolver."""

    metadata_store: AgentArtifactMetadataStore
    ref_resolver: ArtifactStoreRefResolver

    def resolve_artifact(
        self,
        distribution_package_id: str,
        package_digest: str,
    ) -> ResolvedPackageArtifact:
        metadata = self.metadata_store.get_by_digest(package_digest)
        if metadata is None:
            raise MaterializationLockArtifactLocationBlocked(
                MaterializationLockArtifactLocationBlocked.BLOCKER_CODE
                + f": no artifact metadata for digest {package_digest}"
            )
        if metadata.distribution_package_id != distribution_package_id:
            raise MaterializationError(
                "artifact metadata distribution_package_id mismatch for "
                f"{distribution_package_id}"
            )
        if metadata.tombstoned:
            raise MaterializationError(
                f"artifact metadata tombstoned for {distribution_package_id}"
            )
        local_path = self.ref_resolver.resolve_local_path(metadata.artifact_store_ref)
        verify_artifact_file_digest(local_path, package_digest)
        suffix = local_path.suffix.lower()
        if suffix == ".whl":
            kind: ArtifactKind = "wheel"
        elif suffix in {".tar.gz", ".zip"}:
            kind = "source_bundle"
        else:
            kind = "wheel"
        return ResolvedPackageArtifact(
            distribution_package_id=distribution_package_id,
            package_digest=package_digest,
            local_source_path=local_path,
            artifact_kind=kind,
        )

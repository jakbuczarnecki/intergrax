"""Embedding artifact compatibility identity — fail closed on drift."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.errors import (
    ArtifactCompatibilityError,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.model import (
    EmbeddingArtifactManifest,
)


@dataclass(frozen=True, slots=True)
class EmbeddingArtifactCompatibilityIdentity:
    dataset_checksum: str
    dataset_record_count: int
    search_representation_derivation_version: str
    embedding_configuration_version: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    artifact_schema_version: str
    catalog_id: str


def compatibility_identity_from_manifest(
    manifest: EmbeddingArtifactManifest,
) -> EmbeddingArtifactCompatibilityIdentity:
    return EmbeddingArtifactCompatibilityIdentity(
        dataset_checksum=manifest.dataset_checksum,
        dataset_record_count=manifest.dataset_record_count,
        search_representation_derivation_version=manifest.search_representation_derivation_version,
        embedding_configuration_version=manifest.embedding_configuration_version,
        embedding_provider=manifest.embedding_provider,
        embedding_model=manifest.embedding_model,
        embedding_dimension=manifest.embedding_dimension,
        artifact_schema_version=manifest.artifact_schema_version,
        catalog_id=manifest.catalog_id,
    )


def assert_manifest_compatible(
    *,
    existing: EmbeddingArtifactManifest,
    expected: EmbeddingArtifactCompatibilityIdentity,
) -> None:
    mismatches: list[str] = []
    if existing.dataset_checksum != expected.dataset_checksum:
        mismatches.append("dataset_checksum")
    if existing.dataset_record_count != expected.dataset_record_count:
        mismatches.append("dataset_record_count")
    if existing.search_representation_derivation_version != expected.search_representation_derivation_version:
        mismatches.append("search_representation_derivation_version")
    if existing.embedding_configuration_version != expected.embedding_configuration_version:
        mismatches.append("embedding_configuration_version")
    if existing.embedding_provider != expected.embedding_provider:
        mismatches.append("embedding_provider")
    if existing.embedding_model != expected.embedding_model:
        mismatches.append("embedding_model")
    if existing.embedding_dimension != expected.embedding_dimension:
        mismatches.append("embedding_dimension")
    if existing.artifact_schema_version != expected.artifact_schema_version:
        mismatches.append("artifact_schema_version")
    if existing.catalog_id != expected.catalog_id:
        mismatches.append("catalog_id")
    if mismatches:
        joined = ", ".join(mismatches)
        raise ArtifactCompatibilityError(
            f"existing embedding artifact manifest is incompatible ({joined}); "
            "explicit rebuild in a new artifact location is required"
        )


def artifact_directory_fingerprint(identity: EmbeddingArtifactCompatibilityIdentity) -> str:
    """Short deterministic fingerprint for artifact directory naming."""
    canonical = (
        f"{identity.dataset_checksum}|"
        f"{identity.dataset_record_count}|"
        f"{identity.search_representation_derivation_version}|"
        f"{identity.embedding_configuration_version}|"
        f"{identity.embedding_provider}|"
        f"{identity.embedding_model}|"
        f"{identity.embedding_dimension}|"
        f"{identity.artifact_schema_version}|"
        f"{identity.catalog_id}"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

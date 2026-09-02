"""Manifest compatibility rules — fail closed on identity drift."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.errors import (
    VpiBootstrapCompatibilityError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.model import (
    VpiBootstrapManifest,
)


@dataclass(frozen=True, slots=True)
class BootstrapCompatibilityIdentity:
    dataset_checksum: str
    dataset_record_count: int
    search_representation_derivation_version: str
    embedding_configuration_version: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    catalog_schema_version: str
    search_index_schema_version: str
    bootstrap_implementation_version: str
    catalog_id: str


def compatibility_identity_from_manifest(manifest: VpiBootstrapManifest) -> BootstrapCompatibilityIdentity:
    return BootstrapCompatibilityIdentity(
        dataset_checksum=manifest.dataset_checksum,
        dataset_record_count=manifest.dataset_record_count,
        search_representation_derivation_version=manifest.search_representation_derivation_version,
        embedding_configuration_version=manifest.embedding_configuration_version,
        embedding_provider=manifest.embedding_provider,
        embedding_model=manifest.embedding_model,
        embedding_dimension=manifest.embedding_dimension,
        catalog_schema_version=manifest.catalog_schema_version,
        search_index_schema_version=manifest.search_index_schema_version,
        bootstrap_implementation_version=manifest.bootstrap_implementation_version,
        catalog_id=manifest.catalog_id,
    )


def assert_manifest_compatible(
    *,
    existing: VpiBootstrapManifest,
    expected: BootstrapCompatibilityIdentity,
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
    if existing.catalog_schema_version != expected.catalog_schema_version:
        mismatches.append("catalog_schema_version")
    if existing.search_index_schema_version != expected.search_index_schema_version:
        mismatches.append("search_index_schema_version")
    if existing.bootstrap_implementation_version != expected.bootstrap_implementation_version:
        mismatches.append("bootstrap_implementation_version")
    if existing.catalog_id != expected.catalog_id:
        mismatches.append("catalog_id")
    if mismatches:
        joined = ", ".join(mismatches)
        raise VpiBootstrapCompatibilityError(
            f"existing bootstrap manifest is incompatible ({joined}); "
            "explicit rebuild is required — destructive auto-rebuild is not performed"
        )

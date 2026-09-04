"""VPI semantic validation for installed data packages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from intergrax.proof_data.descriptor import ProofDataPackageDescriptor, load_proof_data_package_descriptor

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    EMBEDDING_CONFIGURATION_VERSION,
    load_vpi_embedding_configuration,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.search_representation import (
    SEARCH_REPRESENTATION_DERIVATION_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.data_package.errors import (
    VpiDataPackageCompatibilityError,
)
from platform_proofs.scenarios.verified_product_identification.data_package.identity import (
    CANONICAL_BUILDER_VERSION,
    CANONICAL_CATALOG_ID,
    CANONICAL_DATASET_CHECKSUM,
    CANONICAL_SELECTED_RECORD_COUNT,
    VPI_PACKAGE_ID,
    VPI_PACKAGE_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.data_package.paths import (
    assert_installed_data_present,
    resolve_installed_data_paths,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.compatibility import (
    EmbeddingArtifactCompatibilityIdentity,
    assert_manifest_compatible,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.model import (
    EMBEDDING_ARTIFACT_SCHEMA_VERSION,
    EmbeddingArtifactState,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.stores.parquet.manifest_io import (
    read_manifest_file,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.identity import (
    DatasetIdentity,
    resolve_dataset_identity,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.config import (
    DatasetVerificationMode,
)


@dataclass(frozen=True, slots=True)
class VpiDataPackageValidationReport:
    package_id: str
    package_version: str
    dataset_checksum: str
    dataset_record_count: int
    embedding_model: str
    embedding_dimension: int
    redistribution_status: str


def load_committed_descriptor(descriptor_path: Path) -> ProofDataPackageDescriptor:
    descriptor = load_proof_data_package_descriptor(descriptor_path)
    if descriptor.package_id != VPI_PACKAGE_ID:
        raise VpiDataPackageCompatibilityError(
            f"unexpected package_id: {descriptor.package_id}"
        )
    return descriptor


def validate_installed_vpi_data_package(
    install_root: Path,
    *,
    descriptor_path: Path | None = None,
) -> VpiDataPackageValidationReport:
    paths = resolve_installed_data_paths(install_root)
    assert_installed_data_present(paths)

    if descriptor_path is not None and descriptor_path.is_file():
        load_committed_descriptor(descriptor_path)

    dataset_identity = resolve_dataset_identity(
        dataset_path=paths.dataset_path,
        dataset_manifest_path=paths.dataset_manifest_path,
        verification_mode=DatasetVerificationMode.FAST,
    )
    _assert_dataset_manifest_semantics(paths.dataset_manifest_path)
    _assert_dataset_identity(dataset_identity)

    embedding_manifest = read_manifest_file(paths.embedding_manifest_path)
    if embedding_manifest.state is not EmbeddingArtifactState.READY:
        raise VpiDataPackageCompatibilityError(
            f"embedding artifact state must be READY, got {embedding_manifest.state.value}"
        )

    embedding_configuration = load_vpi_embedding_configuration()
    model = embedding_configuration.model
    if model is None:
        raise VpiDataPackageCompatibilityError("embedding model is required")

    expected = EmbeddingArtifactCompatibilityIdentity(
        dataset_checksum=dataset_identity.dataset_checksum,
        dataset_record_count=dataset_identity.dataset_record_count,
        search_representation_derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        embedding_configuration_version=EMBEDDING_CONFIGURATION_VERSION,
        embedding_provider=embedding_configuration.provider,
        embedding_model=model,
        embedding_dimension=embedding_configuration.expected_dimension,
        artifact_schema_version=EMBEDDING_ARTIFACT_SCHEMA_VERSION,
        catalog_id=CANONICAL_CATALOG_ID,
        source_revision=None,
    )
    assert_manifest_compatible(
        existing=embedding_manifest,
        expected=expected,
    )

    package_version = VPI_PACKAGE_VERSION
    redistribution_status = "REDISTRIBUTION_REVIEW_REQUIRED"
    if descriptor_path is not None and descriptor_path.is_file():
        descriptor = load_committed_descriptor(descriptor_path)
        package_version = descriptor.package_version
        redistribution_status = descriptor.redistribution_status.value

    return VpiDataPackageValidationReport(
        package_id=VPI_PACKAGE_ID,
        package_version=package_version,
        dataset_checksum=dataset_identity.dataset_checksum,
        dataset_record_count=dataset_identity.dataset_record_count,
        embedding_model=model,
        embedding_dimension=embedding_configuration.expected_dimension,
        redistribution_status=redistribution_status,
    )


def _assert_dataset_identity(dataset_identity: DatasetIdentity) -> None:
    if dataset_identity.dataset_checksum != CANONICAL_DATASET_CHECKSUM:
        raise VpiDataPackageCompatibilityError(
            "dataset checksum does not match canonical VPI selected corpus"
        )
    if dataset_identity.dataset_record_count != CANONICAL_SELECTED_RECORD_COUNT:
        raise VpiDataPackageCompatibilityError(
            "dataset record count does not match canonical VPI selected corpus"
        )


def _assert_dataset_manifest_semantics(manifest_path: Path) -> None:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VpiDataPackageCompatibilityError(
            f"failed to read dataset manifest: {manifest_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise VpiDataPackageCompatibilityError("dataset manifest must be a JSON object")

    builder_version = payload.get("builder_version")
    selected_count = payload.get("selected_record_count")
    output_sha256 = payload.get("output_sha256")

    if builder_version != CANONICAL_BUILDER_VERSION:
        raise VpiDataPackageCompatibilityError("dataset builder_version mismatch")
    if selected_count != CANONICAL_SELECTED_RECORD_COUNT:
        raise VpiDataPackageCompatibilityError("dataset selected_record_count mismatch")
    if output_sha256 != CANONICAL_DATASET_CHECKSUM:
        raise VpiDataPackageCompatibilityError("dataset output_sha256 mismatch")

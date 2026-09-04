"""Artifact input validation for storage bootstrap — no embedding provider dependency."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    EMBEDDING_CONFIGURATION_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.search_representation import (
    SEARCH_REPRESENTATION_DERIVATION_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.errors import (
    ArtifactCompatibilityError,
    ArtifactIntegrityError,
    VpiEmbeddingMaterializationError,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.compatibility import (
    EmbeddingArtifactCompatibilityIdentity,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.model import (
    EMBEDDING_ARTIFACT_SCHEMA_VERSION,
    EmbeddingArtifactManifest,
    EmbeddingArtifactState,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.config import (
    VpiBootstrapConfig,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.errors import (
    VpiBootstrapCompatibilityError,
    VpiBootstrapProviderError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationCheck,
    ValidationReport,
    ValidationStatus,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.identity import (
    DatasetIdentity,
)


def expected_artifact_identity(
    dataset_identity: DatasetIdentity,
    config: VpiBootstrapConfig,
) -> EmbeddingArtifactCompatibilityIdentity:
    embedding = config.embedding_configuration
    model = embedding.model
    if model is None:
        raise VpiBootstrapProviderError("embedding model is required for artifact identity")
    return EmbeddingArtifactCompatibilityIdentity(
        dataset_checksum=dataset_identity.dataset_checksum,
        dataset_record_count=dataset_identity.dataset_record_count,
        search_representation_derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        embedding_configuration_version=EMBEDDING_CONFIGURATION_VERSION,
        embedding_provider=embedding.provider,
        embedding_model=model,
        embedding_dimension=embedding.expected_dimension,
        artifact_schema_version=EMBEDDING_ARTIFACT_SCHEMA_VERSION,
        catalog_id=config.catalog_id,
        source_revision=config.source_revision,
    )


def assert_artifact_ready(manifest: EmbeddingArtifactManifest) -> None:
    if manifest.state is not EmbeddingArtifactState.READY:
        raise VpiBootstrapCompatibilityError(
            f"embedding artifact is not READY (state={manifest.state.value}); "
            "run materialize_embeddings before storage bootstrap"
        )


def assert_artifact_covers_target(
    manifest: EmbeddingArtifactManifest,
    requested_target_rows: int,
) -> None:
    available_rows = manifest.checkpoint_rows_materialized
    if requested_target_rows > available_rows:
        raise VpiBootstrapCompatibilityError(
            "requested storage bootstrap target "
            f"({requested_target_rows}) exceeds READY artifact coverage "
            f"({available_rows}); materialize additional rows or lower --max-records"
        )


def assert_dataset_covers_target(
    dataset_record_count: int,
    requested_target_rows: int,
) -> None:
    if requested_target_rows > dataset_record_count:
        raise VpiBootstrapCompatibilityError(
            "requested storage bootstrap target "
            f"({requested_target_rows}) exceeds dataset record count "
            f"({dataset_record_count})"
        )


def artifact_input_report_from_validation(
    identity_report: ValidationReport,
) -> ValidationReport:
    checks = tuple(
        ValidationCheck(
            name=f"embedding_artifact_{check.name}",
            status=check.status,
            detail=check.detail,
        )
        for check in identity_report.checks
    )
    return ValidationReport.from_checks(checks)


def translate_artifact_reader_error(exc: Exception) -> Exception:
    if isinstance(exc, ArtifactCompatibilityError):
        return VpiBootstrapCompatibilityError(str(exc))
    if isinstance(exc, ArtifactIntegrityError):
        return VpiBootstrapProviderError(str(exc))
    if isinstance(exc, VpiEmbeddingMaterializationError):
        return VpiBootstrapProviderError(str(exc))
    return VpiBootstrapProviderError(f"embedding artifact read failed: {exc}")


def ready_artifact_input_check(artifact_input_validation: ValidationReport) -> ValidationCheck:
    return ValidationCheck(
        name="embedding_artifact_ready",
        status=artifact_input_validation.status,
        detail=(
            artifact_input_validation.checks[0].detail
            if artifact_input_validation.checks
            else "artifact input validation"
        ),
    )

"""Operator preconditions resolved before provider writes begin."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    DataPackManifest,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.status import (
    DataPackStatus,
)
from platform_proofs.scenarios.verified_product_identification.qualification.integration.storage_environment import (
    storage_environment_gap,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.configuration import (
    PostgreSqlBootstrapConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.errors import (
    PostgreSqlBootstrapConfigurationError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.target_mapping import (
    resolve_physical_target,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.configuration import (
    QdrantBootstrapConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.errors import (
    QdrantBootstrapConfigurationError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.target_mapping import (
    resolve_physical_target as resolve_vector_physical_target,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.config import (
    DEFAULT_POSTGRESQL_SCHEMA,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.compatibility import (
    build_run_identity,
    validate_checkpoint_compatibility,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.filesystem_store import (
    FilesystemBootstrapCheckpointStore,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    BootstrapRequest,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.reader.errors import (
    DataPackReaderError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.reader.filesystem_reader import (
    FilesystemDataPackBootstrapReader,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.config import (
    OperatorRunMode,
    StorageLoadOperatorConfig,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.errors import (
    StorageLoadOperatorPreconditionError,
)


@dataclass(frozen=True, slots=True)
class ResolvedOperatorPreconditions:
    manifest: DataPackManifest
    bootstrap_request: BootstrapRequest


def _resolve_postgresql_schema_name() -> str:
    integration = PostgreSQLIntegrationConfig.from_env()
    if integration.tenant_schema:
        return integration.tenant_schema
    return DEFAULT_POSTGRESQL_SCHEMA


def _validate_provider_configuration(config: StorageLoadOperatorConfig) -> None:
    gap = storage_environment_gap()
    if gap is not None:
        raise StorageLoadOperatorPreconditionError(gap)
    schema_name = _resolve_postgresql_schema_name()
    try:
        pg_configuration = PostgreSqlBootstrapConfiguration.from_env(schema_name=schema_name)
        qdrant_configuration = QdrantBootstrapConfiguration.from_env(
            logical_collection_name=str(config.vector_target),
        )
        resolve_physical_target(config.relational_target, pg_configuration)
        resolve_vector_physical_target(config.vector_target, qdrant_configuration)
    except (
        ValueError,
        OSError,
        PostgreSqlBootstrapConfigurationError,
        QdrantBootstrapConfigurationError,
    ) as exc:
        raise StorageLoadOperatorPreconditionError(str(exc)) from exc


def _validate_artifact_root(artifact_root: Path) -> None:
    if not artifact_root.is_dir():
        raise StorageLoadOperatorPreconditionError(
            f"artifact root does not exist or is not a directory: {artifact_root}"
        )


def _validate_checkpoint_root(checkpoint_root: Path) -> None:
    try:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StorageLoadOperatorPreconditionError(
            f"checkpoint root is not usable: {checkpoint_root}"
        ) from exc


def _validate_manifest_expectations(
    manifest: DataPackManifest,
    config: StorageLoadOperatorConfig,
) -> None:
    if manifest.status is not DataPackStatus.READY:
        raise StorageLoadOperatorPreconditionError(
            f"data pack status must be READY (got {manifest.status.value})"
        )
    if manifest.record_count <= 0:
        raise StorageLoadOperatorPreconditionError("data pack record_count must be > 0")
    if not manifest.content_identity.strip():
        raise StorageLoadOperatorPreconditionError("data pack content identity is not readable")
    if (
        config.expected_record_count is not None
        and manifest.record_count != config.expected_record_count
    ):
        raise StorageLoadOperatorPreconditionError(
            "expected record count mismatch: "
            f"configured {config.expected_record_count}, manifest {manifest.record_count}"
        )
    if (
        config.expected_data_pack_content_identity is not None
        and manifest.content_identity != config.expected_data_pack_content_identity
    ):
        raise StorageLoadOperatorPreconditionError(
            "expected data pack content identity mismatch"
        )


def _validate_checkpoint_mode(
    *,
    config: StorageLoadOperatorConfig,
    manifest: DataPackManifest,
    bootstrap_request: BootstrapRequest,
) -> None:
    checkpoint_store = FilesystemBootstrapCheckpointStore(config.checkpoint_root)
    run_identity = build_run_identity(manifest=manifest, request=bootstrap_request)
    existing = checkpoint_store.load(run_identity)
    if config.run_mode is OperatorRunMode.FRESH and existing is not None:
        raise StorageLoadOperatorPreconditionError(
            "checkpoint already exists for FRESH run identity"
        )
    if config.run_mode is OperatorRunMode.RESUME:
        if existing is None:
            raise StorageLoadOperatorPreconditionError(
                "checkpoint not found for RESUME run identity"
            )
        compatibility = validate_checkpoint_compatibility(
            run_identity=run_identity,
            checkpoint=existing,
            manifest=manifest,
        )
        if not compatibility.is_compatible:
            raise StorageLoadOperatorPreconditionError(
                compatibility.reason or "checkpoint incompatible"
            )


def resolve_operator_preconditions(
    config: StorageLoadOperatorConfig,
    *,
    validate_providers: bool = True,
) -> ResolvedOperatorPreconditions:
    _validate_artifact_root(config.artifact_root)
    _validate_checkpoint_root(config.checkpoint_root)
    if validate_providers:
        _validate_provider_configuration(config)

    reader = FilesystemDataPackBootstrapReader(config.artifact_root)
    try:
        try:
            manifest = reader.read_manifest()
        except DataPackReaderError as exc:
            raise StorageLoadOperatorPreconditionError(str(exc)) from exc
        _validate_manifest_expectations(manifest, config)
        bootstrap_request = BootstrapRequest(
            artifact_root=config.artifact_root,
            relational_target=config.relational_target,
            vector_target=config.vector_target,
            batch_size=config.batch_size,
            resume_mode=config.resume_mode,
            verification_mode=config.verification_mode,
            plan_only=config.plan_only,
        )
        if not config.plan_only:
            _validate_checkpoint_mode(
                config=config,
                manifest=manifest,
                bootstrap_request=bootstrap_request,
            )
        return ResolvedOperatorPreconditions(
            manifest=manifest,
            bootstrap_request=bootstrap_request,
        )
    finally:
        reader.close()

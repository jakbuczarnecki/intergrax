"""Production composition root for VPI PostgreSQL + Qdrant storage bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.adapter import (
    PostgreSqlRelationalStorageAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.adapter import (
    QdrantVectorStorageAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.config import (
    DEFAULT_POSTGRESQL_SCHEMA,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.filesystem_store import (
    FilesystemBootstrapCheckpointStore,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.reader.filesystem_reader import (
    FilesystemDataPackBootstrapReader,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.service import (
    StorageBootstrapDependencies,
    StorageBootstrapService,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.config import (
    StorageLoadOperatorConfig,
)


def _resolve_postgresql_schema_name() -> str:
    integration = PostgreSQLIntegrationConfig.from_env()
    if integration.tenant_schema:
        return integration.tenant_schema
    return DEFAULT_POSTGRESQL_SCHEMA


@dataclass(slots=True)
class StorageLoadOperatorRuntime:
    service: StorageBootstrapService
    reader: FilesystemDataPackBootstrapReader
    relational: PostgreSqlRelationalStorageAdapter
    vector: QdrantVectorStorageAdapter

    def close(self) -> None:
        self.reader.close()
        try:
            self.vector.close()
        except AttributeError:
            return


def build_storage_load_operator_runtime(
    config: StorageLoadOperatorConfig,
) -> StorageLoadOperatorRuntime:
    schema_name = _resolve_postgresql_schema_name()
    reader = FilesystemDataPackBootstrapReader(config.artifact_root)
    relational = PostgreSqlRelationalStorageAdapter.from_env(schema_name=schema_name)
    vector = QdrantVectorStorageAdapter.from_env(
        logical_collection_name=str(config.vector_target),
    )
    checkpoint_store = FilesystemBootstrapCheckpointStore(config.checkpoint_root)
    service = StorageBootstrapService(
        dependencies=StorageBootstrapDependencies(
            reader=reader,
            relational=relational,
            vector=vector,
            checkpoint_store=checkpoint_store,
        )
    )
    return StorageLoadOperatorRuntime(
        service=service,
        reader=reader,
        relational=relational,
        vector=vector,
    )

"""Typed pgvector configuration for VPI vector storage bootstrap."""

from __future__ import annotations

import os
from dataclasses import dataclass

from intergrax.integrations._shared.p2.configs import SqlIntegrationConfig
from intergrax.integrations.providers.relational_store.postgresql.config import (
    validate_schema_identifier,
)

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.configuration import (
    validate_table_identifier,
)

CANONICAL_EMBEDDING_PROVIDER = "hf"
CANONICAL_EMBEDDING_MODEL = "BAAI/bge-m3"
CANONICAL_EMBEDDING_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
CANONICAL_EMBEDDING_DIMENSION = 1024
CANONICAL_DISTANCE_METRIC = "cosine"

DEFAULT_LOGICAL_TABLE_NAME = "vpi_data_pack_vector_embedding"
DEFAULT_LOGICAL_TARGET_NAME = "vpi-product-embeddings"
DEFAULT_APPLICATION_NAME = "vpi-pgvector-bootstrap"
DEFAULT_INSERT_BATCH_SIZE = 64

VECTOR_TRANSPORT_FLOAT32_TOLERANCE = 0.0


@dataclass(frozen=True, slots=True)
class ExpectedVectorIdentity:
    provider: str
    model: str
    revision: str
    dimension: int

    @classmethod
    def canonical_vpi(cls) -> ExpectedVectorIdentity:
        return cls(
            provider=CANONICAL_EMBEDDING_PROVIDER,
            model=CANONICAL_EMBEDDING_MODEL,
            revision=CANONICAL_EMBEDDING_REVISION,
            dimension=CANONICAL_EMBEDDING_DIMENSION,
        )


def _resolve_dimension(raw_dimension: str | int | None) -> int:
    if raw_dimension is None:
        raw_dimension = os.environ.get("INTERGRAX_PGVECTOR_DIMENSION", "").strip()
    if raw_dimension == "":
        return CANONICAL_EMBEDDING_DIMENSION
    try:
        dimension = int(raw_dimension)
    except (TypeError, ValueError) as exc:
        raise ValueError("INTERGRAX_PGVECTOR_DIMENSION must be a positive integer") from exc
    if dimension <= 0:
        raise ValueError("INTERGRAX_PGVECTOR_DIMENSION must be > 0")
    return dimension


@dataclass(frozen=True, slots=True)
class PgVectorBootstrapConfiguration:
    sql_integration: SqlIntegrationConfig
    schema_name: str
    table_name: str
    expected_vector_identity: ExpectedVectorIdentity
    insert_batch_size: int = DEFAULT_INSERT_BATCH_SIZE
    vector_transport_tolerance: float = VECTOR_TRANSPORT_FLOAT32_TOLERANCE
    allow_create_extension: bool = True
    application_name: str = DEFAULT_APPLICATION_NAME

    @classmethod
    def from_env(
        cls,
        *,
        schema_name: str,
        table_name: str = DEFAULT_LOGICAL_TABLE_NAME,
        insert_batch_size: int = DEFAULT_INSERT_BATCH_SIZE,
        expected_vector_identity: ExpectedVectorIdentity | None = None,
        allow_create_extension: bool = True,
        application_name: str = DEFAULT_APPLICATION_NAME,
        dimension: int | None = None,
    ) -> PgVectorBootstrapConfiguration:
        if not application_name.strip():
            raise ValueError("application_name must be non-empty")
        if insert_batch_size <= 0:
            raise ValueError("insert_batch_size must be > 0")
        resolved_dimension = _resolve_dimension(dimension)
        identity = expected_vector_identity or ExpectedVectorIdentity.canonical_vpi()
        if identity.dimension != resolved_dimension:
            raise ValueError(
                "configured pgvector dimension does not match expected vector identity"
            )
        sql_integration = SqlIntegrationConfig.from_env(
            "INTERGRAX_PGVECTOR",
            tenant_schema=schema_name,
        )
        return cls(
            sql_integration=sql_integration,
            schema_name=validate_schema_identifier(schema_name),
            table_name=validate_table_identifier(table_name),
            expected_vector_identity=identity,
            insert_batch_size=insert_batch_size,
            allow_create_extension=allow_create_extension,
            application_name=application_name,
        )

    def __repr__(self) -> str:
        return (
            "PgVectorBootstrapConfiguration("
            f"schema_name={self.schema_name!r}, "
            f"table_name={self.table_name!r}, "
            f"insert_batch_size={self.insert_batch_size}, "
            f"expected_dimension={self.expected_vector_identity.dimension})"
        )

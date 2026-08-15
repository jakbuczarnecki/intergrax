# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Low-level pgvector openers — internal to the pgvector integration package."""

from __future__ import annotations

import os
from typing import Callable, Optional

from intergrax.integrations._shared.p3.configs import VectorIntegrationConfig
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations._shared.p2.configs import SqlIntegrationConfig
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.vector_store.pgvector.integration import PgvectorVectorStoreIntegration


def _open_rag_store(
    config: VectorIntegrationConfig,
    *,
    config_overrides: dict[str, object],
    store_factory: Optional[Callable[[], VectorStore]] = None,
) -> VectorStore:
    if store_factory is not None:
        return store_factory()
    from intergrax.integrations.providers.vector_store.pgvector.rag_store import PgVectorRagStore

    sql_config = SqlIntegrationConfig.from_env("INTERGRAX_PGVECTOR", **config_overrides)
    dsn = sql_config.connection_dsn()
    raw_dimension = config_overrides.get(
        "dimension",
        os.environ.get("INTERGRAX_PGVECTOR_DIMENSION", ""),
    )
    try:
        dimension = int(raw_dimension)
    except (TypeError, ValueError) as exc:
        raise IntegrationConfigurationError(
            "pgvector requires INTERGRAX_PGVECTOR_DIMENSION or dimension override"
        ) from exc
    return PgVectorRagStore(
        tenant_id=config.tenant_id,
        dsn=dsn,
        dimension=dimension,
    )


def open_pgvector_vector_store(
    config: VectorIntegrationConfig,
    *,
    config_overrides: dict[str, object] | None = None,
    implementation: Optional[VectorStore] = None,
    store: Optional[VectorStore] = None,
    store_factory: Optional[Callable[[], VectorStore]] = None,
) -> VectorStore:
    if implementation is not None:
        return implementation
    overrides = config_overrides or {}
    inner = (
        store
        if store is not None
        else _open_rag_store(config, config_overrides=overrides, store_factory=store_factory)
    )
    return PgvectorVectorStoreIntegration.from_store(config, inner)

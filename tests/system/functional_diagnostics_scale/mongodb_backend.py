# © Artur Czarnecki. All rights reserved.

"""MongoDB scale backend probe for DIAG-FUNCTIONAL-SCALE-S1."""

from __future__ import annotations

import os

from intergrax.integrations._shared.conformance import assert_conditional_document_store
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from intergrax.integrations.providers.document_store.mongodb.bundle import (
    create_mongodb_document_store,
)
from intergrax.integrations.providers.document_store.mongodb.config import (
    ENV_MONGODB_URI,
    MongoDBIntegrationConfig,
)
from intergrax.integrations.providers.document_store.mongodb.opens import (
    DOCUMENT_KEY_INDEX_KEYS,
    DOCUMENT_KEY_INDEX_NAME,
    _import_pymongo,
)
from tests.system.functional_diagnostics_scale.backend import (
    BackendIndexObservation,
    BackendQueryEfficiencyObservation,
    BackendResourceObservation,
    ScaleBackendIdentity,
)

_DEFAULT_URI = "mongodb://localhost:27017"
_DEFAULT_DATABASE = "intergrax_diag_scale_s1"
_DOCUMENT_PARTITION_PREFIX = "intergrax.functional_evidence.v1"
_EXEC_ROW_PREFIX = "exec:"


class MongoFunctionalDiagnosticsScaleProbe:
    """Production MongoDB DocumentStore provider probe for scale qualification."""

    def __init__(self, *, collection_name: str) -> None:
        self._collection_name = collection_name
        self._config = MongoDBIntegrationConfig.from_env(
            database=_DEFAULT_DATABASE,
            collection_name=collection_name,
        )
        self._pymongo_client: object | None = None
        self._pymongo_collection: object | None = None

    @property
    def provider_id(self) -> str:
        return "mongodb"

    def prepare(self) -> None:
        uri = resolve_mongodb_uri()
        MongoClient = _import_pymongo()
        database_name, collection_name = self._config.qualified_collection()
        self._pymongo_client = MongoClient(uri)
        self._pymongo_collection = self._pymongo_client[database_name][collection_name]

    def build_document_store(self) -> ConditionalDocumentStore:
        uri = resolve_mongodb_uri()
        if not uri:
            raise IntegrationConfigurationError(
                "INTERGRAX_MONGODB_URI is required for S1 scale qualification",
            )
        store = create_mongodb_document_store(
            uri=uri,
            database=self._config.database,
            collection_name=self._config.collection_name,
        )
        return assert_conditional_document_store(store)

    def backend_identity(self) -> ScaleBackendIdentity:
        database, collection = self._config.qualified_collection()
        return ScaleBackendIdentity(
            provider_id=self.provider_id,
            document_store_type="_MongoDBDocumentStore",
            database_name=database,
            collection_name=collection,
        )

    def collect_backend_metrics(self) -> BackendResourceObservation:
        collection = self._pymongo_collection
        if collection is None:
            return BackendResourceObservation(
                document_count=None,
                storage_size_bytes=None,
                indexes=(),
            )
        database = collection.database
        stats = database.command("collStats", collection.name)
        indexes: list[BackendIndexObservation] = []
        for item in collection.index_information().values():
            key_spec = item.get("key")
            if not isinstance(key_spec, list):
                continue
            keys = tuple((str(key), int(direction)) for key, direction in key_spec)
            indexes.append(
                BackendIndexObservation(
                    index_name=str(item.get("name", "")),
                    keys=keys,
                    unique=bool(item.get("unique", False)),
                ),
            )
        return BackendResourceObservation(
            document_count=int(stats.get("count", 0)),
            storage_size_bytes=int(stats.get("storageSize", 0)),
            indexes=tuple(indexes),
        )

    def observe_execution_query_efficiency(
        self,
        *,
        tenant_id: str,
        task_id: str,
        run_id: str,
    ) -> BackendQueryEfficiencyObservation | None:
        collection = self._pymongo_collection
        if collection is None:
            return None
        partition_key = f"{_DOCUMENT_PARTITION_PREFIX}:{tenant_id}"
        row_key_prefix = f"{_EXEC_ROW_PREFIX}{task_id}:{run_id}:"
        query_filter = {
            "partition_key": partition_key,
            "row_key": {"$regex": f"^{row_key_prefix}"},
        }
        explain = collection.find(query_filter).sort("row_key", 1).limit(1).explain()
        execution_stats = explain.get("executionStats")
        if not isinstance(execution_stats, dict):
            return None
        return BackendQueryEfficiencyObservation(
            partition_key=partition_key,
            row_key_prefix=row_key_prefix,
            documents_examined=int(execution_stats.get("totalDocsExamined", 0)),
            keys_examined=int(execution_stats.get("totalKeysExamined", 0)),
            n_returned=int(execution_stats.get("nReturned", 0)),
        )

    def cleanup(self) -> None:
        if self._pymongo_collection is not None:
            self._pymongo_collection.drop()
        client = self._pymongo_client
        if client is not None:
            client.close()

    def close_document_store(self, store: ConditionalDocumentStore) -> None:
        store.close()

    def production_index_observations(self) -> tuple[BackendIndexObservation, ...]:
        return (
            BackendIndexObservation(
                index_name=DOCUMENT_KEY_INDEX_NAME,
                keys=DOCUMENT_KEY_INDEX_KEYS,
                unique=True,
            ),
        )


def resolve_mongodb_uri() -> str:
    return os.environ.get(ENV_MONGODB_URI, _DEFAULT_URI).strip() or _DEFAULT_URI


def mongodb_available() -> bool:
    return bool(resolve_mongodb_uri().strip())


__all__ = [
    "MongoFunctionalDiagnosticsScaleProbe",
    "mongodb_available",
    "resolve_mongodb_uri",
]

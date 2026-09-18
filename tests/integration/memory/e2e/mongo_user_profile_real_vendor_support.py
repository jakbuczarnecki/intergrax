# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-5C — reusable real MongoDB UserProfile qualification harness."""

from __future__ import annotations

import os
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from intergrax.applications._shared.memory_wiring import MemoryPlatformWiring, resolve_memory_platform_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile, MemoryProfile
from intergrax.integrations.core.binding import IntegrationBinding
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.integrations.providers.document_store.mongodb.adapter import _MongoDBDocumentStore
from intergrax.integrations.providers.document_store.mongodb.bundle import create_mongodb_document_store
from intergrax.integrations.providers.document_store.mongodb.manifest import MANIFEST
from intergrax.integrations.providers.document_store.mongodb.opens import (
    DOCUMENT_KEY_INDEX_NAME,
)
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.memory.stores.document_store_user_profile_store import DocumentStoreUserProfileStore
from intergrax.memory.user_profile_store import UserProfileStore

_COLLECTION_PREFIX = "mem_audit_5c_"
_DEFAULT_DATABASE = "intergrax_mem_audit_5c"

CreateUserProfileStore = Callable[[], UserProfileStore]
DisposeUserProfileStore = Callable[[UserProfileStore], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class MongoUserProfileQualificationEnv:
    uri: str
    database: str
    collection: str
    qualification_run_id: str


def unique_qualification_run_id(prefix: str = "mem-5c") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


def apply_mongo_env(monkeypatch: Any, env: MongoUserProfileQualificationEnv) -> None:
    monkeypatch.setenv("INTERGRAX_MONGODB_URI", env.uri)
    monkeypatch.setenv("INTERGRAX_MONGODB_DATABASE", env.database)
    monkeypatch.setenv("INTERGRAX_MONGODB_COLLECTION", env.collection)


def build_qualification_env(
    *,
    uri: str,
    qualification_run_id: str,
    database: str = _DEFAULT_DATABASE,
) -> MongoUserProfileQualificationEnv:
    return MongoUserProfileQualificationEnv(
        uri=uri,
        database=database,
        collection=f"{_COLLECTION_PREFIX}{qualification_run_id.replace('-', '_')}",
        qualification_run_id=qualification_run_id,
    )


def open_mongo_document_store(env: MongoUserProfileQualificationEnv) -> _MongoDBDocumentStore:
    store = create_mongodb_document_store(
        uri=env.uri,
        database=env.database,
        collection_name=env.collection,
    )
    assert isinstance(store, _MongoDBDocumentStore)
    return store


def mongo_topology_metadata(document_store: _MongoDBDocumentStore) -> dict[str, str | None]:
    client = document_store.mongo_client._client
    driver_version: str | None = None
    server_version: str | None = None
    if client is not None:
        try:
            import pymongo

            driver_version = pymongo.version
            server_version = str(client.server_info().get("version"))
        except Exception:
            pass
    return {
        "driver_version": driver_version,
        "server_version": server_version,
        "database": document_store.mongo_client.config.database,
        "collection": document_store.mongo_client.config.collection_name,
        "backend_provider_id": "mongodb",
    }


def list_document_key_indexes(document_store: _MongoDBDocumentStore) -> list[dict[str, Any]]:
    collection = document_store.mongo_client._collection
    indexes: list[dict[str, Any]] = []
    for spec in collection.list_indexes():
        indexes.append(dict(spec))
    return indexes


def drop_qualification_collection(env: MongoUserProfileQualificationEnv) -> None:
    store = open_mongo_document_store(env)
    try:
        store.mongo_client._collection.drop()
    finally:
        store.close()


def mongo_user_profile_store_factory(
    env: MongoUserProfileQualificationEnv,
) -> tuple[CreateUserProfileStore, DisposeUserProfileStore]:
    stores: list[_MongoDBDocumentStore] = []

    def _create() -> UserProfileStore:
        document_store = open_mongo_document_store(env)
        stores.append(document_store)
        return DocumentStoreUserProfileStore(document_store)

    async def _dispose(store: UserProfileStore) -> None:
        del store
        while stores:
            document_store = stores.pop()
            document_store.close()

    return _create, _dispose


def persistent_memory_profile() -> MemoryProfile:
    return MemoryProfile(
        enable_user_memory=True,
        enable_long_term_memory=True,
        enable_entity_graph_memory=True,
    )


def product_mongo_environment(
    *,
    profile_id: str,
    env: MongoUserProfileQualificationEnv,
) -> ApplicationEnvironmentProfile:
    application = ApplicationEnvironmentProfile.product_defaults(profile_id=profile_id)
    application.memory_profile = persistent_memory_profile()
    application.integration_profile = IntegrationProfile(
        document_store=IntegrationBinding.from_manifest(MANIFEST),
    )
    application.integration_profile.options = {
        **(application.integration_profile.options or {}),
        "mongodb": {
            "uri": env.uri,
            "database": env.database,
            "collection_name": env.collection,
        },
    }
    return application


def resolve_product_mongo_wiring(
    application: ApplicationEnvironmentProfile,
    *,
    qualification_evidence_registry: object | None = None,
    durability_evidence_registry: object | None = None,
) -> MemoryPlatformWiring:
    return resolve_memory_platform_wiring(
        application,
        qualification_evidence_registry=qualification_evidence_registry,
        durability_evidence_registry=durability_evidence_registry,
    )


def close_mongo_wiring(wiring: MemoryPlatformWiring) -> None:
    if wiring.mongodb_bundle is not None:
        store = wiring.mongodb_bundle.document_store.as_document_store()
        if isinstance(store, _MongoDBDocumentStore):
            store.close()


def assert_unique_document_key_index(document_store: _MongoDBDocumentStore) -> None:
    indexes = list_document_key_indexes(document_store)
    names = {index.get("name") for index in indexes}
    assert DOCUMENT_KEY_INDEX_NAME in names


def count_profile_documents(document_store: _MongoDBDocumentStore, *, tenant_id: str, user_id: str) -> int:
    collection = document_store.mongo_client._collection
    return collection.count_documents(
        {"partition_key": tenant_id, "row_key": user_id},
    )

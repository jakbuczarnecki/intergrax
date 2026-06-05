# © Artur Czarnecki. All rights reserved.

"""MEM-PERS.2: DocumentStore-backed user profile persistence."""

from __future__ import annotations

import pytest

from intergrax.integrations.contracts.document_store import DocumentRecord, DocumentQueryResult
from intergrax.memory.stores.document_store_user_profile_store import DocumentStoreUserProfileStore
from intergrax.memory.user_profile_memory import UserIdentity, UserPreferences, UserProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _InMemoryDocumentStore:
    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], DocumentRecord] = {}

    def get(self, partition_key: str, row_key: str) -> DocumentRecord | None:
        return self._rows.get((partition_key, row_key))

    def put(self, document: DocumentRecord) -> None:
        self._rows[(document.partition_key, document.row_key)] = document

    def delete(self, partition_key: str, row_key: str) -> None:
        self._rows.pop((partition_key, row_key), None)

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: str | None = None,
    ) -> DocumentQueryResult:
        rows = [
            record
            for (pk, rk), record in self._rows.items()
            if pk == partition_key and (row_key_prefix is None or rk.startswith(row_key_prefix))
        ]
        return DocumentQueryResult(documents=rows[:limit], total=len(rows))

    def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_document_store_user_profile_roundtrip() -> None:
    backend = _InMemoryDocumentStore()
    store = DocumentStoreUserProfileStore(backend)

    loaded = await store.get_profile(tenant_id="t1", user_id="u1")
    assert loaded.identity.user_id == "u1"

    profile = UserProfile(
        identity=UserIdentity(user_id="u1", display_name="Ada"),
        preferences=UserPreferences(preferred_language="en"),
    )
    await store.save_profile(tenant_id="t1", profile=profile)

    reloaded = await store.get_profile(tenant_id="t1", user_id="u1")
    assert reloaded.identity.display_name == "Ada"
    assert reloaded.preferences.preferred_language == "en"


@pytest.mark.asyncio
async def test_document_store_user_profile_delete_is_idempotent() -> None:
    backend = _InMemoryDocumentStore()
    store = DocumentStoreUserProfileStore(backend)
    profile = UserProfile(
        identity=UserIdentity(user_id="u2"),
        preferences=UserPreferences(),
    )
    await store.save_profile(tenant_id="t1", profile=profile)
    await store.delete_profile(tenant_id="t1", user_id="u2")
    await store.delete_profile(tenant_id="t1", user_id="u2")
    loaded = await store.get_profile(tenant_id="t1", user_id="u2")
    assert loaded.identity.user_id == "u2"

# © Artur Czarnecki. All rights reserved.

"""DocumentStore-backed user profile persistence (Phase MEM-PERS.2)."""

from __future__ import annotations

from intergrax.integrations.contracts.document_store import DocumentRecord, DocumentStore
from intergrax.memory.user_profile_memory import UserIdentity, UserPreferences, UserProfile
from intergrax.memory.user_profile_serialization import user_profile_from_json, user_profile_to_json
from intergrax.memory.user_profile_store import UserProfileStore
from intergrax.utils.time_provider import SystemTimeProvider


class DocumentStoreUserProfileStore(UserProfileStore):
    """
    Persist ``UserProfile`` aggregates via a Tier-0 ``DocumentStore``.

    Partition key = tenant_id; row key = user_id; payload field ``profile_json``.
    """

    _PROFILE_FIELD = "profile_json"

    def __init__(self, document_store: DocumentStore) -> None:
        self._document_store = document_store

    async def get_profile(
        self,
        *,
        tenant_id: str,
        user_id: str,
    ) -> UserProfile:
        record = self._document_store.get(tenant_id, user_id)
        if record is None:
            identity = UserIdentity(user_id=user_id)
            preferences = UserPreferences()
            return UserProfile(identity=identity, preferences=preferences)
        raw = record.data.get(self._PROFILE_FIELD)
        if not isinstance(raw, str) or not raw.strip():
            identity = UserIdentity(user_id=user_id)
            preferences = UserPreferences()
            return UserProfile(identity=identity, preferences=preferences)
        return user_profile_from_json(raw)

    async def save_profile(
        self,
        *,
        tenant_id: str,
        profile: UserProfile,
    ) -> None:
        user_id = profile.identity.user_id
        self._document_store.put(
            DocumentRecord(
                partition_key=tenant_id,
                row_key=user_id,
                data={
                    self._PROFILE_FIELD: user_profile_to_json(profile),
                    "updated_at_utc": SystemTimeProvider.utc_now().isoformat(),
                },
            )
        )

    async def delete_profile(
        self,
        *,
        tenant_id: str,
        user_id: str,
    ) -> None:
        self._document_store.delete(tenant_id, user_id)

    def close(self) -> None:
        self._document_store.close()

# © Artur Czarnecki. All rights reserved.

"""Platform-resolved memory provider identity (MEM-FINAL-AUDIT-5A-R2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.memory.contracts.provider_qualification import MemoryProviderCapabilityKind

__all__ = [
    "MemoryProviderIdentity",
    "MemoryProviderIdentitySource",
    "builtin_user_profile_store_identity",
    "builtin_session_turn_index_store_identity",
    "plugin_user_profile_store_identity",
    "plugin_session_turn_index_store_identity",
    "BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID",
    "BUILTIN_IN_MEMORY_USER_PROFILE_ID",
    "BUILTIN_SQLITE_USER_PROFILE_ID",
    "BUILTIN_VECTOR_SESSION_TURN_INDEX_ID",
]


class MemoryProviderIdentitySource(StrEnum):
    BUILT_IN = "built_in"
    PLUGIN = "plugin"
    DIRECT_INJECTION = "direct_injection"


BUILTIN_SQLITE_USER_PROFILE_ID = "sqlite.user_profile"
BUILTIN_IN_MEMORY_USER_PROFILE_ID = "reference.in_memory.user_profile"
BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID = "document_store.user_profile"
BUILTIN_VECTOR_SESSION_TURN_INDEX_ID = "vector.session_turn_index"


@dataclass(frozen=True, slots=True)
class MemoryProviderIdentity:
    """Immutable platform-owned provider identity for qualification evidence lookup."""

    provider_id: str
    capability: MemoryProviderCapabilityKind
    source: MemoryProviderIdentitySource
    provider_version: str | None = None
    backing_provider_id: str | None = None
    backing_provider_version: str | None = None


def memory_provider_backing_identity_mismatch(
    identity: MemoryProviderIdentity,
    *,
    backing_provider_id: str | None,
    backing_provider_version: str | None = None,
) -> bool:
    """True when authoritative backing fields differ (exact ``None`` semantics)."""
    if identity.backing_provider_id != backing_provider_id:
        return True
    if identity.backing_provider_version is None and backing_provider_version is None:
        return False
    if identity.backing_provider_version is None or backing_provider_version is None:
        return True
    return identity.backing_provider_version != backing_provider_version


def builtin_user_profile_store_identity(
    provider_id: str,
    *,
    provider_version: str | None = None,
    backing_provider_id: str | None = None,
    backing_provider_version: str | None = None,
) -> MemoryProviderIdentity:
    return MemoryProviderIdentity(
        provider_id=provider_id,
        capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
        source=MemoryProviderIdentitySource.BUILT_IN,
        provider_version=provider_version,
        backing_provider_id=backing_provider_id,
        backing_provider_version=backing_provider_version,
    )


def builtin_session_turn_index_store_identity(
    provider_id: str,
    *,
    provider_version: str | None = None,
    backing_provider_id: str | None = None,
    backing_provider_version: str | None = None,
) -> MemoryProviderIdentity:
    return MemoryProviderIdentity(
        provider_id=provider_id,
        capability=MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,
        source=MemoryProviderIdentitySource.BUILT_IN,
        provider_version=provider_version,
        backing_provider_id=backing_provider_id,
        backing_provider_version=backing_provider_version,
    )


def plugin_user_profile_store_identity(
    plugin_id: str,
    *,
    provider_version: str | None = None,
) -> MemoryProviderIdentity:
    return MemoryProviderIdentity(
        provider_id=plugin_id,
        capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
        source=MemoryProviderIdentitySource.PLUGIN,
        provider_version=provider_version,
    )


def plugin_session_turn_index_store_identity(
    plugin_id: str,
    *,
    provider_version: str | None = None,
    backing_provider_id: str | None = None,
    backing_provider_version: str | None = None,
) -> MemoryProviderIdentity:
    return MemoryProviderIdentity(
        provider_id=plugin_id,
        capability=MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,
        source=MemoryProviderIdentitySource.PLUGIN,
        provider_version=provider_version,
        backing_provider_id=backing_provider_id,
        backing_provider_version=backing_provider_version,
    )

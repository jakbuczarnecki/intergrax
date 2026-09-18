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
    "plugin_user_profile_store_identity",
    "BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID",
    "BUILTIN_IN_MEMORY_USER_PROFILE_ID",
    "BUILTIN_SQLITE_USER_PROFILE_ID",
]


class MemoryProviderIdentitySource(StrEnum):
    BUILT_IN = "built_in"
    PLUGIN = "plugin"
    DIRECT_INJECTION = "direct_injection"


BUILTIN_SQLITE_USER_PROFILE_ID = "sqlite.user_profile"
BUILTIN_IN_MEMORY_USER_PROFILE_ID = "reference.in_memory.user_profile"
BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID = "document_store.user_profile"


@dataclass(frozen=True, slots=True)
class MemoryProviderIdentity:
    """Immutable platform-owned provider identity for qualification evidence lookup."""

    provider_id: str
    capability: MemoryProviderCapabilityKind
    source: MemoryProviderIdentitySource
    provider_version: str | None = None


def builtin_user_profile_store_identity(
    provider_id: str,
    *,
    provider_version: str | None = None,
) -> MemoryProviderIdentity:
    return MemoryProviderIdentity(
        provider_id=provider_id,
        capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
        source=MemoryProviderIdentitySource.BUILT_IN,
        provider_version=provider_version,
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

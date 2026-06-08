# © Artur Czarnecki. All rights reserved.

"""Tier-3 memory platform wiring (Phase MEM-1.3, MEM-2.2, MEM-PERS.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.integrations.providers.document_store.mongodb.bundle import (
    MongoDBIntegrationBundle,
    create_mongodb_integration,
)
from intergrax.integrations.providers.relational_store.sqlite.bundle import (
    SQLiteIntegrationBundle,
    create_sqlite_integration,
)
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.memory.stores.document_store_user_profile_store import DocumentStoreUserProfileStore
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.user_profile_store import UserProfileStore
from intergrax.runtime.nexus.session.document_store_session_storage import DocumentStoreSessionStorage
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.nexus.session.session_storage import SessionStorage
from intergrax.runtime.organization.organization_profile_manager import OrganizationProfileManager
from intergrax.runtime.organization.organization_profile_store import OrganizationProfileStore


@dataclass(frozen=True)
class MemoryPlatformWiring:
    """Resolved session + profile stores for a Tier-3 host."""

    session_storage: SessionStorage
    user_profile_store: UserProfileStore
    organization_profile_store: OrganizationProfileStore | None
    sqlite_bundle: SQLiteIntegrationBundle | None = None
    mongodb_bundle: MongoDBIntegrationBundle | None = None


def _sqlite_enabled(profile: IntegrationProfile) -> bool:
    binding = profile.relational_store
    if binding is None:
        return False
    return binding.resolved_slug() == "sqlite"


def _mongodb_enabled(profile: IntegrationProfile) -> bool:
    binding = profile.document_store
    if binding is None:
        return False
    return binding.resolved_slug() == "mongodb"


def _sqlite_integration_overrides(profile: IntegrationProfile) -> dict[str, object]:
    options = profile.options or {}
    raw = options.get("sqlite")
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def _mongodb_integration_overrides(profile: IntegrationProfile) -> dict[str, object]:
    options = profile.options or {}
    raw = options.get("mongodb")
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def resolve_memory_platform_wiring(
    env: ApplicationEnvironmentProfile,
    *,
    integration_profile: IntegrationProfile | None = None,
) -> MemoryPlatformWiring:
    """
    Resolve durable memory backends from the integration profile.

    Priority:
    1. SQLite relational_store — session + user LTM + org profile (lab default).
    2. MongoDB document_store — user LTM artifacts when SQLite is not enabled (MEM-PERS.2).
    3. In-memory fallbacks for session and user LTM.
    """
    profile = integration_profile or env.integration_profile
    if _sqlite_enabled(profile):
        bundle = create_sqlite_integration(**_sqlite_integration_overrides(profile))
        return MemoryPlatformWiring(
            session_storage=bundle.session_storage,
            user_profile_store=bundle.user_profile_store,
            organization_profile_store=bundle.organization_profile_store,
            sqlite_bundle=bundle,
            mongodb_bundle=None,
        )

    if _mongodb_enabled(profile):
        mongo_bundle = create_mongodb_integration(**_mongodb_integration_overrides(profile))
        document_store: DocumentStore = mongo_bundle.document_store
        return MemoryPlatformWiring(
            session_storage=DocumentStoreSessionStorage(document_store),
            user_profile_store=DocumentStoreUserProfileStore(document_store),
            organization_profile_store=None,
            sqlite_bundle=None,
            mongodb_bundle=mongo_bundle,
        )

    return MemoryPlatformWiring(
        session_storage=InMemorySessionStorage(),
        user_profile_store=InMemoryUserProfileStore(),
        organization_profile_store=None,
        sqlite_bundle=None,
        mongodb_bundle=None,
    )


def build_session_manager_from_environment(
    env: ApplicationEnvironmentProfile,
    *,
    integration_profile: IntegrationProfile | None = None,
    memory_wiring: MemoryPlatformWiring | None = None,
) -> SessionManager:
    """Construct ``SessionManager`` with profile managers driven by ``MemoryProfile``."""
    wiring = memory_wiring or resolve_memory_platform_wiring(
        env,
        integration_profile=integration_profile,
    )
    memory_profile = env.memory_profile

    user_manager: Optional[UserProfileManager] = None
    if memory_profile.enable_user_memory or memory_profile.enable_long_term_memory:
        user_manager = UserProfileManager(wiring.user_profile_store)

    org_manager: Optional[OrganizationProfileManager] = None
    if memory_profile.enable_org_memory and wiring.organization_profile_store is not None:
        org_manager = OrganizationProfileManager(wiring.organization_profile_store)

    return SessionManager(
        wiring.session_storage,
        user_profile_manager=user_manager,
        organization_profile_manager=org_manager,
    )

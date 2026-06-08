# © Artur Czarnecki. All rights reserved.

"""MEM-1.3 / MEM-2.2: Tier-3 memory platform wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.memory_wiring import (
    build_session_manager_from_environment,
    resolve_memory_platform_wiring,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    MemoryProfile,
)
from intergrax.integrations.providers.relational_store.sqlite.bundle import create_sqlite_integration
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.stores.sqlite_user_profile_store import SQLiteUserProfileStore
from intergrax.runtime.nexus.session.document_store_session_storage import DocumentStoreSessionStorage
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.sqlite_session_storage import SQLiteSessionStorage
from intergrax.runtime.organization.stores.sqlite_organization_profile_store import (
    SQLiteOrganizationProfileStore,
)
from tests.unit.integrations.providers.document_store.test_mongodb import _collection_factory

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_resolve_memory_platform_wiring_uses_sqlite_when_relational_store_sqlite(
    tmp_path: Path,
) -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="mem.wiring.sqlite")
    env.integration_profile = IntegrationProfile.lab_harness_preset()
    env.integration_profile.options = {
        **(env.integration_profile.options or {}),
        "sqlite": {"data_dir": str(tmp_path)},
    }

    wiring = resolve_memory_platform_wiring(env)

    assert wiring.sqlite_bundle is not None
    assert isinstance(wiring.session_storage, SQLiteSessionStorage)
    assert isinstance(wiring.user_profile_store, SQLiteUserProfileStore)
    assert isinstance(wiring.organization_profile_store, SQLiteOrganizationProfileStore)


def test_resolve_memory_platform_wiring_falls_back_to_in_memory_without_sqlite() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.wiring.inmem")
    env.integration_profile = IntegrationProfile()

    wiring = resolve_memory_platform_wiring(env)

    assert wiring.sqlite_bundle is None
    assert wiring.mongodb_bundle is None
    assert isinstance(wiring.session_storage, InMemorySessionStorage)
    assert isinstance(wiring.user_profile_store, InMemoryUserProfileStore)
    assert wiring.organization_profile_store is None


def test_resolve_memory_platform_wiring_uses_mongodb_document_store_for_user_ltm() -> None:
    from intergrax.integrations.core.binding import IntegrationBinding
    from intergrax.integrations.providers.document_store.mongodb.manifest import MANIFEST
    from intergrax.memory.stores.document_store_user_profile_store import DocumentStoreUserProfileStore

    factory, _ = _collection_factory()
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.wiring.mongo")
    env.integration_profile = IntegrationProfile(
        document_store=IntegrationBinding.from_manifest(MANIFEST),
    )
    env.integration_profile.options = {
        **(env.integration_profile.options or {}),
        "mongodb": {
            "uri": "mongodb://localhost:27017",
            "database": "intergrax_test",
            "collection_name": "user_profiles",
            "collection_factory": factory,
        },
    }

    wiring = resolve_memory_platform_wiring(env)

    assert wiring.sqlite_bundle is None
    assert wiring.mongodb_bundle is not None
    assert isinstance(wiring.session_storage, DocumentStoreSessionStorage)
    assert isinstance(wiring.user_profile_store, DocumentStoreUserProfileStore)
    assert wiring.organization_profile_store is None


def test_build_session_manager_from_environment_respects_memory_profile_flags(
    tmp_path: Path,
) -> None:
    create_sqlite_integration(data_dir=tmp_path)
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="mem.session")
    env.memory_profile = MemoryProfile(
        enable_user_memory=True,
        enable_org_memory=True,
        enable_long_term_memory=False,
        enable_task_memory=False,
    )
    wiring = resolve_memory_platform_wiring(env)
    session_manager = build_session_manager_from_environment(env, memory_wiring=wiring)

    assert session_manager is not None
    assert session_manager._user_profile_manager is not None
    assert session_manager._organization_profile_manager is not None


def test_build_session_manager_skips_managers_when_memory_flags_disabled(
    tmp_path: Path,
) -> None:
    create_sqlite_integration(data_dir=tmp_path)
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="mem.session.off")
    env.memory_profile = MemoryProfile(
        enable_user_memory=False,
        enable_org_memory=False,
        enable_long_term_memory=False,
        enable_task_memory=False,
    )
    wiring = resolve_memory_platform_wiring(env)
    session_manager = build_session_manager_from_environment(env, memory_wiring=wiring)

    assert session_manager._user_profile_manager is None
    assert session_manager._organization_profile_manager is None
    assert isinstance(session_manager._storage, SQLiteSessionStorage)

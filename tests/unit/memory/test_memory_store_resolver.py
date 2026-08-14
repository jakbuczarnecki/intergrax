# © Artur Czarnecki. All rights reserved.

"""ENTERPRISE-5 / BLOCK D: typed Memory store plugin classifier and resolver."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.memory_wiring import resolve_memory_platform_wiring
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    MemoryProfile,
)
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.core.memory_bootstrap import discover_session_turn_index_plugin_types
from intergrax.core.plugins.discovery import EP_MEMORY_STORES, reset_entry_point_spec_cache_for_tests
from intergrax.memory.resolver import (
    MemoryStoreMaterializationContext,
    MemoryStorePluginKind,
    MemoryStorePluginResolutionError,
    classify_memory_store_plugin,
    discover_classified_memory_store_plugins,
    materialize_session_storage,
    materialize_user_profile_store,
)
from intergrax.memory.resolver.discovery import index_classified_memory_store_plugins
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from tests.fixtures.plugin_packages.memory_store_plugin.memory_store_plugin.plugin import (
    ExternalInMemorySessionStoragePlugin,
    ExternalInMemoryUserProfileStorePlugin,
    FixtureExternalSessionStorage,
    FixtureExternalUserProfileStore,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_entry_point_spec_cache() -> None:
    reset_entry_point_spec_cache_for_tests()
    yield
    reset_entry_point_spec_cache_for_tests()


class _WrongKindSessionPlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "test.wrong_kind_session"

    @classmethod
    def create_session_storage(cls, **_kwargs):
        return InMemorySessionStorage()


class _InvalidUserProfilePlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "test.invalid_user_profile"

    @classmethod
    def create_user_profile_store(cls, **_kwargs):
        return object()


class _BrokenUserProfilePlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "test.broken_user_profile"

    @classmethod
    def create_user_profile_store(cls, **_kwargs):
        raise RuntimeError("materialization failed")


class _DuplicateIdUserProfilePluginA:
    @classmethod
    def plugin_id(cls) -> str:
        return "test.duplicate_id"

    @classmethod
    def create_user_profile_store(cls, **_kwargs):
        return InMemoryUserProfileStore()


class _DuplicateIdUserProfilePluginB:
    @classmethod
    def plugin_id(cls) -> str:
        return "test.duplicate_id"

    @classmethod
    def create_user_profile_store(cls, **_kwargs):
        return InMemoryUserProfileStore()


def test_classifier_recognizes_fixture_plugins() -> None:
    assert (
        classify_memory_store_plugin(ExternalInMemoryUserProfileStorePlugin)
        is MemoryStorePluginKind.USER_PROFILE_STORE
    )
    assert (
        classify_memory_store_plugin(ExternalInMemorySessionStoragePlugin)
        is MemoryStorePluginKind.SESSION_STORAGE
    )


def test_classifier_rejects_unsupported_target() -> None:
    class _NotAPlugin:
        pass

    assert classify_memory_store_plugin(_NotAPlugin) is None


def test_duplicate_plugin_id_fails_closed() -> None:
    records = discover_classified_memory_store_plugins(
        discover_entry_points=False,
        explicit_plugins=(_DuplicateIdUserProfilePluginA, _DuplicateIdUserProfilePluginB),
    )
    with pytest.raises(MemoryStorePluginResolutionError, match="Duplicate memory store plugin_id"):
        index_classified_memory_store_plugins(records)


def test_materialize_external_user_profile_store() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.resolver.user")
    ctx = MemoryStoreMaterializationContext(
        env=env,
        tenant_id="tenant-a",
        integration_profile=IntegrationProfile(),
        selected_plugin_id="external.in_memory_user_profile",
    )
    store = materialize_user_profile_store(
        "external.in_memory_user_profile",
        ctx,
        discover_entry_points=False,
        explicit_plugins=(ExternalInMemoryUserProfileStorePlugin,),
    )
    assert isinstance(store, FixtureExternalUserProfileStore)


def test_materialize_external_session_storage() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.resolver.session")
    ctx = MemoryStoreMaterializationContext(
        env=env,
        tenant_id="tenant-a",
        integration_profile=IntegrationProfile(),
        selected_plugin_id="external.in_memory_session_storage",
    )
    store = materialize_session_storage(
        "external.in_memory_session_storage",
        ctx,
        discover_entry_points=False,
        explicit_plugins=(ExternalInMemorySessionStoragePlugin,),
    )
    assert isinstance(store, FixtureExternalSessionStorage)


def test_resolve_memory_platform_wiring_external_user_profile_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.wiring.external.user")
    env.integration_profile = IntegrationProfile()
    env.memory_profile = MemoryProfile(
        user_profile_store_plugin_id="external.in_memory_user_profile",
    )
    wiring = resolve_memory_platform_wiring(
        env,
        discover_entry_points=True,
        explicit_memory_plugins=(ExternalInMemoryUserProfileStorePlugin,),
    )
    assert isinstance(wiring.user_profile_store, FixtureExternalUserProfileStore)
    assert isinstance(wiring.session_storage, InMemorySessionStorage)


def test_resolve_memory_platform_wiring_external_session_storage_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.wiring.external.session")
    env.integration_profile = IntegrationProfile()
    env.memory_profile = MemoryProfile(
        session_storage_plugin_id="external.in_memory_session_storage",
    )
    wiring = resolve_memory_platform_wiring(
        env,
        discover_entry_points=True,
        explicit_memory_plugins=(ExternalInMemorySessionStoragePlugin,),
    )
    assert isinstance(wiring.session_storage, FixtureExternalSessionStorage)
    assert isinstance(wiring.user_profile_store, InMemoryUserProfileStore)


def test_resolve_memory_platform_wiring_both_external() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.wiring.external.both")
    env.integration_profile = IntegrationProfile()
    env.memory_profile = MemoryProfile(
        user_profile_store_plugin_id="external.in_memory_user_profile",
        session_storage_plugin_id="external.in_memory_session_storage",
    )
    wiring = resolve_memory_platform_wiring(
        env,
        discover_entry_points=True,
        explicit_memory_plugins=(
            ExternalInMemoryUserProfileStorePlugin,
            ExternalInMemorySessionStoragePlugin,
        ),
    )
    assert isinstance(wiring.user_profile_store, FixtureExternalUserProfileStore)
    assert isinstance(wiring.session_storage, FixtureExternalSessionStorage)


def test_resolve_memory_platform_wiring_unknown_plugin_fails() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.wiring.unknown")
    env.integration_profile = IntegrationProfile()
    env.memory_profile = MemoryProfile(user_profile_store_plugin_id="missing.plugin")
    with pytest.raises(MemoryStorePluginResolutionError, match="not available"):
        resolve_memory_platform_wiring(env, discover_entry_points=True)


def test_resolve_memory_platform_wiring_wrong_kind_fails() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.wiring.wrong_kind")
    env.integration_profile = IntegrationProfile()
    env.memory_profile = MemoryProfile(user_profile_store_plugin_id="test.wrong_kind_session")
    with pytest.raises(MemoryStorePluginResolutionError, match="expected user_profile_store"):
        resolve_memory_platform_wiring(
            env,
            discover_entry_points=True,
            explicit_memory_plugins=(_WrongKindSessionPlugin,),
        )


def test_resolve_memory_platform_wiring_discovery_disabled_fails() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.wiring.no_discover")
    env.integration_profile = IntegrationProfile()
    env.memory_profile = MemoryProfile(user_profile_store_plugin_id="external.in_memory_user_profile")
    with pytest.raises(MemoryStorePluginResolutionError, match="discovery"):
        resolve_memory_platform_wiring(
            env,
            discover_entry_points=False,
            explicit_memory_plugins=(ExternalInMemoryUserProfileStorePlugin,),
        )


def test_materialize_invalid_return_fails() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.resolver.invalid")
    ctx = MemoryStoreMaterializationContext(
        env=env,
        tenant_id=None,
        integration_profile=IntegrationProfile(),
    )
    with pytest.raises(MemoryStorePluginResolutionError, match="invalid UserProfileStore"):
        materialize_user_profile_store(
            "test.invalid_user_profile",
            ctx,
            discover_entry_points=False,
            explicit_plugins=(_InvalidUserProfilePlugin,),
        )


def test_materialize_exception_fails() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.resolver.broken")
    ctx = MemoryStoreMaterializationContext(
        env=env,
        tenant_id=None,
        integration_profile=IntegrationProfile(),
    )
    with pytest.raises(MemoryStorePluginResolutionError, match="failed to materialize"):
        materialize_user_profile_store(
            "test.broken_user_profile",
            ctx,
            discover_entry_points=False,
            explicit_plugins=(_BrokenUserProfilePlugin,),
        )


def test_fixture_ep_discovery_materializes_external_stores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.metadata

    from tests.unit.core.plugins.test_plugin_discovery import _EntryPoint, _EntryPoints

    user_profile_ep = (
        "tests.fixtures.plugin_packages.memory_store_plugin.memory_store_plugin.plugin:"
        "ExternalInMemoryUserProfileStorePlugin"
    )
    session_storage_ep = (
        "tests.fixtures.plugin_packages.memory_store_plugin.memory_store_plugin.plugin:"
        "ExternalInMemorySessionStoragePlugin"
    )
    entries = _EntryPoints(
        [
            _EntryPoint("external_user_profile", user_profile_ep, EP_MEMORY_STORES),
            _EntryPoint("external_session_storage", session_storage_ep, EP_MEMORY_STORES),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.wiring.fixture_ep")
    env.integration_profile = IntegrationProfile()
    env.memory_profile = MemoryProfile(
        user_profile_store_plugin_id="external.in_memory_user_profile",
        session_storage_plugin_id="external.in_memory_session_storage",
    )
    wiring = resolve_memory_platform_wiring(env, discover_entry_points=True)

    assert isinstance(wiring.user_profile_store, FixtureExternalUserProfileStore)
    assert isinstance(wiring.session_storage, FixtureExternalSessionStorage)


def test_discover_session_turn_index_plugin_types_uses_classifier() -> None:
    class _TurnIndexPlugin:
        @classmethod
        def plugin_id(cls) -> str:
            return "test.turn_index"

        @classmethod
        def create_session_turn_index(cls, **_kwargs):
            raise NotImplementedError

    plugins = discover_session_turn_index_plugin_types()
    assert isinstance(plugins, list)
    assert _TurnIndexPlugin not in plugins

    classified = discover_classified_memory_store_plugins(
        discover_entry_points=False,
        explicit_plugins=(_TurnIndexPlugin,),
    )
    assert classified[0].kind is MemoryStorePluginKind.SESSION_TURN_INDEX


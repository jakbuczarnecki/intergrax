# © Artur Czarnecki. All rights reserved.

"""ENTERPRISE-5 / BLOCK D: typed Memory store plugin classifier and resolver."""

from __future__ import annotations

import importlib.metadata

import pytest

from intergrax.applications._shared.memory_wiring import resolve_memory_platform_wiring
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    MemoryProfile,
)
from intergrax.core.memory_bootstrap import discover_session_turn_index_plugin_types
from intergrax.core.plugins.admission import PluginAdmissionReasonCode
from intergrax.core.plugins.discovery import EP_MEMORY_STORES, reset_entry_point_spec_cache_for_tests
from intergrax.core.plugins.errors import PluginLoadError
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.memory.resolver import (
    MemoryStoreMaterializationContext,
    MemoryStorePluginCatalog,
    MemoryStorePluginKind,
    MemoryStorePluginResolutionError,
    classify_memory_store_plugin,
    discover_classified_memory_store_plugins,
    materialize_session_storage,
    materialize_user_profile_store,
)
from intergrax.memory.resolver.discovery import index_classified_memory_store_plugins
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_store import UserProfileStore
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_storage import SessionStorage
from tests.fixtures.plugin_packages.memory_store_plugin.memory_store_plugin.plugin import (
    ExternalInMemorySessionStoragePlugin,
    ExternalInMemoryUserProfileStorePlugin,
    FixtureExternalSessionStorage,
    FixtureExternalUserProfileStore,
)
from tests.unit.core.plugins.test_plugin_discovery import _EntryPoint, _EntryPoints

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


class _IncompleteUserProfileStore:
    async def get_profile(self, *, tenant_id: str, user_id: str):
        raise NotImplementedError


class _IncompleteSessionStorage:
    async def get_session(self, *, tenant_id: str, session_id: str):
        raise NotImplementedError


class _UnsupportedMemoryStoreTarget:
    pass


def _catalog(*plugins: type) -> MemoryStorePluginCatalog:
    return MemoryStorePluginCatalog.from_discovery(
        discover_classified_memory_store_plugins(
            discover_entry_points=False,
            explicit_plugins=plugins,
        )
    )


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
    result = discover_classified_memory_store_plugins(
        discover_entry_points=False,
        explicit_plugins=(_DuplicateIdUserProfilePluginA, _DuplicateIdUserProfilePluginB),
    )
    with pytest.raises(MemoryStorePluginResolutionError, match="Duplicate memory store plugin_id"):
        index_classified_memory_store_plugins(result.plugins)


def test_materialize_external_user_profile_store() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.resolver.user")
    ctx = MemoryStoreMaterializationContext(
        env=env,
        tenant_id="tenant-a",
        integration_profile=IntegrationProfile(),
    )
    store = materialize_user_profile_store(
        "external.in_memory_user_profile",
        ctx,
        catalog=_catalog(ExternalInMemoryUserProfileStorePlugin),
    )
    assert isinstance(store, FixtureExternalUserProfileStore)


def test_materialize_external_session_storage() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.resolver.session")
    ctx = MemoryStoreMaterializationContext(
        env=env,
        tenant_id="tenant-a",
        integration_profile=IntegrationProfile(),
    )
    store = materialize_session_storage(
        "external.in_memory_session_storage",
        ctx,
        catalog=_catalog(ExternalInMemorySessionStoragePlugin),
    )
    assert isinstance(store, FixtureExternalSessionStorage)


def test_resolve_memory_platform_wiring_external_user_profile_only() -> None:
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
    assert wiring.memory_store_plugin_load_report.group == EP_MEMORY_STORES


def test_resolve_memory_platform_wiring_external_session_storage_only() -> None:
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


def test_resolve_memory_platform_wiring_explicit_user_profile_only() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.wiring.explicit.user")
    env.integration_profile = IntegrationProfile()
    env.memory_profile = MemoryProfile(
        user_profile_store_plugin_id="external.in_memory_user_profile",
    )
    wiring = resolve_memory_platform_wiring(
        env,
        discover_entry_points=False,
        explicit_memory_plugins=(ExternalInMemoryUserProfileStorePlugin,),
    )
    assert isinstance(wiring.user_profile_store, FixtureExternalUserProfileStore)
    assert isinstance(wiring.session_storage, InMemorySessionStorage)
    assert wiring.memory_store_plugin_load_report.accepted == ()
    assert wiring.memory_store_plugin_load_report.rejected == ()
    assert wiring.memory_store_plugin_load_report.failed == ()


def test_resolve_memory_platform_wiring_explicit_session_storage_only() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.wiring.explicit.session")
    env.integration_profile = IntegrationProfile()
    env.memory_profile = MemoryProfile(
        session_storage_plugin_id="external.in_memory_session_storage",
    )
    wiring = resolve_memory_platform_wiring(
        env,
        discover_entry_points=False,
        explicit_memory_plugins=(ExternalInMemorySessionStoragePlugin,),
    )
    assert isinstance(wiring.session_storage, FixtureExternalSessionStorage)
    assert isinstance(wiring.user_profile_store, InMemoryUserProfileStore)


def test_resolve_memory_platform_wiring_explicit_both() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.wiring.explicit.both")
    env.integration_profile = IntegrationProfile()
    env.memory_profile = MemoryProfile(
        user_profile_store_plugin_id="external.in_memory_user_profile",
        session_storage_plugin_id="external.in_memory_session_storage",
    )
    wiring = resolve_memory_platform_wiring(
        env,
        discover_entry_points=False,
        explicit_memory_plugins=(
            ExternalInMemoryUserProfileStorePlugin,
            ExternalInMemorySessionStoragePlugin,
        ),
    )
    assert isinstance(wiring.user_profile_store, FixtureExternalUserProfileStore)
    assert isinstance(wiring.session_storage, FixtureExternalSessionStorage)


def test_resolve_memory_platform_wiring_explicit_no_candidates_fails() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.wiring.explicit.no_candidates")
    env.integration_profile = IntegrationProfile()
    env.memory_profile = MemoryProfile(user_profile_store_plugin_id="external.in_memory_user_profile")
    with pytest.raises(MemoryStorePluginResolutionError, match="explicit_memory_plugins"):
        resolve_memory_platform_wiring(env, discover_entry_points=False)


def test_resolve_memory_platform_wiring_explicit_unknown_id_fails() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.wiring.explicit.unknown")
    env.integration_profile = IntegrationProfile()
    env.memory_profile = MemoryProfile(user_profile_store_plugin_id="missing.plugin")
    with pytest.raises(MemoryStorePluginResolutionError, match="not available"):
        resolve_memory_platform_wiring(
            env,
            discover_entry_points=False,
            explicit_memory_plugins=(ExternalInMemoryUserProfileStorePlugin,),
        )


def test_resolve_memory_platform_wiring_explicit_duplicate_id_fails() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.wiring.explicit.duplicate")
    env.integration_profile = IntegrationProfile()
    env.memory_profile = MemoryProfile(user_profile_store_plugin_id="test.duplicate_id")
    with pytest.raises(MemoryStorePluginResolutionError, match="Duplicate memory store plugin_id"):
        resolve_memory_platform_wiring(
            env,
            discover_entry_points=False,
            explicit_memory_plugins=(_DuplicateIdUserProfilePluginA, _DuplicateIdUserProfilePluginB),
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
            catalog=_catalog(_InvalidUserProfilePlugin),
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
            catalog=_catalog(_BrokenUserProfilePlugin),
        )


def test_fixture_ep_discovery_materializes_external_stores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    assert len(wiring.memory_store_plugin_load_report.accepted) == 2
    assert wiring.memory_store_plugin_load_report.registered_count == 2


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

    result = discover_classified_memory_store_plugins(
        discover_entry_points=False,
        explicit_plugins=(_TurnIndexPlugin,),
    )
    assert result.plugins[0].kind is MemoryStorePluginKind.SESSION_TURN_INDEX


def test_discovery_healthy_ep_accepted_report(monkeypatch: pytest.MonkeyPatch) -> None:
    user_profile_ep = (
        "tests.fixtures.plugin_packages.memory_store_plugin.memory_store_plugin.plugin:"
        "ExternalInMemoryUserProfileStorePlugin"
    )
    entries = _EntryPoints(
        [_EntryPoint("external_user_profile", user_profile_ep, EP_MEMORY_STORES)]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    result = discover_classified_memory_store_plugins(discover_entry_points=True)

    assert len(result.plugins) == 1
    assert len(result.load_report.accepted) == 1
    assert result.load_report.registered_count == 1


def test_discovery_broken_ep_isolated_into_failed_report(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = _EntryPoints(
        [_EntryPoint("broken_ep", "not-a-valid-target", EP_MEMORY_STORES)]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    result = discover_classified_memory_store_plugins(discover_entry_points=True)

    assert result.plugins == ()
    assert len(result.load_report.failed) == 1
    assert isinstance(result.load_report.failed[0].error, PluginLoadError)


def test_discovery_unsupported_target_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint(
                "unsupported_ep",
                f"{__name__}:_UnsupportedMemoryStoreTarget",
                EP_MEMORY_STORES,
            )
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    result = discover_classified_memory_store_plugins(discover_entry_points=True)

    assert result.plugins == ()
    assert len(result.load_report.rejected) == 1
    assert (
        result.load_report.rejected[0].reason_code
        is PluginAdmissionReasonCode.INVALID_TARGET_TYPE
    )


def test_discovery_factory_resolution_failure_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    def _broken_factory() -> type:
        raise RuntimeError("factory boom")

    entries = _EntryPoints(
        [_EntryPoint("broken_factory", f"{__name__}:_broken_factory", EP_MEMORY_STORES)]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    result = discover_classified_memory_store_plugins(discover_entry_points=True)

    assert result.plugins == ()
    assert len(result.load_report.failed) == 1
    assert isinstance(result.load_report.failed[0].error, PluginLoadError)


def test_selected_failed_ep_precise_error(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = _EntryPoints(
        [_EntryPoint("failed.plugin", "not-a-valid-target", EP_MEMORY_STORES)]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.wiring.failed_ep")
    env.integration_profile = IntegrationProfile()
    env.memory_profile = MemoryProfile(user_profile_store_plugin_id="failed.plugin")
    with pytest.raises(MemoryStorePluginResolutionError, match="failed during entry-point loading"):
        resolve_memory_platform_wiring(env, discover_entry_points=True)


def test_selected_rejected_ep_precise_error(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint(
                "rejected.plugin",
                f"{__name__}:_UnsupportedMemoryStoreTarget",
                EP_MEMORY_STORES,
            )
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.wiring.rejected_ep")
    env.integration_profile = IntegrationProfile()
    env.memory_profile = MemoryProfile(user_profile_store_plugin_id="rejected.plugin")
    with pytest.raises(MemoryStorePluginResolutionError, match="was rejected"):
        resolve_memory_platform_wiring(env, discover_entry_points=True)


def test_unrelated_failed_sibling_does_not_block_selected_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_profile_ep = (
        "tests.fixtures.plugin_packages.memory_store_plugin.memory_store_plugin.plugin:"
        "ExternalInMemoryUserProfileStorePlugin"
    )
    entries = _EntryPoints(
        [
            _EntryPoint("external_user_profile", user_profile_ep, EP_MEMORY_STORES),
            _EntryPoint("broken_sibling", "not-a-valid-target", EP_MEMORY_STORES),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.wiring.isolate_sibling")
    env.integration_profile = IntegrationProfile()
    env.memory_profile = MemoryProfile(
        user_profile_store_plugin_id="external.in_memory_user_profile",
    )
    wiring = resolve_memory_platform_wiring(env, discover_entry_points=True)

    assert isinstance(wiring.user_profile_store, FixtureExternalUserProfileStore)
    assert len(wiring.memory_store_plugin_load_report.failed) == 1


def test_baseline_wiring_has_empty_plugin_load_report() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.wiring.baseline.report")
    env.integration_profile = IntegrationProfile()
    wiring = resolve_memory_platform_wiring(env, discover_entry_points=False)
    assert wiring.memory_store_plugin_load_report.accepted == ()
    assert wiring.memory_store_plugin_load_report.failed == ()
    assert wiring.memory_store_plugin_load_report.rejected == ()


def test_canonical_user_profile_store_runtime_checkable_valid() -> None:
    assert isinstance(InMemoryUserProfileStore(), UserProfileStore)


def test_canonical_user_profile_store_runtime_checkable_invalid() -> None:
    assert not isinstance(_IncompleteUserProfileStore(), UserProfileStore)


def test_canonical_session_storage_runtime_checkable_valid() -> None:
    assert isinstance(InMemorySessionStorage(), SessionStorage)


def test_canonical_session_storage_runtime_checkable_invalid() -> None:
    assert not isinstance(_IncompleteSessionStorage(), SessionStorage)


def test_tenant_id_propagation_to_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _TenantCapturingPlugin:
        @classmethod
        def plugin_id(cls) -> str:
            return "test.tenant_capture"

        @classmethod
        def create_user_profile_store(cls, **kwargs):
            captured.update(kwargs)
            return InMemoryUserProfileStore()

    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.resolver.tenant")
    env.integration_profile = IntegrationProfile()
    env.memory_profile = MemoryProfile(user_profile_store_plugin_id="test.tenant_capture")
    resolve_memory_platform_wiring(
        env,
        tenant_id="tenant-z",
        discover_entry_points=False,
        explicit_memory_plugins=(_TenantCapturingPlugin,),
    )
    assert captured.get("tenant_id") == "tenant-z"


def test_both_external_stores_use_single_discovery_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = discover_classified_memory_store_plugins

    def _counting_discovery(**kwargs):
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(
        "intergrax.applications._shared.memory_wiring.discover_classified_memory_store_plugins",
        _counting_discovery,
    )
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.wiring.single_discovery")
    env.integration_profile = IntegrationProfile()
    env.memory_profile = MemoryProfile(
        user_profile_store_plugin_id="external.in_memory_user_profile",
        session_storage_plugin_id="external.in_memory_session_storage",
    )
    resolve_memory_platform_wiring(
        env,
        discover_entry_points=False,
        explicit_memory_plugins=(
            ExternalInMemoryUserProfileStorePlugin,
            ExternalInMemorySessionStoragePlugin,
        ),
    )
    assert calls == 1

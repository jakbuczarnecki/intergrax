# © Artur Czarnecki. All rights reserved.

"""MEM-HARDEN-FINAL-4: specialized memory platform composition."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.canonical_memory_governance_wiring import (
    resolve_canonical_memory_governance_source_authority,
)
from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.memory_wiring import resolve_memory_platform_wiring
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    MemoryProfile,
)
from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest
from intergrax.memory.contracts.long_horizon_memory import CanonicalMemorySourceAuthority
from intergrax.memory.contracts.memory_security_governance import (
    CanonicalMemoryGovernanceEntryReader,
)
from intergrax.memory.memory_security_governance_service import (
    build_default_memory_security_governance_service,
)
from intergrax.memory.resolver import MemoryStorePluginResolutionError
from intergrax.memory.stores.in_memory_long_horizon_memory_store import (
    InMemoryLongHorizonMemoryStore,
)
from intergrax.memory.stores.in_memory_procedural_memory_store import (
    InMemoryProceduralMemoryStore,
)
from intergrax.memory.contracts.memory_store_creation_context import (
    LongHorizonMemoryStoreCreationContext,
    ProceduralMemoryStoreCreationContext,
)
from intergrax.memory.stores.in_memory_long_horizon_memory_plugin import (
    InMemoryLongHorizonMemoryStorePlugin,
)
from intergrax.memory.stores.in_memory_procedural_memory_plugin import (
    InMemoryProceduralMemoryStorePlugin,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


class _EmptyGovernanceReader(CanonicalMemoryGovernanceEntryReader):
    def read_canonical_governance_entry(
        self,
        scope: object,
        memory_id: str,
        revision: int,
    ) -> object | None:
        _ = (scope, memory_id, revision)
        return None


class _FixedLongHorizonSourceAuthority(CanonicalMemorySourceAuthority):
    def resolve_canonical_source(
        self,
        scope: object,
        memory_id: str,
        revision: int,
    ) -> object:
        from intergrax.memory.contracts.long_horizon_memory import CanonicalMemorySourceSnapshot

        _ = scope
        return CanonicalMemorySourceSnapshot(
            memory_id=memory_id,
            revision=revision,
            content="c",
            observed_at="2025-01-01T00:00:00+00:00",
        )


def test_specialized_memory_disabled_flags_yield_empty_surfaces() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.specialized.off")
    env.memory_profile = MemoryProfile(
        enable_procedural_memory=False,
        enable_long_horizon_memory=False,
    )
    wiring = resolve_memory_platform_wiring(env)
    assert wiring.specialized_memory.procedural_memory_store is None
    assert wiring.specialized_memory.long_horizon_memory_store is None
    assert wiring.specialized_memory.procedural_memory_capability is None
    assert wiring.specialized_memory.long_horizon_memory_capability is None


def test_specialized_memory_enabled_flags_materialize_stores_via_platform_wiring() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.specialized.on")
    env.memory_profile.enable_procedural_memory = True
    env.memory_profile.enable_long_horizon_memory = True
    wiring = resolve_memory_platform_wiring(env)
    assert isinstance(
        wiring.specialized_memory.procedural_memory_store,
        InMemoryProceduralMemoryStore,
    )
    assert isinstance(
        wiring.specialized_memory.long_horizon_memory_store,
        InMemoryLongHorizonMemoryStore,
    )
    assert wiring.specialized_memory.procedural_memory_capability is None
    assert wiring.specialized_memory.long_horizon_memory_capability is None


def test_specialized_memory_capabilities_when_authorities_supplied() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.specialized.cap")
    env.memory_profile.enable_procedural_memory = True
    env.memory_profile.enable_long_horizon_memory = True
    shared = build_default_memory_security_governance_service()
    governance_authority = resolve_canonical_memory_governance_source_authority(
        _EmptyGovernanceReader(),
    )
    wiring = resolve_memory_platform_wiring(
        env,
        security_governance=shared,
        governance_source_authority=governance_authority,
        long_horizon_source_authority=_FixedLongHorizonSourceAuthority(),
    )
    assert wiring.specialized_memory.procedural_memory_capability is not None
    assert wiring.specialized_memory.long_horizon_memory_capability is not None
    assert (
        wiring.specialized_memory.procedural_memory_capability._security_governance
        is shared
    )
    assert (
        wiring.specialized_memory.long_horizon_memory_capability._security_governance
        is shared
    )


def test_specialized_memory_unknown_plugin_id_fails_closed() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.specialized.bad")
    env.memory_profile = MemoryProfile(
        enable_procedural_memory=True,
        procedural_memory_store_plugin_id="plugin.does.not.exist",
    )
    with pytest.raises(MemoryStorePluginResolutionError):
        resolve_memory_platform_wiring(
            env,
            discover_entry_points=False,
            explicit_memory_plugins=(InMemoryProceduralMemoryStorePlugin,),
        )


class _CountingProceduralMemoryStorePlugin:
    factory_calls = 0

    @classmethod
    def plugin_id(cls) -> str:
        return "test.counting.procedural"

    @classmethod
    def create_procedural_memory_store(
        cls,
        context: ProceduralMemoryStoreCreationContext,
    ) -> InMemoryProceduralMemoryStore:
        _ = context
        cls.factory_calls += 1
        return InMemoryProceduralMemoryStore()


class _CountingLongHorizonMemoryStorePlugin:
    factory_calls = 0

    @classmethod
    def plugin_id(cls) -> str:
        return "test.counting.long_horizon"

    @classmethod
    def create_long_horizon_memory_store(
        cls,
        context: LongHorizonMemoryStoreCreationContext,
    ) -> InMemoryLongHorizonMemoryStore:
        _ = context
        cls.factory_calls += 1
        return InMemoryLongHorizonMemoryStore()


def test_specialized_procedural_factory_materialized_once() -> None:
    _CountingProceduralMemoryStorePlugin.factory_calls = 0
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.specialized.proc.once")
    env.memory_profile.enable_procedural_memory = True
    env.memory_profile.procedural_memory_store_plugin_id = (
        _CountingProceduralMemoryStorePlugin.plugin_id()
    )
    governance_authority = resolve_canonical_memory_governance_source_authority(
        _EmptyGovernanceReader(),
    )
    resolve_memory_platform_wiring(
        env,
        discover_entry_points=False,
        explicit_memory_plugins=(_CountingProceduralMemoryStorePlugin,),
        governance_source_authority=governance_authority,
    )
    assert _CountingProceduralMemoryStorePlugin.factory_calls == 1


def test_specialized_long_horizon_factory_materialized_once() -> None:
    _CountingLongHorizonMemoryStorePlugin.factory_calls = 0
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.specialized.lh.once")
    env.memory_profile.enable_long_horizon_memory = True
    env.memory_profile.long_horizon_memory_store_plugin_id = (
        _CountingLongHorizonMemoryStorePlugin.plugin_id()
    )
    governance_authority = resolve_canonical_memory_governance_source_authority(
        _EmptyGovernanceReader(),
    )
    resolve_memory_platform_wiring(
        env,
        discover_entry_points=False,
        explicit_memory_plugins=(_CountingLongHorizonMemoryStorePlugin,),
        governance_source_authority=governance_authority,
        long_horizon_source_authority=_FixedLongHorizonSourceAuthority(),
    )
    assert _CountingLongHorizonMemoryStorePlugin.factory_calls == 1


def test_specialized_procedural_capability_uses_exposed_store_instance() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.specialized.proc.identity")
    env.memory_profile.enable_procedural_memory = True
    governance_authority = resolve_canonical_memory_governance_source_authority(
        _EmptyGovernanceReader(),
    )
    wiring = resolve_memory_platform_wiring(
        env,
        discover_entry_points=False,
        governance_source_authority=governance_authority,
    )
    capability = wiring.specialized_memory.procedural_memory_capability
    assert capability is not None
    assert wiring.specialized_memory.procedural_memory_store is capability._store


def test_specialized_long_horizon_capability_uses_exposed_store_instance() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.specialized.lh.identity")
    env.memory_profile.enable_long_horizon_memory = True
    governance_authority = resolve_canonical_memory_governance_source_authority(
        _EmptyGovernanceReader(),
    )
    wiring = resolve_memory_platform_wiring(
        env,
        discover_entry_points=False,
        governance_source_authority=governance_authority,
        long_horizon_source_authority=_FixedLongHorizonSourceAuthority(),
    )
    capability = wiring.specialized_memory.long_horizon_memory_capability
    assert capability is not None
    assert wiring.specialized_memory.long_horizon_memory_store is capability._store


def test_explicit_procedural_plugin_without_entry_point_discovery() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.specialized.proc.explicit")
    env.memory_profile.enable_procedural_memory = True
    env.memory_profile.procedural_memory_store_plugin_id = (
        _CountingProceduralMemoryStorePlugin.plugin_id()
    )
    _CountingProceduralMemoryStorePlugin.factory_calls = 0
    wiring = resolve_memory_platform_wiring(
        env,
        discover_entry_points=False,
        explicit_memory_plugins=(_CountingProceduralMemoryStorePlugin,),
    )
    assert _CountingProceduralMemoryStorePlugin.factory_calls == 1
    assert wiring.specialized_memory.procedural_memory_store is not None


def test_explicit_long_horizon_plugin_without_entry_point_discovery() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.specialized.lh.explicit")
    env.memory_profile.enable_long_horizon_memory = True
    env.memory_profile.long_horizon_memory_store_plugin_id = (
        _CountingLongHorizonMemoryStorePlugin.plugin_id()
    )
    _CountingLongHorizonMemoryStorePlugin.factory_calls = 0
    wiring = resolve_memory_platform_wiring(
        env,
        discover_entry_points=False,
        explicit_memory_plugins=(_CountingLongHorizonMemoryStorePlugin,),
    )
    assert _CountingLongHorizonMemoryStorePlugin.factory_calls == 1
    assert wiring.specialized_memory.long_horizon_memory_store is not None


def test_builtin_procedural_fallback_without_entry_point_discovery() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.specialized.proc.builtin")
    env.memory_profile.enable_procedural_memory = True
    wiring = resolve_memory_platform_wiring(env, discover_entry_points=False)
    assert isinstance(
        wiring.specialized_memory.procedural_memory_store,
        InMemoryProceduralMemoryStore,
    )


def test_builtin_long_horizon_fallback_without_entry_point_discovery() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.specialized.lh.builtin")
    env.memory_profile.enable_long_horizon_memory = True
    wiring = resolve_memory_platform_wiring(env, discover_entry_points=False)
    assert isinstance(
        wiring.specialized_memory.long_horizon_memory_store,
        InMemoryLongHorizonMemoryStore,
    )


def test_wire_application_environment_exposes_specialized_memory_stores() -> None:
    settings = LabApplicationSettings.from_env()
    env = build_lab_environment_profile(settings)
    env.memory_profile.enable_procedural_memory = True
    env.memory_profile.enable_long_horizon_memory = True
    host_wiring = wire_application_environment(
        build_lab_manifest(settings),
        env,
        conformance_check=False,
    )
    assert isinstance(
        host_wiring.specialized_memory.procedural_memory_store,
        InMemoryProceduralMemoryStore,
    )
    assert isinstance(
        host_wiring.specialized_memory.long_horizon_memory_store,
        InMemoryLongHorizonMemoryStore,
    )

# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-ZERO-GAP-1: unified Memory plugin discovery policy and evidence."""

from __future__ import annotations

import importlib.metadata
from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared import environment_wiring as environment_wiring_module
from intergrax.applications._shared.entity_graph_wiring import resolve_entity_temporal_memory_store
from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.memory_vector_wiring import build_session_turn_index_store
from intergrax.applications._shared.memory_wiring import (
    assert_strict_memory_bootstrap_acceptable,
    resolve_memory_platform_wiring,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    MemoryProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.core.plugins.discovery import EP_MEMORY_STORES, reset_entry_point_spec_cache_for_tests
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.memory.contracts.entity_temporal_memory import EntityTemporalMemoryStore
from intergrax.memory.contracts.memory_store_creation_context import (
    EntityTemporalMemoryStoreCreationContext,
)
from intergrax.memory.contracts.session_turn_index import SessionTurnIndexStoreCreationContext
from intergrax.memory.resolver import MemoryStorePluginResolutionError
from intergrax.memory.session_turn_index_service import VectorSessionTurnIndexStore
from intergrax.memory.stores.in_memory_entity_temporal_memory_store import (
    InMemoryEntityTemporalMemoryStore,
)
from intergrax.rag.bootstrap.rag_stack_bootstrap import RagStack
from intergrax.rag.profiles.rag_profile import RagProfile
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest
from tests.unit.core.plugins.test_plugin_discovery import _EntryPoint, _EntryPoints
from tests.unit.memory.test_memory_store_resolver import _UnsupportedMemoryStoreTarget

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


@pytest.fixture(autouse=True)
def _reset_entry_point_spec_cache() -> None:
    reset_entry_point_spec_cache_for_tests()
    yield
    reset_entry_point_spec_cache_for_tests()


def _rag_stack() -> RagStack:
    return RagStack(
        profile=RagProfile(),
        vectorstore_manager=MagicMock(),
        embedding_manager=MagicMock(),
        retriever_manager=MagicMock(),
        reranker_manager=MagicMock(),
        retrieval_service=MagicMock(),
    )


def test_specialized_failed_ep_evidence_propagates_without_user_session_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint("broken_sibling", "not-a-valid-target", EP_MEMORY_STORES),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.zero_gap.failed_evidence")
    env.integration_profile = IntegrationProfile()
    env.memory_profile = MemoryProfile(
        enable_procedural_memory=True,
        enable_long_horizon_memory=False,
    )
    wiring = resolve_memory_platform_wiring(env, discover_entry_points=True)

    assert len(wiring.memory_store_plugin_load_report.failed) == 1


def test_strict_fails_on_specialized_ep_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint("broken_sibling", "not-a-valid-target", EP_MEMORY_STORES),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.zero_gap.strict.failed")
    env.execution_mode = ExecutionMode.STRICT
    env.integration_profile = IntegrationProfile()
    env.memory_profile = MemoryProfile(enable_procedural_memory=True)
    wiring = resolve_memory_platform_wiring(env, discover_entry_points=True)

    with pytest.raises(MemoryStorePluginResolutionError):
        assert_strict_memory_bootstrap_acceptable(env, wiring)


def test_non_strict_preserves_failed_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint("broken_sibling", "not-a-valid-target", EP_MEMORY_STORES),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.zero_gap.non_strict")
    env.execution_mode = ExecutionMode.BALANCED
    env.integration_profile = IntegrationProfile()
    env.memory_profile = MemoryProfile(enable_procedural_memory=True)
    wiring = resolve_memory_platform_wiring(env, discover_entry_points=True)

    assert_strict_memory_bootstrap_acceptable(env, wiring)
    assert len(wiring.memory_store_plugin_load_report.failed) == 1


def test_strict_fails_on_fail_closed_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint(
                "rejected.plugin",
                f"{__name__}:_UnsupportedMemoryStoreTarget",
                EP_MEMORY_STORES,
            ),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.zero_gap.strict.rejected")
    env.execution_mode = ExecutionMode.STRICT
    env.integration_profile = IntegrationProfile()
    env.memory_profile = MemoryProfile(enable_procedural_memory=True)
    wiring = resolve_memory_platform_wiring(env, discover_entry_points=True)

    assert wiring.memory_store_plugin_load_report.rejected
    with pytest.raises(MemoryStorePluginResolutionError):
        assert_strict_memory_bootstrap_acceptable(env, wiring)


def test_entity_discovery_policy_off_skips_ep_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    import intergrax.applications._shared.memory_wiring as memory_wiring_module

    original = memory_wiring_module.discover_classified_memory_store_plugins

    def _track(**kwargs: object) -> object:
        calls.append(bool(kwargs.get("discover_entry_points")))
        return original(**kwargs)

    monkeypatch.setattr(
        memory_wiring_module,
        "discover_classified_memory_store_plugins",
        _track,
    )

    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.zero_gap.entity.off")
    env.memory_profile.enable_entity_graph_memory = True
    resolve_memory_platform_wiring(env, discover_entry_points=False)

    assert calls == [False]


class _CustomEntityTemporalPlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "test.custom.entity_temporal"

    @classmethod
    def create_entity_temporal_memory_store(
        cls,
        context: EntityTemporalMemoryStoreCreationContext,
    ) -> EntityTemporalMemoryStore:
        _ = context
        return InMemoryEntityTemporalMemoryStore()


def test_entity_explicit_plugin_without_ep_discovery() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.zero_gap.entity.explicit")
    env.memory_profile.enable_entity_graph_memory = True
    env.memory_profile.entity_temporal_memory_store_plugin_id = (
        _CustomEntityTemporalPlugin.plugin_id()
    )
    store = resolve_entity_temporal_memory_store(
        env,
        discover_entry_points=False,
        explicit_memory_plugins=(_CustomEntityTemporalPlugin,),
    )
    assert store is not None


def test_sti_discovery_policy_off_uses_builtin_vector_path() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="mem.zero_gap.sti.off")
    env.memory_profile.enable_session_vector_index = True
    store = build_session_turn_index_store(
        env,
        tenant_id="tenant-a",
        rag_stack=_rag_stack(),
        integration_profile=IntegrationProfile(),
        discover_entry_points=False,
    )
    assert isinstance(store, VectorSessionTurnIndexStore)


def test_sti_explicit_plugin_without_ep_discovery() -> None:
    class _CustomStiPlugin:
        @classmethod
        def plugin_id(cls) -> str:
            return "test.custom.sti"

        @classmethod
        def create_session_turn_index(
            cls,
            context: SessionTurnIndexStoreCreationContext,
        ) -> VectorSessionTurnIndexStore:
            return VectorSessionTurnIndexStore(
                embedding_port=context.embedding_manager,
                vectorstore_port=context.vectorstore_manager,
                index_roles=context.index_roles,
                tenant_id=context.tenant_id,
                vector_index_namespace=context.vector_index_namespace,
                workspace_id=context.workspace_id,
            )

    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="mem.zero_gap.sti.explicit")
    env.memory_profile.enable_session_vector_index = True
    store = build_session_turn_index_store(
        env,
        tenant_id="tenant-a",
        rag_stack=_rag_stack(),
        integration_profile=IntegrationProfile(),
        session_turn_index_plugins=(_CustomStiPlugin,),
        discover_entry_points=False,
    )
    assert isinstance(store, VectorSessionTurnIndexStore)


def test_platform_evidence_memory_report_matches_canonical_wiring_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []
    original = resolve_memory_platform_wiring

    def _capture(env: ApplicationEnvironmentProfile, **kwargs: object) -> object:
        result = original(env, **kwargs)
        captured.append(result.memory_store_plugin_load_report)
        return result

    monkeypatch.setattr(environment_wiring_module, "resolve_memory_platform_wiring", _capture)
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="mem.zero_gap.platform_evidence")
    env.memory_profile.enable_procedural_memory = True
    wiring = wire_application_environment(
        build_lab_manifest(settings),
        env,
        tenant_id="tenant-a",
        conformance_check=False,
    )

    assert captured
    assert wiring.platform_plugin_evidence.memory_report() is captured[0]

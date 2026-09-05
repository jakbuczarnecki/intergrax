# © Artur Czarnecki. All rights reserved.

"""AC-5 — canonical production factory invocation contract (Phase 1 + Phase 2)."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from echo.echo_agent import EchoAgent
from intergrax.agent_distribution.binding import AgentBindingFactoryReference
from intergrax.agent_distribution.roster import EffectiveRoster, EffectiveRosterEntry
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.applications._shared.runtime_agent_factory_resolver import (
    InMemoryRuntimeAgentFactoryResolver,
    RuntimeAgentFactoryResolutionError,
)
from intergrax.applications._shared.wiring import (
    build_application_registry,
    build_manifest_development_registry,
    invoke_canonical_agent_factory,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.errors import AgentImportError
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO = Path(__file__).resolve().parents[3]
_WIRING_SOURCE = REPO / "intergrax" / "applications" / "_shared" / "wiring.py"
_REGISTRY_PROJECTION_SOURCE = (
    REPO / "intergrax" / "applications" / "_shared" / "registry_projection.py"
)

_APP = "app_ac5"
_ENV = "env-ac5"
_RELEASE = "rel-ac5"
_DIGEST = "sha256:" + ("a" * 64)
_LOCK_ID = "lock-ac5"
_LOCK_DIGEST = "sha256:" + ("b" * 64)
_GRAPH_DIGEST = "sha256:" + ("c" * 64)
_ARTIFACT = "sha256:" + ("d" * 64)
_ROSTER = "sha256:" + ("e" * 64)
_ECHO_REF = AgentBindingFactoryReference(builder_key="echo")


def _echo_factory(_ctx: ApplicationBuildContext, _binding: AgentBinding) -> EchoAgent:
    return EchoAgent()


def _manifest() -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id=_APP,
        name="AC5",
        agents=[
            AgentBinding.mount(EchoAgent, contract_id="search", factory=_echo_factory),
        ],
    )


def _entry(
    logical_agent_id: str = "search",
    *,
    factory_reference: AgentBindingFactoryReference | None = None,
) -> EffectiveRosterEntry:
    return EffectiveRosterEntry(
        logical_agent_id=logical_agent_id,
        installation_slot_id=f"slot-{logical_agent_id}",
        package_digest=_DIGEST,
        distribution_package_id=f"pkg-{logical_agent_id}",
        effective_enablement=True,
        factory_reference=factory_reference or _ECHO_REF,
        manifest_origin_ref=f"manifest:agents/{logical_agent_id}",
    )


def _roster(entries: tuple[EffectiveRosterEntry, ...]) -> EffectiveRoster:
    return EffectiveRoster(
        application_id=_APP,
        application_environment_id=_ENV,
        manifest_release_id=_RELEASE,
        entries=entries,
    ).with_revision_id()


def _revision() -> RuntimeRevision:
    return RuntimeRevision(
        runtime_revision_id="rev-ac5",
        application_id=_APP,
        application_environment_id=_ENV,
        application_release_id=_RELEASE,
        platform_version="0.1.0",
        effective_roster_revision_id=_ROSTER,
        installed_agent_package_digests=(_DIGEST,),
        materialized_runtime_lock_id=_LOCK_ID,
        materialized_runtime_lock_digest=_LOCK_DIGEST,
        runtime_graph_digest=_GRAPH_DIGEST,
        materialization_artifact_digest=_ARTIFACT,
        materialization_topology=MaterializationTopology.VENV_BUNDLE,
        revision_state=RuntimeRevisionState.VALIDATED,
        activated_at=datetime.now(UTC),
    )


def _resolver_with_factory(factory: object) -> InMemoryRuntimeAgentFactoryResolver:
    resolver = InMemoryRuntimeAgentFactoryResolver()
    resolver.register(
        package_digest=_DIGEST,
        factory_reference=_ECHO_REF,
        factory=factory,
    )
    return resolver


def test_production_path_uses_canonical_invoker_not_legacy() -> None:
    source = _WIRING_SOURCE.read_text(encoding="utf-8")
    register_block_start = source.index("def _register_binding(")
    register_block = source[
        register_block_start : source.index("\ndef build_manifest_development_registry")
    ]
    assert "invoke_canonical_agent_factory" in register_block
    assert "invoke_agent_factory" not in register_block
    assert "invoke_legacy_compatible_agent_factory" not in register_block


def test_internal_typeerror_invoked_once_in_production_path() -> None:
    calls = 0

    def _factory_raises_typeerror(
        _ctx: ApplicationBuildContext,
        _binding: AgentBinding,
    ) -> EchoAgent:
        nonlocal calls
        calls += 1
        raise TypeError("internal factory failure")

    manifest = _manifest()
    ctx = ApplicationBuildContext.for_manifest(manifest)
    roster = _roster((_entry(),))
    revision = _revision()
    resolver = _resolver_with_factory(_factory_raises_typeerror)

    with pytest.raises(TypeError, match="internal factory failure"):
        build_application_registry(
            manifest,
            ctx,
            effective_roster=roster,
            runtime_revision=revision,
            factory_resolver=resolver,
        )
    assert calls == 1


def test_legacy_zero_arg_factory_works_in_development_registry() -> None:
    def legacy_zero_arg_factory() -> EchoAgent:
        return EchoAgent()

    manifest = ApplicationManifest.lab(
        app_id=_APP,
        name="AC5 dev",
        agents=[
            AgentBinding.mount(
                EchoAgent,
                contract_id="echo",
                factory=legacy_zero_arg_factory,
            ),
        ],
    )
    ctx = ApplicationBuildContext.for_manifest(manifest)
    registry = build_manifest_development_registry(manifest, ctx)
    assert registry.list_agent_ids() == ["echo"]


def test_legacy_zero_arg_factory_fails_in_revision_bound_production() -> None:
    def legacy_zero_arg_factory() -> EchoAgent:
        return EchoAgent()

    manifest = _manifest()
    ctx = ApplicationBuildContext.for_manifest(manifest)
    roster = _roster((_entry(),))
    revision = _revision()
    resolver = _resolver_with_factory(legacy_zero_arg_factory)

    with pytest.raises(TypeError):
        build_application_registry(
            manifest,
            ctx,
            effective_roster=roster,
            runtime_revision=revision,
            factory_resolver=resolver,
        )


def test_canonical_factory_registers_agent_with_identity() -> None:
    seen: list[tuple[ApplicationBuildContext, AgentBinding]] = []

    def _tracking_factory(
        ctx: ApplicationBuildContext,
        binding: AgentBinding,
    ) -> EchoAgent:
        seen.append((ctx, binding))
        return EchoAgent()

    manifest = _manifest()
    ctx = ApplicationBuildContext.for_manifest(manifest)
    roster = _roster((_entry(),))
    revision = _revision()
    resolver = _resolver_with_factory(_tracking_factory)

    registry = build_application_registry(
        manifest,
        ctx,
        effective_roster=roster,
        runtime_revision=revision,
        factory_resolver=resolver,
    )
    assert registry.list_agent_ids() == ["search"]
    assert len(seen) == 1
    assert seen[0][0] is ctx
    assert seen[0][1].contract_id == "search"


def test_invalid_factory_result_fails_closed() -> None:
    def _bad_factory(
        _ctx: ApplicationBuildContext,
        _binding: AgentBinding,
    ) -> object:
        return object()

    manifest = _manifest()
    ctx = ApplicationBuildContext.for_manifest(manifest)
    binding = manifest.agents[0]

    with pytest.raises(AgentImportError, match="must return Agent"):
        invoke_canonical_agent_factory(_bad_factory, ctx, binding)


def test_resolution_failure_does_not_invoke_factory() -> None:
    calls = 0

    def _factory(
        _ctx: ApplicationBuildContext,
        _binding: AgentBinding,
    ) -> EchoAgent:
        nonlocal calls
        calls += 1
        return EchoAgent()

    manifest = _manifest()
    ctx = ApplicationBuildContext.for_manifest(manifest)
    roster = _roster((_entry(),))
    revision = _revision()
    resolver = InMemoryRuntimeAgentFactoryResolver()
    # deliberate: factory not registered

    with pytest.raises(
        RuntimeAgentFactoryResolutionError, match="cannot resolve factory"
    ):
        build_application_registry(
            manifest,
            ctx,
            effective_roster=roster,
            runtime_revision=revision,
            factory_resolver=resolver,
        )
    assert calls == 0


def test_revision_bound_path_forbids_host_builders_fallback() -> None:
    manifest = _manifest()
    ctx = ApplicationBuildContext.for_manifest(manifest)
    roster = _roster((_entry(),))
    revision = _revision()

    with pytest.raises(
        RuntimeAgentFactoryResolutionError, match="requires RuntimeAgentFactoryResolver"
    ):
        build_application_registry(
            manifest,
            ctx,
            builders={EchoAgent: _echo_factory},
            effective_roster=roster,
            runtime_revision=revision,
            factory_resolver=None,
        )


def _revision_bound_registry_block() -> str:
    source = _WIRING_SOURCE.read_text(encoding="utf-8")
    start = source.index("    if effective_roster is None:")
    end = source.index("\ndef build_registry_from_manifest(")
    return source[start:end]


_REVISION_BOUND_FORBIDDEN = (
    "invoke_legacy_compatible_agent_factory",
    "invoke_agent_factory",
    "build_agent_from_binding",
    "load_agent_from_binding",
    "load_callable(",
    "resolved_agent_type()()",
    "builders[",
)


@pytest.mark.parametrize("token", _REVISION_BOUND_FORBIDDEN)
def test_revision_bound_assembly_has_no_legacy_or_bypass_tokens(token: str) -> None:
    block = _revision_bound_registry_block()
    assert token not in block


def test_register_binding_canonical_branch_has_no_legacy_tokens() -> None:
    source = _WIRING_SOURCE.read_text(encoding="utf-8")
    start = source.index("def _register_binding(")
    end = source.index("\ndef build_manifest_development_registry")
    block = source[start:end]
    assert "invoke_canonical_agent_factory" in block
    assert "invoke_agent_factory" not in block
    assert "invoke_legacy_compatible_agent_factory" not in block


def test_registry_projection_builds_with_canonical_resolver_path_only() -> None:
    text = _REGISTRY_PROJECTION_SOURCE.read_text(encoding="utf-8")
    start = text.index("def build_registry_projection(")
    end = text.index("\ndef ", start + 1)
    block = text[start:end]
    assert "build_application_registry(" in block
    assert "builders=None" in block
    assert "build_agent_from_binding" not in block


def _assert_canonical_factory_signature(factory: Callable[..., object]) -> None:
    sig = inspect.signature(factory)
    params = list(sig.parameters.values())
    assert len(params) == 2, f"{factory.__name__}: expected 2 parameters, got {len(params)}"
    for param in params:
        assert param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ), f"{factory.__name__}: parameter {param.name!r} must be positional or keyword"
        assert param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ), f"{factory.__name__}: variadic parameters are forbidden"
    type_hints = getattr(factory, "__annotations__", {})
    if type_hints:
        param_names = tuple(param.name for param in params)
        assert "ctx" in param_names or "ApplicationBuildContext" in str(
            type_hints.get(param_names[0], "")
        )


def _load_production_factory_inventory() -> list[tuple[str, str, Callable[..., object]]]:
    from dispute_sim_application.host.agent_factories import (
        build_dispute_sim_dispute_analyst_from_context,
        build_dispute_sim_dispute_intake_from_context,
        build_dispute_sim_dispute_scenario_from_context,
        build_dispute_sim_dispute_strategist_from_context,
    )
    from governed_contractor_application.host.agent_factories import (
        build_governed_contractor_external_contractor_adapter_from_context,
    )
    from legal_application.host.agent_factories import build_legal_agent_from_context
    from local_workspace_application.host.agent_factories import (
        build_local_workspace_local_indexer_from_context,
        build_local_workspace_local_search_from_context,
        build_local_workspace_local_synthesizer_from_context,
        build_local_workspace_model_routing_qualifier_from_context,
        build_local_workspace_tool_selection_qualifier_from_context,
        build_local_workspace_web_search_qualifier_from_context,
    )

    return [
        ("legal_application", "build_legal_agent_from_context", build_legal_agent_from_context),
        (
            "governed_contractor_application",
            "build_governed_contractor_external_contractor_adapter_from_context",
            build_governed_contractor_external_contractor_adapter_from_context,
        ),
        (
            "local_workspace_application",
            "build_local_workspace_local_indexer_from_context",
            build_local_workspace_local_indexer_from_context,
        ),
        (
            "local_workspace_application",
            "build_local_workspace_local_search_from_context",
            build_local_workspace_local_search_from_context,
        ),
        (
            "local_workspace_application",
            "build_local_workspace_local_synthesizer_from_context",
            build_local_workspace_local_synthesizer_from_context,
        ),
        (
            "local_workspace_application",
            "build_local_workspace_tool_selection_qualifier_from_context",
            build_local_workspace_tool_selection_qualifier_from_context,
        ),
        (
            "local_workspace_application",
            "build_local_workspace_web_search_qualifier_from_context",
            build_local_workspace_web_search_qualifier_from_context,
        ),
        (
            "local_workspace_application",
            "build_local_workspace_model_routing_qualifier_from_context",
            build_local_workspace_model_routing_qualifier_from_context,
        ),
        (
            "dispute_sim_application",
            "build_dispute_sim_dispute_intake_from_context",
            build_dispute_sim_dispute_intake_from_context,
        ),
        (
            "dispute_sim_application",
            "build_dispute_sim_dispute_analyst_from_context",
            build_dispute_sim_dispute_analyst_from_context,
        ),
        (
            "dispute_sim_application",
            "build_dispute_sim_dispute_strategist_from_context",
            build_dispute_sim_dispute_strategist_from_context,
        ),
        (
            "dispute_sim_application",
            "build_dispute_sim_dispute_scenario_from_context",
            build_dispute_sim_dispute_scenario_from_context,
        ),
    ]


_PRODUCTION_FACTORY_INVENTORY = _load_production_factory_inventory()
_PRODUCTION_FACTORY_IDS = [f"{app}::{name}" for app, name, _ in _PRODUCTION_FACTORY_INVENTORY]


@pytest.mark.parametrize(
    ("application", "factory_name", "factory"),
    _PRODUCTION_FACTORY_INVENTORY,
    ids=_PRODUCTION_FACTORY_IDS,
)
def test_real_production_factory_has_canonical_signature(
    application: str,
    factory_name: str,
    factory: Callable[..., object],
) -> None:
    del application, factory_name
    _assert_canonical_factory_signature(factory)


def test_real_production_factory_inventory_is_complete() -> None:
    inventory = _PRODUCTION_FACTORY_INVENTORY
    applications = {entry[0] for entry in inventory}
    assert applications == {
        "legal_application",
        "governed_contractor_application",
        "local_workspace_application",
        "dispute_sim_application",
    }
    assert len(inventory) == 12


def test_one_arg_factory_from_resolver_fails_without_fallback() -> None:
    def _settings_only_factory(_settings: object) -> EchoAgent:
        return EchoAgent()

    manifest = _manifest()
    ctx = ApplicationBuildContext.for_manifest(manifest)
    roster = _roster((_entry(),))
    revision = _revision()
    resolver = _resolver_with_factory(_settings_only_factory)

    with pytest.raises(TypeError):
        build_application_registry(
            manifest,
            ctx,
            effective_roster=roster,
            runtime_revision=revision,
            factory_resolver=resolver,
        )

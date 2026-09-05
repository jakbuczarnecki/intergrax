# © Artur Czarnecki. All rights reserved.

"""AC-5 Phase 1 — canonical production factory invocation contract."""

from __future__ import annotations

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
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO = Path(__file__).resolve().parents[3]
_WIRING_SOURCE = REPO / "intergrax" / "applications" / "_shared" / "wiring.py"

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

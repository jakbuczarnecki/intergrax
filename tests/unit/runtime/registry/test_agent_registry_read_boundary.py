# © Artur Czarnecki. All rights reserved.

"""Read-only runtime boundary for materialized registry projections."""

from __future__ import annotations

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.harness_registry_authority import (
    HarnessHostRegistryAuthorityError,
    RegistryAssemblyMode,
    resolve_harness_host_registry,
    resolve_registry_assembly_mode,
)
from intergrax.applications._shared.registry_projection import (
    MaterializedRegistryProjection,
    RegistryProjectionInputBundle,
    build_registry_projection,
)
from intergrax.applications._shared.wiring import build_manifest_development_registry
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.runtime.registry.agent_registry_read_view import (
    AgentRegistryReadView,
    freeze_agent_registry,
)
from legal_application.serving.fastapi_router import LegalAgentServingConfig
from tests.unit.applications.test_registry_projection_ap10 import (
    ECHO_BUILDERS,
    _ARTIFACT,
    _ENV,
    _ROSTER_A,
    _echo_factory,
    _entry,
    _manifest,
    _revision,
    _resolver_from_roster,
    _roster,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _build_projection() -> MaterializedRegistryProjection:
    roster = _roster((_entry("search"),))
    manifest = _manifest()
    revision = _revision(
        "rev-read-boundary",
        roster_revision_id=roster.effective_roster_revision_id or _ROSTER_A,
    )
    bundle = RegistryProjectionInputBundle(
        runtime_revision=revision,
        effective_roster=roster,
        manifest=manifest,
        build_context=ApplicationBuildContext.for_manifest(manifest),
        factory_resolver=_resolver_from_roster(roster),
        builders=ECHO_BUILDERS,
        materialization_artifact_digest=_ARTIFACT,
    )
    return build_registry_projection(bundle)


def test_materialized_projection_exposes_agent_registry_read_contract() -> None:
    projection = _build_projection()
    registry = projection.agent_registry
    assert isinstance(registry, AgentRegistryRead)
    assert isinstance(registry, AgentRegistryReadView)
    assert registry.list_agent_ids() == ["search"]
    assert registry.has("search")


def test_materialized_projection_runtime_surface_has_no_register() -> None:
    projection = _build_projection()
    registry = projection.agent_registry
    with pytest.raises(AttributeError):
        registry.register(EchoAgent())  # type: ignore[attr-defined]


def test_nexus_loop_accepts_read_only_registry_surface() -> None:
    projection = _build_projection()
    nexus = NexusLoop(projection.agent_registry)
    assert nexus.registry.list_agent_ids() == ["search"]
    assert isinstance(nexus.registry, AgentRegistryRead)


def test_development_agent_registry_is_structural_agent_registry_read() -> None:
    manifest = ApplicationManifest.lab(
        app_id="dev_read",
        name="Dev Read",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", factory=_echo_factory)],
    )
    ctx = ApplicationBuildContext.for_manifest(manifest)
    registry = build_manifest_development_registry(manifest, ctx, builders=ECHO_BUILDERS)
    assert isinstance(registry, AgentRegistry)
    assert isinstance(registry, AgentRegistryRead)
    nexus = NexusLoop(registry)
    assert nexus.registry.list_agent_ids() == ["echo"]


def test_freeze_agent_registry_wraps_mutable_registry_without_register() -> None:
    mutable = AgentRegistry()
    mutable.register(EchoAgent())
    frozen = freeze_agent_registry(mutable)
    assert isinstance(frozen, AgentRegistryReadView)
    assert frozen.list_agent_ids() == ["echo"]
    with pytest.raises(AttributeError):
        frozen.register(EchoAgent())  # type: ignore[attr-defined]


def test_revision_bound_host_registry_resolution_uses_read_only_projection_surface() -> None:
    manifest = _manifest()
    environment = ApplicationEnvironmentProfile.product_defaults(profile_id=_ENV)
    projection = _build_projection()
    registry, evidence = resolve_harness_host_registry(
        manifest=manifest,
        build_context=ApplicationBuildContext.for_manifest(manifest),
        environment=environment,
        assembly_mode=RegistryAssemblyMode.REVISION_BOUND,
        registry_projection=projection,
    )
    assert evidence is projection.evidence
    assert isinstance(registry, AgentRegistryRead)
    assert registry is projection.agent_registry
    with pytest.raises(AttributeError):
        registry.register(EchoAgent())  # type: ignore[attr-defined]


def test_strict_host_fails_closed_without_revision_bound_projection() -> None:
    manifest = _manifest()
    environment = ApplicationEnvironmentProfile.product_defaults(profile_id=_ENV)
    assembly_mode = resolve_registry_assembly_mode(environment)
    assert assembly_mode is RegistryAssemblyMode.REVISION_BOUND
    with pytest.raises(HarnessHostRegistryAuthorityError, match="MaterializedRegistryProjection"):
        resolve_harness_host_registry(
            manifest=manifest,
            build_context=ApplicationBuildContext.for_manifest(manifest),
            environment=environment,
            assembly_mode=assembly_mode,
        )


def test_legal_serving_config_accepts_read_only_registry_surface() -> None:
    projection = _build_projection()
    from unittest.mock import MagicMock

    config = LegalAgentServingConfig(
        registry=projection.agent_registry,
        default_agent_id="search",
        host_execution=MagicMock(),
    )
    assert config.registry.has("search")

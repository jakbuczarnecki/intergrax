# © Artur Czarnecki. All rights reserved.

"""Canonical minimal production-scenario composition helpers for architecture gates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.scenario_runtime_baseline import (
    ScenarioRuntimeComposition,
    build_scenario_runtime_from_environment,
)
from intergrax.applications._shared.scenario_runtime_profiles import (
    ScenarioRuntimeMode,
    build_scenario_production_runtime,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.registry.agent_registry import AgentRegistry


@dataclass(frozen=True, slots=True)
class MinimalProductionScenarioTestConfig:
    """Explicit production-scenario prerequisites for deterministic architecture gates."""

    tenant_id: str
    profile_id: str
    app_id: str
    document_store: InMemoryDocumentStore | None = None


def echo_agent_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    return registry


def minimal_echo_scenario_manifest(
    app_id: str,
    *,
    name: str | None = None,
) -> ApplicationManifest:
    resolved_name = name or app_id.replace("_", " ").title()
    route_token = app_id.replace("_", "")
    return ApplicationManifest.lab(
        app_id=app_id,
        name=resolved_name,
        route_prefix=f"/v1/{route_token}",
        env_prefix=f"{app_id.upper()}_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )


def production_attached_environment(profile_id: str) -> ApplicationEnvironmentProfile:
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id=profile_id)
    environment.execution_mode = ExecutionMode.STRICT
    return environment


def build_valid_minimal_production_scenario_fixture(
    tmp_path: Path,
    config: MinimalProductionScenarioTestConfig,
    *,
    registry: AgentRegistry | None = None,
) -> ScenarioRuntimeComposition:
    """Compose a production-attached scenario runtime with full package closure prerequisites."""
    manifest = minimal_echo_scenario_manifest(config.app_id)
    document_store = config.document_store
    return build_scenario_production_runtime(
        environment=production_attached_environment(config.profile_id),
        manifest=manifest,
        registry=registry or echo_agent_registry(),
        tenant_id=config.tenant_id,
        runtime_events_db_path=tmp_path / "events.db",
        trace_db_path=tmp_path / "trace.db",
        document_store=document_store,
    )


def build_valid_minimal_lab_scenario_fixture(
    tmp_path: Path,
    *,
    tenant_id: str,
    profile_id: str,
    app_id: str,
    registry: AgentRegistry | None = None,
    document_store: InMemoryDocumentStore | None = None,
) -> ScenarioRuntimeComposition:
    """Compose a lab scenario runtime through the same public composition root as production."""
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id=profile_id)
    return build_scenario_runtime_from_environment(
        environment=environment,
        registry=registry or echo_agent_registry(),
        tenant_id=tenant_id,
        manifest=minimal_echo_scenario_manifest(app_id),
        runtime_events_db_path=tmp_path / "events.db",
        trace_db_path=tmp_path / "trace.db",
        document_store=document_store,
        use_in_memory_trace=True,
        runtime_mode=ScenarioRuntimeMode.LAB,
    )

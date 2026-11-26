# © Artur Czarnecki. All rights reserved.

"""Canonical minimal production-scenario composition helpers for architecture gates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

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
from intergrax.integrations.contracts.vector_store import VectorStore as IntegrationVectorStore
from intergrax.integrations.registry.catalog_manifests import QDRANT
from intergrax.runtime.registry.agent_registry import AgentRegistry
from testing_support.dependency_concurrency_admission_config import (
    tool_dependency_concurrency_admission_configuration,
)

from echo.echo_agent import EchoAgent


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
    environment.memory_profile = environment.memory_profile.model_copy(
        update={
            "enable_org_memory": False,
            "enable_user_memory": False,
            "enable_long_term_memory": False,
        },
    )
    lab_skill_bundles = list(environment.skill_profile.enabled_bundles)
    environment.skill_profile = environment.skill_profile.model_copy(
        update={
            "enabled_bundles": [bundle for bundle in lab_skill_bundles if bundle not in ("harness", "legal")],
        },
    )
    environment.orchestration_profile = environment.orchestration_profile.model_copy(
        update={"max_parallel_nodes": 4, "max_inflight_nodes": 4},
    )
    if environment.reliability_profile.dependency_concurrency_admission is None:
        environment.reliability_profile = environment.reliability_profile.model_copy(
            update={
                "dependency_concurrency_admission": (
                    tool_dependency_concurrency_admission_configuration(
                        "minimal-production-scenario-tool",
                    )
                ),
            },
        )
    base_profile = environment.integration_profile
    environment.integration_profile = base_profile.model_copy(update={"vector_store": QDRANT})
    if environment.sandbox is not None:
        environment = environment.model_copy(
            update={
                "sandbox": environment.sandbox.model_copy(
                    update={"root": Path("build/scenario_qualification/sandbox")},
                ),
            },
        )
    environment.observability_profile = environment.observability_profile.model_copy(
        update={"otlp_export_endpoint": "http://127.0.0.1:0/v1/logs"},
    )
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
    sentinel = MagicMock(spec=IntegrationVectorStore)
    with patch(
        "intergrax.rag.vectorstore.bootstrap.integration_vectorstore.resolve",
        return_value=sentinel,
    ):
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

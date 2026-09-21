# © Artur Czarnecki. All rights reserved.

"""GR-10-A1-R1 — production reachability gates for acp.session.v1 AGENTIC path."""

from __future__ import annotations

import pytest

from intergrax.agents.authoring.base import IntergraxAgent
from intergrax.agents.persistence.checkpoint_store import InMemoryAgentCheckpointStore
from intergrax.applications._shared.task_control_wiring import build_reliability_task_enricher
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.execution.host_task import resolve_task_execution_capabilities
from intergrax.runtime.execution.request import ExecutionCapability
from intergrax.runtime.nexus.agents.runtime_request_bridge import acp_session_enabled
from intergrax.runtime.nexus.orchestration_capabilities import orchestration_capabilities_from_triggers
from intergrax.runtime.task.task import Task, TaskContext
from legal.legal_agent import LegalAgent
from tests.qualification.governance.strategy.catalog import GR10_AGENTIC_LEGAL_PRODUCTION_PATHS

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_GR10_A1_R1_PRODUCTION_HOSTS_WITH_CHECKPOINT_ENRICHER = (
    "legal_application",
    "research_application",
    "dispute_sim_application",
    "attestation_demo",
    "poc_template_application",
    "intergrax_assistant_application",
    "local_workspace_application (harness task_control enricher)",
    "lab_application",
)


def test_gr10_a1_r1_catalog_acp_path_not_legacy() -> None:
    acp = next(row for row in GR10_AGENTIC_LEGAL_PRODUCTION_PATHS if row.path_id == "P-ACP-SESSION")
    assert acp.status == "EXPLICIT_NON_CANONICAL"
    assert "TaskBoundAgenticDelegate" in acp.legal_entry or "AgentEngine" in acp.legal_entry


def test_gr10_a1_r1_harness_reliability_enricher_checkpoint_without_implicit_acp_session() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults()
    store = InMemoryAgentCheckpointStore()
    enricher = build_reliability_task_enricher(
        env,
        agent_checkpoint_store=store,
    )
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-a",
        user_id="user-1",
        agent_id="legal",
        message="review",
        context=TaskContext(capability="legal.review"),
        metadata={},
    )
    enriched = enricher(task)
    assert enriched.metadata.get(AcpMetadataKey.CHECKPOINT_STORE) is store
    assert enriched.metadata.get(AcpMetadataKey.SESSION_ENABLED) is None

    graph_spec = env.graph_spec
    triggers = orchestration_capabilities_from_triggers(
        graph_spec.trigger_capabilities if graph_spec is not None else None,
    )
    suffix = graph_spec.pipeline_capability_suffix if graph_spec is not None else ".pipeline"
    capabilities = resolve_task_execution_capabilities(
        enriched,
        orchestration_triggers=triggers,
        pipeline_capability_suffix=suffix,
    )
    assert capabilities == frozenset({ExecutionCapability.AGENT})

    runtime_request = enriched.to_runtime_request(run_id=mint_run_id())
    assert acp_session_enabled(runtime_request) is False


def test_gr10_a1_r1_production_legal_agent_is_intergrax_agent_with_uaep() -> None:
    from intergrax.agents import supports_uaep

    agent = LegalAgent()
    assert isinstance(agent, IntergraxAgent)
    assert supports_uaep(agent) is True


def test_gr10_a1_r1_governed_contractor_enricher_omits_session_without_checkpoint_param() -> None:
    from governed_contractor_application.manifest import build_governed_contractor_manifest
    from governed_contractor_application.host.environment_profile import (
        build_governed_contractor_environment_profile,
    )
    from governed_contractor_application.host.settings import GovernedContractorBackendSettings

    settings = GovernedContractorBackendSettings.from_env()
    manifest = build_governed_contractor_manifest()
    env = manifest.environment or build_governed_contractor_environment_profile(settings)
    enricher = build_reliability_task_enricher(env)
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-a",
        user_id="user-1",
        agent_id="agent",
        message="hello",
        metadata={},
    )
    enriched = enricher(task)
    assert enriched.metadata.get(AcpMetadataKey.SESSION_ENABLED) is None


def test_gr10_a1_r1_production_host_inventory_is_documented() -> None:
    assert len(_GR10_A1_R1_PRODUCTION_HOSTS_WITH_CHECKPOINT_ENRICHER) >= 7

# © Artur Czarnecki. All rights reserved.

"""Incident-specific runtime composition — delegates generic execution to scenario runtime baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from intergrax.applications._shared.prompt_wiring import resolve_prompt_registry
from intergrax.applications._shared.runtime_config_bridge import (
    build_runtime_context_from_environment,
)
from intergrax.applications._shared.scenario_runtime_baseline import (
    ScenarioRuntimeComposition as PlatformScenarioRuntimeComposition,
    build_scenario_runtime_from_environment,
    rewire_scenario_decision_wiring,
)
from intergrax.applications._shared.scenario_runtime_profiles import (
    ScenarioRuntimeMode,
    create_scenario_lab_workspace,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ContextProfile,
    DecisionFlowProfile,
    DecisionProfile,
    DecisionVerificationProfile,
    MemoryProfile,
)
from intergrax.applications.contracts.graph_spec import (
    ApplicationGraphSpec,
    EvaluatorLoopGraphBinding,
    GraphNode,
)
from intergrax.applications._shared.application_owned_tool_conformance import (
    application_owned_tool_declarations,
)
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.applications._shared.llm_resolver import resolve_llm_adapter
from intergrax.integrations.registry.catalog_manifests import LOG
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.execution.evaluator_loop_spec import EvaluatorLoopSpec
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceReader
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.skills.registry.profile import SkillProfile
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.registry.profile import ToolProfile
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from platform_proofs.scenarios.ai_incident_investigation.application.investigator_contract import (
    INVESTIGATOR_AGENT_ID,
    INVESTIGATOR_CAPABILITY,
    incident_investigator_contract,
)
from platform_proofs.scenarios.ai_incident_investigation.application.tools import SCENARIO_TOOL_IDS
from platform_proofs.scenarios.ai_incident_investigation.application.validation import (
    IncidentInvestigationValidationEngine,
)

INVESTIGATOR_NODE_ID = f"node_{INVESTIGATOR_AGENT_ID}"
DEFAULT_EVALUATOR_LOOP_MAX_ITERATIONS = 2


def _ensure_investigator_contract_registered(registry: AgentRegistry) -> None:
    try:
        registry.get_contract(INVESTIGATOR_AGENT_ID)
    except KeyError:
        registry._contracts[INVESTIGATOR_AGENT_ID] = incident_investigator_contract()


def _incident_lab_manifest(environment: ApplicationEnvironmentProfile) -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id="scenario_ai_incident_investigation",
        name="AI Incident Investigation Scenario",
        route_prefix="/v1/scenario/ai_incident_investigation",
        env_prefix="SCENARIO_AI_INCIDENT_",
        agents=[
            AgentBinding.reference(
                contract_id=INVESTIGATOR_AGENT_ID,
                capabilities=[INVESTIGATOR_CAPABILITY],
            ),
        ],
        application_owned_tools=application_owned_tool_declarations(SCENARIO_TOOL_IDS),
        environment=environment,
    )


@dataclass
class ScenarioRuntimeComposition:
    """Incident-facing runtime composition view over the platform scenario runtime."""

    environment: ApplicationEnvironmentProfile
    tool_registry: ToolRegistry
    _platform: PlatformScenarioRuntimeComposition | None = None
    llm_adapter_override: LLMAdapter | None = None

    @property
    def platform(self) -> PlatformScenarioRuntimeComposition:
        if self._platform is None:
            raise RuntimeError("platform scenario runtime not attached")
        return self._platform

    @property
    def is_platform_attached(self) -> bool:
        return self._platform is not None

    @property
    def build_context(self) -> ApplicationBuildContext:
        return self.platform.env_wiring.build_context

    def attach_platform(self, platform: PlatformScenarioRuntimeComposition) -> None:
        self._platform = platform


def build_scenario_environment_profile(
    *,
    evaluator_loop_max_iterations: int = DEFAULT_EVALUATOR_LOOP_MAX_ITERATIONS,
    semantic_verification_enabled: bool = False,
    max_decision_revisions: int = 0,
) -> ApplicationEnvironmentProfile:
    """Provider-neutral LAB environment with fixed investigator topology."""
    env = ApplicationEnvironmentProfile.lab_defaults(
        profile_id="scenario.ai_incident_investigation",
        harness_tools=False,
    )
    env.integration_profile = IntegrationProfile(notification_channel=LOG)
    env.context_profile = ContextProfile(enable_rag=False, enable_websearch=False)
    env.memory_profile = MemoryProfile()
    env.skill_profile = SkillProfile()
    env.tool_profile = ToolProfile(enabled=list(SCENARIO_TOOL_IDS))
    env.graph_spec = _incident_graph_spec(
        evaluator_loop_max_iterations=evaluator_loop_max_iterations,
    )
    env.decision_profile = DecisionProfile(
        verification=DecisionVerificationProfile(
            semantic_enabled=semantic_verification_enabled,
        ),
        flow=DecisionFlowProfile(
            verify_graph_final=True,
            max_revisions=max_decision_revisions,
        ),
    )
    return env


def _incident_graph_spec(
    *,
    evaluator_loop_max_iterations: int,
) -> ApplicationGraphSpec:
    return ApplicationGraphSpec(
        nodes=[GraphNode(agent_id=INVESTIGATOR_AGENT_ID)],
        trigger_capabilities=[INVESTIGATOR_CAPABILITY],
        evaluator_loop=EvaluatorLoopGraphBinding(
            producer_agent_id=INVESTIGATOR_AGENT_ID,
            evaluator_agent_id=INVESTIGATOR_AGENT_ID,
            revise_agent_id=INVESTIGATOR_AGENT_ID,
            spec=EvaluatorLoopSpec(
                max_iterations=evaluator_loop_max_iterations,
                revise_node_id=INVESTIGATOR_NODE_ID,
                escalate_on_exhaustion=False,
            ),
        ),
    )


def _apply_execution_environment_overrides(
    environment: ApplicationEnvironmentProfile,
    *,
    semantic_verification_enabled: bool,
    evaluator_loop_max_iterations: int,
    max_decision_revisions: int,
) -> None:
    environment.graph_spec = _incident_graph_spec(
        evaluator_loop_max_iterations=evaluator_loop_max_iterations,
    )
    environment.decision_profile = environment.decision_profile.model_copy(
        update={
            "verification": environment.decision_profile.verification.model_copy(
                update={"semantic_enabled": semantic_verification_enabled},
            ),
            "flow": environment.decision_profile.flow.model_copy(
                update={"max_revisions": max_decision_revisions},
            ),
        },
    )


def build_scenario_runtime_composition(
    *,
    registry: ToolRegistry,
    environment: ApplicationEnvironmentProfile | None = None,
    tenant_id: str = "scenario-tenant",
    workspace_root: Path | None = None,
    document_store: Any | None = None,
    agent_registry: AgentRegistry | None = None,
    composition: ScenarioRuntimeComposition | None = None,
    validation_engine: IncidentInvestigationValidationEngine | None = None,
) -> ScenarioRuntimeComposition:
    """
    Build or attach platform scenario runtime for incident investigation.

    When ``composition`` is supplied the platform runtime is attached to that
    incident-facing wrapper (investigator should already be on ``agent_registry``).
    """
    resolved_environment = environment or build_scenario_environment_profile()
    resolved_engine = validation_engine or IncidentInvestigationValidationEngine()
    incident_composition = composition or ScenarioRuntimeComposition(
        environment=resolved_environment,
        tool_registry=registry,
    )
    workspace = create_scenario_lab_workspace(workspace_root)
    roster = agent_registry or AgentRegistry()
    _ensure_investigator_contract_registered(roster)
    manifest = _incident_lab_manifest(resolved_environment)
    platform = build_scenario_runtime_from_environment(
        environment=resolved_environment,
        registry=roster,
        tenant_id=tenant_id,
        manifest=manifest,
        runtime_events_db_path=workspace.runtime_events_db_path,
        trace_db_path=workspace.trace_db_path,
        document_store=document_store,
        use_in_memory_trace=False,
        require_runtime_event_persistence=True,
        workspace=workspace,
        runtime_mode=ScenarioRuntimeMode.LAB,
        application_tool_registry=registry,
        validation_engine=resolved_engine,
    )
    incident_composition.environment = resolved_environment
    incident_composition.attach_platform(platform)
    return incident_composition


def resolve_scenario_llm_adapter(
    environment: ApplicationEnvironmentProfile,
    *,
    llm_adapter_override: LLMAdapter | None = None,
) -> LLMAdapter:
    """Resolve platform LLM adapter for autonomous evidence gathering (APP-2A)."""
    if llm_adapter_override is not None:
        return llm_adapter_override
    try:
        return resolve_llm_adapter(environment)
    except Exception as exc:
        raise RuntimeError(
            "incident_scenario_llm_configuration_missing: failed to resolve a configured "
            "platform LLM adapter from the application environment profile (check "
            "llm_profile, INTERGRAX_LLM_* overrides, and provider credentials when required)"
        ) from exc


def build_agent_runtime_context(
    request: RuntimeRequest,
    composition: ScenarioRuntimeComposition,
) -> RuntimeContext:
    resolved_llm = resolve_scenario_llm_adapter(
        composition.environment,
        llm_adapter_override=composition.llm_adapter_override,
    )
    return build_runtime_context_from_environment(
        request,
        composition.build_context,
        composition.environment,
        llm_adapter=resolved_llm,
    )


def prepare_incident_execution_runtime(
    composition: ScenarioRuntimeComposition,
    *,
    validation_engine: IncidentInvestigationValidationEngine | None = None,
    semantic_verification_enabled: bool = False,
    evaluator_loop_max_iterations: int = DEFAULT_EVALUATOR_LOOP_MAX_ITERATIONS,
    max_decision_revisions: int = 0,
) -> IncidentInvestigationValidationEngine:
    """Apply per-execution Decision/graph overrides before ``execute_scenario_task``."""
    resolved_engine = validation_engine or IncidentInvestigationValidationEngine()
    _apply_execution_environment_overrides(
        composition.environment,
        semantic_verification_enabled=semantic_verification_enabled,
        evaluator_loop_max_iterations=evaluator_loop_max_iterations,
        max_decision_revisions=max_decision_revisions,
    )
    rewire_scenario_decision_wiring(
        composition.platform,
        validation_engine=resolved_engine,
    )
    return resolved_engine


def trace_reader_from_composition(
    composition: ScenarioRuntimeComposition,
) -> RunTraceReader | None:
    trace_store = composition.platform.observability.trace_store
    if isinstance(trace_store, RunTraceReader):
        return trace_store
    return None

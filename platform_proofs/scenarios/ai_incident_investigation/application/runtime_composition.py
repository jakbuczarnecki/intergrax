# © Artur Czarnecki. All rights reserved.

"""Incident-specific runtime composition — delegates generic execution to scenario runtime baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from intergrax.applications._shared.critic_tool_wiring import build_critic_eval_tool_client
from intergrax.applications._shared.critic_wiring import (
    apply_application_critic_wiring,
    wire_application_critic,
)
from intergrax.applications._shared.orchestration_wiring import GraphSpecSeedingPlanner
from intergrax.applications._shared.prompt_wiring import resolve_prompt_registry
from intergrax.applications._shared.runtime_config_bridge import (
    build_runtime_context_from_environment,
)
from intergrax.applications._shared.scenario_runtime_baseline import (
    ScenarioRuntimeComposition as PlatformScenarioRuntimeComposition,
    build_scenario_runtime_from_environment,
)
from intergrax.applications._shared.scenario_runtime_profiles import (
    ScenarioRuntimeMode,
    create_scenario_lab_workspace,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ContextProfile,
    CriticProfile,
    CriticVerificationScopes,
    MemoryProfile,
)
from intergrax.applications.contracts.graph_spec import (
    ApplicationGraphSpec,
    EvaluatorLoopGraphBinding,
    GraphNode,
)
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.applications._shared.llm_resolver import resolve_llm_adapter
from intergrax.integrations.registry.catalog_manifests import LOG
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.runtime.critic.evaluator_loop_spec import EvaluatorLoopSpec
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceReader
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.registry.profile import ToolProfile
from platform_proofs.scenarios.ai_incident_investigation.application.tools import SCENARIO_TOOL_IDS
from platform_proofs.scenarios.ai_incident_investigation.application.validation import (
    IncidentInvestigationValidationEngine,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.incidents import IncidentFixture

INVESTIGATOR_AGENT_ID = "incident_investigator"
INVESTIGATOR_CAPABILITY = "incident_investigation.investigate"
INVESTIGATOR_NODE_ID = f"node_{INVESTIGATOR_AGENT_ID}"
DEFAULT_EVALUATOR_LOOP_MAX_ITERATIONS = 2


def _incident_lab_manifest(environment: ApplicationEnvironmentProfile) -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id="scenario_ai_incident_investigation",
        name="AI Incident Investigation Scenario",
        route_prefix="/v1/scenario/ai_incident_investigation",
        env_prefix="SCENARIO_AI_INCIDENT_",
        agents=[],
        environment=environment,
    )


@dataclass
class ScenarioRuntimeComposition:
    """Incident-facing runtime composition view over the platform scenario runtime."""

    environment: ApplicationEnvironmentProfile
    tool_registry: ToolRegistry
    _platform: PlatformScenarioRuntimeComposition | None = None

    @property
    def platform(self) -> PlatformScenarioRuntimeComposition:
        if self._platform is None:
            raise RuntimeError("platform scenario runtime not attached")
        return self._platform

    @property
    def build_context(self) -> object:
        return self.platform.env_wiring.build_context

    def attach_platform(self, platform: PlatformScenarioRuntimeComposition) -> None:
        self._platform = platform


def build_scenario_environment_profile(
    *,
    evaluator_loop_max_iterations: int = DEFAULT_EVALUATOR_LOOP_MAX_ITERATIONS,
    require_critic_on_completion: bool = True,
    semantic_judge_enabled: bool = False,
) -> ApplicationEnvironmentProfile:
    """Provider-neutral LAB environment with fixed investigator topology."""
    env = ApplicationEnvironmentProfile.lab_defaults(
        profile_id="scenario.ai_incident_investigation",
        harness_tools=False,
    )
    env.integration_profile = IntegrationProfile(notification_channel=LOG)
    env.context_profile = ContextProfile(enable_rag=False, enable_websearch=False)
    env.memory_profile = MemoryProfile()
    env.tool_profile = ToolProfile(enabled=list(SCENARIO_TOOL_IDS))
    env.graph_spec = _incident_graph_spec(
        evaluator_loop_max_iterations=evaluator_loop_max_iterations,
    )
    env.critic_profile = CriticProfile(
        scopes=CriticVerificationScopes(node_partial=True, graph_final=True),
        semantic_judge_enabled=semantic_judge_enabled,
        require_critic_on_completion=require_critic_on_completion,
        evaluator_loop_max_iterations=evaluator_loop_max_iterations,
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


def _wire_incident_validation(
    platform: PlatformScenarioRuntimeComposition,
    *,
    validation_engine: IncidentInvestigationValidationEngine | None = None,
    require_critic_on_completion: bool = True,
    semantic_judge_enabled: bool = False,
) -> IncidentInvestigationValidationEngine:
    resolved_engine = validation_engine or IncidentInvestigationValidationEngine()
    platform.nexus_loop._validation_engine = resolved_engine
    platform.nexus_loop._graph_executor._validation_engine = resolved_engine
    platform.nexus_loop._graph_runner.validation_engine = resolved_engine

    l1_client = build_critic_eval_tool_client(
        platform.environment,
        platform.env_wiring.tool_wiring,
        evaluation_registry=None,
        trace_reader=platform.observability.trace_store,
    )
    critic_wiring = wire_application_critic(
        platform.environment,
        l1_client=l1_client,
        validation_engine=resolved_engine,
    )
    critic_profile = platform.environment.critic_profile
    if critic_profile.require_critic_on_completion != require_critic_on_completion:
        critic_profile = critic_profile.model_copy(
            update={"require_critic_on_completion": require_critic_on_completion}
        )
    if critic_profile.semantic_judge_enabled != semantic_judge_enabled:
        critic_profile = critic_profile.model_copy(
            update={"semantic_judge_enabled": semantic_judge_enabled}
        )
    if critic_profile != platform.environment.critic_profile:
        platform.environment.critic_profile = critic_profile
        critic_wiring = wire_application_critic(
            platform.environment,
            l1_client=l1_client,
            validation_engine=resolved_engine,
        )
    apply_application_critic_wiring(platform.nexus_loop, critic_wiring)
    return resolved_engine


def _patch_evaluator_loop_iterations(
    platform: PlatformScenarioRuntimeComposition,
    evaluator_loop_max_iterations: int,
) -> None:
    planner = platform.nexus_loop._planner
    if not isinstance(planner, GraphSpecSeedingPlanner):
        return
    graph_spec = planner._graph_spec
    if graph_spec.evaluator_loop is None:
        return
    if graph_spec.evaluator_loop.spec.max_iterations == evaluator_loop_max_iterations:
        return
    updated_binding = graph_spec.evaluator_loop.model_copy(
        update={
            "spec": graph_spec.evaluator_loop.spec.model_copy(
                update={"max_iterations": evaluator_loop_max_iterations}
            )
        }
    )
    planner._graph_spec = graph_spec.model_copy(update={"evaluator_loop": updated_binding})


def build_scenario_runtime_composition(
    *,
    registry: ToolRegistry,
    environment: ApplicationEnvironmentProfile | None = None,
    tenant_id: str = "scenario-tenant",
    workspace_root: Path | None = None,
    document_store: Any | None = None,
    agent_registry: AgentRegistry | None = None,
    composition: ScenarioRuntimeComposition | None = None,
) -> ScenarioRuntimeComposition:
    """
    Build or attach platform scenario runtime for incident investigation.

    When ``composition`` is supplied the platform runtime is attached to that
    incident-facing wrapper (investigator should already be on ``agent_registry``).
    """
    resolved_environment = environment or build_scenario_environment_profile()
    incident_composition = composition or ScenarioRuntimeComposition(
        environment=resolved_environment,
        tool_registry=registry,
    )
    workspace = create_scenario_lab_workspace(workspace_root)
    roster = agent_registry or AgentRegistry()
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
        diagnostics_required=False,
        workspace=workspace,
        runtime_mode=ScenarioRuntimeMode.LAB,
        conformance_check=False,
    )
    incident_composition.attach_platform(platform)
    _wire_incident_validation(platform)
    return incident_composition


class _IncidentToolCallIdAdapter:
    """Assign stable tool_call_id values when local providers omit them."""

    def __init__(self, inner: LLMAdapter) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    def generate_with_tools(self, *args: object, **kwargs: object) -> LLMAdapterResponse:
        result = self._inner.generate_with_tools(*args, **kwargs)
        if not result.tool_calls:
            return result
        fixed_calls: list[LLMToolCall] = []
        for index, call in enumerate(result.tool_calls, start=1):
            call_id = call.id.strip() if call.id else f"incident_tool_call_{index}"
            fixed_calls.append(
                LLMToolCall(
                    id=call_id,
                    name=call.name,
                    arguments_json=call.arguments_json,
                )
            )
        return LLMAdapterResponse(content=result.content, tool_calls=tuple(fixed_calls))


def resolve_scenario_llm_adapter(
    environment: ApplicationEnvironmentProfile,
) -> LLMAdapter:
    """Resolve platform LLM adapter for autonomous evidence gathering (APP-2A)."""
    from platform_proofs.scenarios.ai_incident_investigation.fixtures.lab_planner_llm import (
        FixtureDrivenIncidentInvestigationLLM,
        lab_planner_enabled,
    )

    if lab_planner_enabled():
        return _IncidentToolCallIdAdapter(FixtureDrivenIncidentInvestigationLLM())
    try:
        return _IncidentToolCallIdAdapter(resolve_llm_adapter(environment))
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
    resolved_llm = resolve_scenario_llm_adapter(composition.environment)
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
    require_critic_on_completion: bool = True,
    semantic_judge_enabled: bool = False,
    evaluator_loop_max_iterations: int = DEFAULT_EVALUATOR_LOOP_MAX_ITERATIONS,
) -> IncidentInvestigationValidationEngine:
    """Apply per-execution critic/loop overrides before ``execute_scenario_task``."""
    platform = composition.platform
    _patch_evaluator_loop_iterations(platform, evaluator_loop_max_iterations)
    return _wire_incident_validation(
        platform,
        validation_engine=validation_engine,
        require_critic_on_completion=require_critic_on_completion,
        semantic_judge_enabled=semantic_judge_enabled,
    )


def trace_reader_from_composition(
    composition: ScenarioRuntimeComposition,
) -> RunTraceReader | None:
    trace_store = composition.platform.observability.trace_store
    if isinstance(trace_store, RunTraceReader):
        return trace_store
    return None

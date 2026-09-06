"""Scenario runtime composition via platform scenario runtime baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.applications._shared.application_owned_tool_conformance import (
    application_owned_tool_declarations,
)
from intergrax.applications._shared.llm_resolver import resolve_llm_adapter
from intergrax.applications._shared.runtime_config_bridge import (
    build_runtime_context_from_environment,
)
from intergrax.applications._shared.scenario_runtime_baseline import (
    ScenarioRuntimeComposition as PlatformScenarioRuntimeComposition,
    build_scenario_lab_agent_registry,
    build_scenario_runtime_from_environment,
)
from intergrax.applications._shared.scenario_runtime_profiles import (
    ScenarioRuntimeMode,
    create_scenario_lab_workspace,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.graph_spec import ApplicationGraphSpec, GraphNode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceReader
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.tools.registry import ToolRegistry

from platform_proofs.scenarios.indirect_prompt_injection.application.order_provider_client import (
    OrderProviderClient,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.tools import (
    SCENARIO_TOOL_IDS,
    register_scenario_tools,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.workflows import (
    WorkflowKind,
    build_scenario_environment_profile,
)

ORDER_ASSISTANT_AGENT_ID = "order_assistant"
ORDER_ASSISTANT_CAPABILITY = "indirect_prompt_injection.assist"
SYNTHETIC_SCENARIO_TENANT_ID = "synthetic-scenario-indirect_prompt_injection"


def _scenario_lab_manifest(environment: ApplicationEnvironmentProfile) -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id="scenario_indirect_prompt_injection",
        name="AI Order Assistant",
        route_prefix="/v1/scenario/indirect_prompt_injection",
        env_prefix="SCENARIO_INDIRECT_PROMPT_INJECTION_",
        agents=[
            AgentBinding.reference(
                contract_id=ORDER_ASSISTANT_AGENT_ID,
                capabilities=[ORDER_ASSISTANT_CAPABILITY],
            ),
        ],
        application_owned_tools=application_owned_tool_declarations(SCENARIO_TOOL_IDS),
        environment=environment,
    )


@dataclass
class ScenarioRuntimeComposition:
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


def build_scenario_runtime_composition(
    *,
    registry: ToolRegistry,
    environment: ApplicationEnvironmentProfile,
    tenant_id: str = SYNTHETIC_SCENARIO_TENANT_ID,
    workspace_root: Path | None = None,
    agent_registry: AgentRegistry | None = None,
    composition: ScenarioRuntimeComposition | None = None,
    provider_client: OrderProviderClient,
) -> ScenarioRuntimeComposition:
    register_scenario_tools(registry, provider_client=provider_client)
    scenario_composition = composition or ScenarioRuntimeComposition(
        environment=environment,
        tool_registry=registry,
    )
    workspace = create_scenario_lab_workspace(workspace_root)
    roster = agent_registry or build_scenario_lab_agent_registry()
    environment.graph_spec = ApplicationGraphSpec(
        nodes=[GraphNode(agent_id=ORDER_ASSISTANT_AGENT_ID)],
        trigger_capabilities=[ORDER_ASSISTANT_CAPABILITY],
    )
    manifest = _scenario_lab_manifest(environment)
    platform = build_scenario_runtime_from_environment(
        environment=environment,
        registry=roster,
        tenant_id=tenant_id,
        manifest=manifest,
        runtime_events_db_path=workspace.runtime_events_db_path,
        trace_db_path=workspace.trace_db_path,
        use_in_memory_trace=False,
        require_runtime_event_persistence=True,
        workspace=workspace,
        runtime_mode=ScenarioRuntimeMode.LAB,
        application_tool_registry=registry,
    )
    scenario_composition.environment = environment
    scenario_composition.attach_platform(platform)
    return scenario_composition


def resolve_scenario_llm_adapter(
    environment: ApplicationEnvironmentProfile,
    *,
    llm_adapter_override: LLMAdapter | None = None,
) -> LLMAdapter:
    if llm_adapter_override is not None:
        return llm_adapter_override
    return resolve_llm_adapter(environment)


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


def trace_reader_from_composition(
    composition: ScenarioRuntimeComposition,
) -> RunTraceReader | None:
    trace_store = composition.platform.observability.trace_store
    if isinstance(trace_store, RunTraceReader):
        return trace_store
    return None

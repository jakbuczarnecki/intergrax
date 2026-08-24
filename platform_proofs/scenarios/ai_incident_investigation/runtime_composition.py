# © Artur Czarnecki. All rights reserved.

"""Production runtime composition for the incident investigation scenario (APP-1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.runtime_config_bridge import (
    build_runtime_context_from_environment,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ContextProfile,
    MemoryProfile,
)
from intergrax.integrations.registry.catalog_manifests import LOG
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.applications._shared.llm_resolver import resolve_llm_adapter
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.registry.profile import ToolProfile
from platform_proofs.scenarios.ai_incident_investigation.tools import SCENARIO_TOOL_IDS


@dataclass(frozen=True, slots=True)
class ScenarioRuntimeComposition:
    """Application-owned runtime dependencies injected into the investigator agent."""

    environment: ApplicationEnvironmentProfile
    build_context: ApplicationBuildContext


def build_scenario_environment_profile() -> ApplicationEnvironmentProfile:
    """Minimal provider-neutral environment for synthetic incident investigation."""
    env = ApplicationEnvironmentProfile.lab_defaults(
        profile_id="scenario.ai_incident_investigation",
        harness_tools=False,
    )
    env.integration_profile = IntegrationProfile(notification_channel=LOG)
    env.context_profile = ContextProfile(enable_rag=False, enable_websearch=False)
    env.memory_profile = MemoryProfile()
    return env


def build_scenario_runtime_composition(
    *,
    registry: ToolRegistry,
    environment: ApplicationEnvironmentProfile | None = None,
) -> ScenarioRuntimeComposition:
    resolved_environment = environment or build_scenario_environment_profile()
    tool_profile = ToolProfile(enabled=list(SCENARIO_TOOL_IDS))
    build_ctx = ApplicationBuildContext.for_manifest(
        object(),
        tool_profile=tool_profile,
        tool_registry=registry,
        policy_bundle=RuntimePolicyBundle(),
        environment=resolved_environment,
    )
    return ScenarioRuntimeComposition(
        environment=resolved_environment,
        build_context=build_ctx,
    )


def resolve_scenario_llm_adapter(
    environment: ApplicationEnvironmentProfile,
) -> LLMAdapter:
    """
    Resolve the platform LLM adapter for the scenario environment.

    Model calls materialize autonomous evidence gathering via bounded tool loop (APP-2A).
    """
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
    resolved_llm = resolve_scenario_llm_adapter(composition.environment)
    return build_runtime_context_from_environment(
        request,
        composition.build_context,
        composition.environment,
        llm_adapter=resolved_llm,
    )

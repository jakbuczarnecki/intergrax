# © Artur Czarnecki. All rights reserved.

"""Canonical AC-5 factory for the private incident investigator scenario agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from intergrax.agents.agent_contract import Agent
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.factory import CanonicalAgentFactory
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.tools.registry import ToolRegistry
from platform_proofs.scenarios.ai_incident_investigation.application.incident_data_contracts import (
    IncidentOperationalData,
)
from platform_proofs.scenarios.ai_incident_investigation.application.incident_scope import (
    IncidentScope,
)
from platform_proofs.scenarios.ai_incident_investigation.application.investigator_agent import (
    IncidentInvestigatorAgent,
)
from platform_proofs.scenarios.ai_incident_investigation.application.runtime_composition import (
    ScenarioRuntimeComposition,
    build_scenario_environment_profile,
)
from platform_proofs.scenarios.ai_incident_investigation.application.tools import (
    ScenarioEvidenceStore,
    register_scenario_tools,
)

if TYPE_CHECKING:
    from intergrax.runtime.diagnostics.investigation_contracts import (
        IncidentInvestigationInput,
    )


@dataclass(slots=True)
class IncidentInvestigatorProductionSettings:
    """Declarative application-owned settings for ``ApplicationBuildContext.settings``."""

    operational_data: IncidentOperationalData
    investigation_input: IncidentInvestigationInput | None = None


@dataclass(slots=True)
class IncidentInvestigatorRuntimeBootstrap:
    """Host-owned runtime products prepared before factory invocation."""

    settings: IncidentInvestigatorProductionSettings
    factory: CanonicalAgentFactory
    composition: ScenarioRuntimeComposition
    evidence_store: ScenarioEvidenceStore


_bound_factory: CanonicalAgentFactory | None = None


def bind_incident_investigator_factory(factory: CanonicalAgentFactory) -> None:
    """Host binds the configured factory for VENV_BUNDLE path resolution."""
    global _bound_factory
    _bound_factory = factory


def resolve_incident_investigator_production_settings(
    ctx: ApplicationBuildContext,
) -> IncidentInvestigatorProductionSettings:
    settings = ctx.settings
    if not isinstance(settings, IncidentInvestigatorProductionSettings):
        raise TypeError(
            "incident_investigator_factory_requires_incident_investigator_production_settings"
        )
    return settings


def build_incident_investigator_factory(
    *,
    operational_data: IncidentOperationalData,
    runtime_composition: ScenarioRuntimeComposition,
    evidence_store: ScenarioEvidenceStore,
    investigation_input: IncidentInvestigationInput | None = None,
) -> CanonicalAgentFactory:
    """Return a configured factory closed over prepared runtime dependencies."""

    def _factory(ctx: ApplicationBuildContext, binding: AgentBinding) -> Agent:
        del binding
        settings = ctx.settings
        if settings is not None and not isinstance(
            settings, IncidentInvestigatorProductionSettings
        ):
            raise TypeError(
                "incident_investigator_factory_requires_incident_investigator_production_settings"
            )
        resolved_input = investigation_input
        if isinstance(settings, IncidentInvestigatorProductionSettings):
            if settings.operational_data.station_id != operational_data.station_id:
                raise ValueError(
                    "incident_investigator_settings_operational_data_mismatch"
                )
            if settings.investigation_input is not None:
                resolved_input = settings.investigation_input
        return IncidentInvestigatorAgent(
            station_id=operational_data.station_id,
            runtime_composition=runtime_composition,
            incident_scope=IncidentScope.from_operational_defaults(
                station_id=operational_data.station_id,
            ),
            evidence_store=evidence_store,
            investigation_input=resolved_input,
        )

    return _factory


def build_agent(ctx: ApplicationBuildContext, binding: AgentBinding) -> Agent:
    """VENV_BUNDLE trampoline — requires prior host ``bind_incident_investigator_factory``."""
    if _bound_factory is None:
        raise TypeError(
            "incident_investigator_factory_not_bound_by_host_bootstrap"
        )
    return _bound_factory(ctx, binding)


def bootstrap_incident_investigator_runtime(
    operational_data: IncidentOperationalData,
    *,
    investigation_input: IncidentInvestigationInput | None = None,
    llm_adapter_override: LLMAdapter | None = None,
) -> IncidentInvestigatorRuntimeBootstrap:
    """Build tool registry, evidence store, composition, and configured factory."""
    environment = build_scenario_environment_profile()
    tool_registry = ToolRegistry()
    evidence_store = register_scenario_tools(tool_registry, operational_data)
    composition = ScenarioRuntimeComposition(
        environment=environment,
        tool_registry=tool_registry,
        llm_adapter_override=llm_adapter_override,
    )
    settings = IncidentInvestigatorProductionSettings(
        operational_data=operational_data,
        investigation_input=investigation_input,
    )
    factory = build_incident_investigator_factory(
        operational_data=operational_data,
        runtime_composition=composition,
        evidence_store=evidence_store,
        investigation_input=investigation_input,
    )
    return IncidentInvestigatorRuntimeBootstrap(
        settings=settings,
        factory=factory,
        composition=composition,
        evidence_store=evidence_store,
    )


def build_default_production_settings(
    operational_data: IncidentOperationalData,
    *,
    investigation_input: IncidentInvestigationInput | None = None,
    llm_adapter_override: LLMAdapter | None = None,
) -> IncidentInvestigatorProductionSettings:
    """Backward-compatible bootstrap that also binds the configured factory."""
    bootstrap = bootstrap_incident_investigator_runtime(
        operational_data,
        investigation_input=investigation_input,
        llm_adapter_override=llm_adapter_override,
    )
    bind_incident_investigator_factory(bootstrap.factory)
    return bootstrap.settings

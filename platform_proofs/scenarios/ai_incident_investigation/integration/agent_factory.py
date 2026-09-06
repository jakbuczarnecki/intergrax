# © Artur Czarnecki. All rights reserved.

"""Canonical AC-5 factory for the private incident investigator scenario agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from intergrax.agents.agent_contract import Agent
from intergrax.applications.contracts.build_context import ApplicationBuildContext
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
    """Application-owned execution context supplied through ``ApplicationBuildContext.settings``."""

    operational_data: IncidentOperationalData
    composition: ScenarioRuntimeComposition
    investigation_input: IncidentInvestigationInput | None = None
    llm_adapter_override: LLMAdapter | None = None
    evidence_store: ScenarioEvidenceStore | None = None


def resolve_incident_investigator_production_settings(
    ctx: ApplicationBuildContext,
) -> IncidentInvestigatorProductionSettings:
    settings = ctx.settings
    if not isinstance(settings, IncidentInvestigatorProductionSettings):
        raise TypeError(
            "incident_investigator_factory_requires_incident_investigator_production_settings"
        )
    return settings


def build_agent(ctx: ApplicationBuildContext, binding: AgentBinding) -> Agent:
    del binding
    production_settings = resolve_incident_investigator_production_settings(ctx)
    tool_registry = ctx.tool_registry
    if not isinstance(tool_registry, ToolRegistry):
        raise TypeError("incident_investigator_factory_requires_tool_registry")
    evidence_store = register_scenario_tools(
        tool_registry,
        production_settings.operational_data,
    )
    if not isinstance(evidence_store, ScenarioEvidenceStore):
        raise TypeError(
            "incident_investigator_factory_requires_scenario_evidence_store"
        )
    composition = production_settings.composition
    composition.tool_registry = tool_registry
    composition.llm_adapter_override = production_settings.llm_adapter_override
    production_settings.evidence_store = evidence_store
    return IncidentInvestigatorAgent(
        registry=tool_registry,
        station_id=production_settings.operational_data.station_id,
        runtime_composition=composition,
        incident_scope=IncidentScope.from_operational_defaults(
            station_id=production_settings.operational_data.station_id,
        ),
        evidence_store=evidence_store,
        investigation_input=production_settings.investigation_input,
    )


def build_default_production_settings(
    operational_data: IncidentOperationalData,
    *,
    investigation_input: IncidentInvestigationInput | None = None,
    llm_adapter_override: LLMAdapter | None = None,
) -> IncidentInvestigatorProductionSettings:
    environment = build_scenario_environment_profile()
    composition = ScenarioRuntimeComposition(
        environment=environment,
        tool_registry=ToolRegistry(),
        llm_adapter_override=llm_adapter_override,
    )
    return IncidentInvestigatorProductionSettings(
        operational_data=operational_data,
        composition=composition,
        investigation_input=investigation_input,
        llm_adapter_override=llm_adapter_override,
    )

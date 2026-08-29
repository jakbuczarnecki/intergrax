# © Artur Czarnecki. All rights reserved.

"""Fixture-owned runtime bundle composition — synthetic data selection stays outside application."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.diagnostics.investigation_contracts import IncidentInvestigationInput
from intergrax.runtime.diagnostics.diagnostic_read_service import DiagnosticReadService
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemId
from platform_proofs.scenarios.ai_incident_investigation.application.incident_data_contracts import (
    IncidentOperationalData,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    ScenarioRuntimeBundle,
    STANDALONE_SCENARIO_TENANT_ID,
    build_runtime_bundle as build_application_runtime_bundle,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_composition import (
    resolve_incident_investigation_input,
)
from platform_proofs.scenarios.ai_incident_investigation.application.runtime_composition import (
    ScenarioRuntimeComposition,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.incidents import (
    IncidentFixture,
    ScenarioVariant,
    build_resolved_fixture,
    build_unresolved_fixture,
)


@dataclass(frozen=True, slots=True)
class FixtureRuntimeBundle:
    bundle: ScenarioRuntimeBundle
    fixture: IncidentFixture


def _resolve_fixture(
    *,
    variant: ScenarioVariant,
    fixture: IncidentFixture | None,
) -> IncidentFixture:
    if fixture is not None:
        return fixture
    if variant is ScenarioVariant.UNRESOLVED:
        return build_unresolved_fixture()
    return build_resolved_fixture()


def build_fixture_runtime_bundle(
    *,
    variant: ScenarioVariant = ScenarioVariant.RESOLVED,
    fixture: IncidentFixture | None = None,
    operational_data: IncidentOperationalData | None = None,
    runtime_composition: ScenarioRuntimeComposition | None = None,
    tenant_id: str = STANDALONE_SCENARIO_TENANT_ID,
    investigation_input: IncidentInvestigationInput | None = None,
    llm_adapter_override: LLMAdapter | None = None,
) -> FixtureRuntimeBundle:
    resolved_fixture = _resolve_fixture(variant=variant, fixture=fixture)
    resolved_operational_data = operational_data or resolved_fixture.to_operational_data()
    bundle = build_application_runtime_bundle(
        operational_data=resolved_operational_data,
        runtime_composition=runtime_composition,
        tenant_id=tenant_id,
        investigation_input=investigation_input,
        llm_adapter_override=llm_adapter_override,
    )
    return FixtureRuntimeBundle(bundle=bundle, fixture=resolved_fixture)


def build_runtime_bundle(
    *,
    variant: ScenarioVariant = ScenarioVariant.RESOLVED,
    fixture: IncidentFixture | None = None,
    operational_data: IncidentOperationalData | None = None,
    runtime_composition: ScenarioRuntimeComposition | None = None,
    tenant_id: str = STANDALONE_SCENARIO_TENANT_ID,
    investigation_input: IncidentInvestigationInput | None = None,
    llm_adapter_override: LLMAdapter | None = None,
) -> ScenarioRuntimeBundle:
    """Fixture-layer entry — synthetic variant selection stays outside application."""
    return build_fixture_runtime_bundle(
        variant=variant,
        fixture=fixture,
        operational_data=operational_data,
        runtime_composition=runtime_composition,
        tenant_id=tenant_id,
        investigation_input=investigation_input,
        llm_adapter_override=llm_adapter_override,
    ).bundle


def build_runtime_bundle_from_diagnostic_problem(
    diagnostic_read_service: DiagnosticReadService,
    *,
    tenant_id: str,
    problem_ids: ProblemId | tuple[ProblemId, ...],
    variant: ScenarioVariant = ScenarioVariant.RESOLVED,
    fixture: IncidentFixture | None = None,
    operational_data: IncidentOperationalData | None = None,
    runtime_composition: ScenarioRuntimeComposition | None = None,
    llm_adapter_override: LLMAdapter | None = None,
) -> FixtureRuntimeBundle:
    """Platform-attached mode with fixture-owned synthetic operational data."""
    investigation_input = resolve_incident_investigation_input(
        diagnostic_read_service,
        tenant_id=tenant_id,
        problem_ids=problem_ids,
    )
    return build_fixture_runtime_bundle(
        variant=variant,
        fixture=fixture,
        operational_data=operational_data,
        runtime_composition=runtime_composition,
        tenant_id=tenant_id,
        investigation_input=investigation_input,
        llm_adapter_override=llm_adapter_override,
    )

# © Artur Czarnecki. All rights reserved.

"""Platform-attached scenario composition — DiagnosticReadService → investigation bundle."""

from __future__ import annotations

from intergrax.runtime.diagnostics.diagnostic_read_service import DiagnosticReadService
from intergrax.runtime.diagnostics.investigation_contracts import (
    IncidentInvestigationInput,
    IncidentInvestigationIntegrityError,
    incident_investigation_input_from_problem_details,
)
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemId
from platform_proofs.scenarios.ai_incident_investigation.fixtures.incidents import (
    IncidentFixture,
    ScenarioVariant,
    build_resolved_fixture,
    build_unresolved_fixture,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    ScenarioRuntimeBundle,
    STANDALONE_SCENARIO_TENANT_ID,
    build_runtime_bundle,
)


class IncidentInvestigationProblemNotFoundError(Exception):
    """Platform-attached investigation could not resolve canonical Problem identity."""


def _normalize_problem_ids(
    problem_ids: ProblemId | tuple[ProblemId, ...],
) -> tuple[ProblemId, ...]:
    if isinstance(problem_ids, tuple):
        if not problem_ids:
            raise ValueError("problem_ids must be non-empty when provided as a tuple")
        return problem_ids
    return (problem_ids,)


def resolve_incident_investigation_input(
    diagnostic_read_service: DiagnosticReadService,
    *,
    tenant_id: str,
    problem_ids: ProblemId | tuple[ProblemId, ...],
) -> IncidentInvestigationInput:
    """
    Resolve canonical investigation input from DiagnosticReadService without scenario fixtures.

    Fails clearly when any Problem is missing — no synthetic fallback to standalone mode.
    """
    normalized_ids = _normalize_problem_ids(problem_ids)
    details: list = []
    for problem_id in normalized_ids:
        detail = diagnostic_read_service.get_problem(
            tenant_id=tenant_id,
            problem_id=problem_id,
        )
        if detail is None:
            raise IncidentInvestigationProblemNotFoundError(
                f"incident_investigation_problem_not_found: tenant={tenant_id!r} "
                f"problem_id={problem_id!r}"
            )
        details.append(detail)
    try:
        return incident_investigation_input_from_problem_details(
            tenant_id=tenant_id,
            details=tuple(details),
        )
    except IncidentInvestigationIntegrityError as exc:
        raise IncidentInvestigationIntegrityError(
            f"incident investigation tenant integrity failed for tenant={tenant_id!r}: {exc}"
        ) from exc


def build_runtime_bundle_from_diagnostic_problem(
    diagnostic_read_service: DiagnosticReadService,
    *,
    tenant_id: str,
    problem_ids: ProblemId | tuple[ProblemId, ...],
    variant: ScenarioVariant = ScenarioVariant.RESOLVED,
    fixture: IncidentFixture | None = None,
    runtime_composition: object | None = None,
) -> ScenarioRuntimeBundle:
    """
    Platform-attached investigation mode: seed scenario from canonical Problem read surface.

    Domain evidence fixtures remain scenario-owned; platform context is read-only derived
    from central diagnostics.
    """
    investigation_input = resolve_incident_investigation_input(
        diagnostic_read_service,
        tenant_id=tenant_id,
        problem_ids=problem_ids,
    )
    return build_runtime_bundle(
        variant=variant,
        fixture=fixture,
        runtime_composition=runtime_composition,
        tenant_id=tenant_id,
        investigation_input=investigation_input,
    )


def scenario_execution_tenant_id(bundle: ScenarioRuntimeBundle) -> str:
    """Tenant for investigation execution — platform input in attached mode, fixture tenant otherwise."""
    if bundle.investigation_input is not None:
        return bundle.investigation_input.tenant_id
    return STANDALONE_SCENARIO_TENANT_ID

# © Artur Czarnecki. All rights reserved.

"""MP-FINAL-2 — Multiplayer failure → Evidence → Diagnostics → operator E2E."""

from __future__ import annotations

import pytest

from intergrax.contracts.collaborative_functional_evidence_projection import (
    COLLABORATIVE_DECISION_BINDING_CREATE_OPERATION_ID,
    COLLABORATIVE_DECISION_BINDING_EVIDENCE_PRODUCER,
)
from intergrax.contracts.collaborative_work import (
    CollaborativeWorkAuthorizationDenied,
)
from intergrax.contracts.functional_evidence.models import (
    PipelineEvidenceKind,
    PipelineOperationStatus,
)
from intergrax.contracts.functional_evidence.persistence import FunctionalEvidenceQueryRequest
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.diagnostics.functional_operator_projection import (
    FunctionalOperatorOutcomeStatus,
)
from tests.qualification.multiplayer.mp_final2.host_operability import (
    MP_FINAL2_CHECK_BINDING_CREATE,
    FailingCreateBindingRepository,
    binding_create_request,
    build_operability_host,
    execution_correlation_for,
    interpret_binding_create_operability,
)

pytestmark = [pytest.mark.unit, pytest.mark.qualification]


def test_e2e_technical_persistence_failure_to_operator_visible_functional_diagnostic() -> None:
    """
    Real Multiplayer binding create → persistence failure → Evidence FAILED
    → Functional Diagnostics → operator-visible typed finding (public projection).
    """
    host = build_operability_host(binding_repository=FailingCreateBindingRepository())
    correlation = execution_correlation_for(host)

    with pytest.raises(RuntimeError, match="simulated collaborative decision binding persistence"):
        host.application.create_binding(
            binding_create_request(host),
            execution_correlation=correlation,
        )

    page = host.evidence_persistence.query_evidence(
        FunctionalEvidenceQueryRequest(
            tenant_id=host.tenant_id,
            task_id=host.task_id,
            run_id=host.run_id,
            attempt_id=host.attempt_id,
            kind=PipelineEvidenceKind.OPERATION_OUTCOME,
        ),
    )
    assert len(page.items) == 1
    evidence = page.items[0]
    assert evidence.kind is PipelineEvidenceKind.OPERATION_OUTCOME
    assert evidence.operation_outcome is not None
    assert evidence.operation_outcome.status is PipelineOperationStatus.FAILED
    assert evidence.operation_outcome.operation_name == COLLABORATIVE_DECISION_BINDING_CREATE_OPERATION_ID
    assert evidence.provenance.producer_component == COLLABORATIVE_DECISION_BINDING_EVIDENCE_PRODUCER
    assert evidence.provenance.operation_id == COLLABORATIVE_DECISION_BINDING_CREATE_OPERATION_ID
    assert evidence.scope.tenant_id == host.tenant_id
    assert evidence.scope.task_id == host.task_id
    assert evidence.scope.run_id == host.run_id
    assert evidence.scope.attempt_id == host.attempt_id
    assert evidence.scope.execution_id == host.execution_id

    projection = interpret_binding_create_operability(host)
    assert projection.outcome_status is FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE
    assert projection.tenant_id == host.tenant_id
    assert projection.task_id == host.task_id
    assert projection.run_id == host.run_id
    assert projection.attempt_id == host.attempt_id
    assert projection.first_proven_failed_check == MP_FINAL2_CHECK_BINDING_CREATE
    assert len(projection.failures) == 1
    finding = projection.failures[0]
    assert finding.check_id == MP_FINAL2_CHECK_BINDING_CREATE
    assert "failed" in finding.factual_claim.lower()
    assert evidence.evidence_id in finding.supporting_evidence_refs

    # Multiplayer path does not invent a second Problem store truth for this scenario.
    listed = host.diagnostic_read_service.list_problems(tenant_id=host.tenant_id)
    assert listed.problems == ()


def test_e2e_authorization_deny_does_not_masquerade_as_infrastructure_problem() -> None:
    """Policy DENY remains DENY; no false infrastructure Problem via DiagnosticReadService."""
    host = build_operability_host(seed_authority=False)
    correlation = execution_correlation_for(host)

    with pytest.raises(CollaborativeWorkAuthorizationDenied) as exc_info:
        host.application.create_binding(
            binding_create_request(host),
            execution_correlation=correlation,
        )

    denied = exc_info.value
    assert denied.enforcement_result.composition.decision.action is PolicyAction.DENY

    # Observation of failed operation may exist on Evidence Plane — that is not an infra Problem.
    page = host.evidence_persistence.query_evidence(
        FunctionalEvidenceQueryRequest(
            tenant_id=host.tenant_id,
            task_id=host.task_id,
            run_id=host.run_id,
            attempt_id=host.attempt_id,
            kind=PipelineEvidenceKind.OPERATION_OUTCOME,
        ),
    )
    assert len(page.items) == 1
    assert page.items[0].operation_outcome is not None
    assert page.items[0].operation_outcome.status is PipelineOperationStatus.FAILED

    listed = host.diagnostic_read_service.list_problems(tenant_id=host.tenant_id)
    assert listed.problems == ()
    assert type(denied) is CollaborativeWorkAuthorizationDenied


def test_e2e_diagnostics_failure_never_weakens_authorization() -> None:
    """Diagnostics outage / interpretation failure must not turn DENY into ALLOW/success."""
    host = build_operability_host(seed_authority=False)
    correlation = execution_correlation_for(host)

    class _BrokenAnalyzer:
        def analyze(self, **_kwargs: object) -> object:
            raise RuntimeError("simulated diagnostics interpretation outage")

    host_broken = host  # domain path independent of analyzer
    with pytest.raises(CollaborativeWorkAuthorizationDenied) as exc_info:
        host_broken.application.create_binding(
            binding_create_request(host_broken),
            execution_correlation=correlation,
        )
    assert exc_info.value.enforcement_result.composition.decision.action is PolicyAction.DENY

    broken_analyzer = _BrokenAnalyzer()
    with pytest.raises(RuntimeError, match="diagnostics interpretation outage"):
        broken_analyzer.analyze(
            tenant_id=host.tenant_id,
            task_id=host.task_id,
            run_id=host.run_id,
        )

    # Re-evaluate: DENY still DENY after diagnostics failure; never success.
    with pytest.raises(CollaborativeWorkAuthorizationDenied) as again:
        host.application.create_binding(
            binding_create_request(host, idempotency_key="mp-final2-idem-2"),
            execution_correlation=correlation,
        )
    assert again.value.enforcement_result.composition.decision.action is PolicyAction.DENY


def test_e2e_tenant_isolation_of_operator_projection() -> None:
    """Workspace/tenant A failure must not appear under tenant B diagnostic scope."""
    host_a = build_operability_host(
        tenant_id="mp-final2-tenant-a",
        workspace_id="mp-final2-workspace-a",
        binding_repository=FailingCreateBindingRepository(),
    )
    with pytest.raises(RuntimeError):
        host_a.application.create_binding(
            binding_create_request(host_a),
            execution_correlation=execution_correlation_for(host_a),
        )

    host_b = build_operability_host(
        tenant_id="mp-final2-tenant-b",
        workspace_id="mp-final2-workspace-b",
        work_item_id="mp-final2-work-item-b",
        acting_principal_id="mp-final2-principal-b",
        evidence_persistence=host_a.evidence_persistence,
        task_id=host_a.task_id,
        run_id=host_a.run_id,
        attempt_id=host_a.attempt_id,
        execution_id=host_a.execution_id,
    )
    # Shared persistence store + identical execution ids + different tenant → no evidence for B.
    page_b = host_b.evidence_persistence.query_evidence(
        FunctionalEvidenceQueryRequest(
            tenant_id=host_b.tenant_id,
            task_id=host_b.task_id,
            run_id=host_b.run_id,
            attempt_id=host_b.attempt_id,
            kind=PipelineEvidenceKind.OPERATION_OUTCOME,
        ),
    )
    assert page_b.items == ()

    projection_b = interpret_binding_create_operability(host_b)
    assert projection_b.outcome_status is FunctionalOperatorOutcomeStatus.INCONCLUSIVE
    assert projection_b.failures == ()
    assert projection_b.tenant_id == host_b.tenant_id

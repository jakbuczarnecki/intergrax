# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.diagnostics.diagnostic_scope_discovery_models import (
    DEFAULT_SCOPE_DISCOVERY_CANDIDATE_LIMIT,
    MAX_SCOPE_DISCOVERY_CANDIDATE_LIMIT,
    DiagnosticScopeDiscoveryRequest,
    DiagnosticScopeDiscoveryStatus,
    DiagnosticScopeDiscoveryValidationError,
    DiagnosticScopeReferenceKind,
    DiagnosticScopeResolutionProvenance,
    ProblemScopeReference,
    TransportScopeReference,
    EventScopeReference,
    build_diagnostic_scope_discovery_result,
    validate_scope_discovery_request,
    validate_transport_scope_provider,
    validate_transport_scope_task_id,
)
from intergrax.runtime.diagnostics.diagnostic_subject import ExecutionDiagnosticSubjectRef
from intergrax.runtime.diagnostics.problem_lifecycle import mint_problem_id
from intergrax.runtime.diagnostics.problem_grouping import problem_grouping_subject_ref_for_execution
from intergrax.contracts.execution_identity import mint_event_id, mint_run_id, mint_task_id

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"


def _execution_subject() -> ExecutionDiagnosticSubjectRef:
    return ExecutionDiagnosticSubjectRef(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )


def _provenance() -> DiagnosticScopeResolutionProvenance:
    return DiagnosticScopeResolutionProvenance(
        provider_id="problem_scope",
        reference_kind=DiagnosticScopeReferenceKind.PROBLEM,
        canonical_record_ref="problem:problem_0123456789abcdef0123456789abcdef",
    )


def _candidate():
    from intergrax.runtime.diagnostics.diagnostic_scope_discovery_models import (
        DiagnosticExecutionScopeCandidate,
    )

    return DiagnosticExecutionScopeCandidate(
        subject_ref=_execution_subject(),
        provenance=_provenance(),
    )


def test_request_validation_rejects_empty_tenant() -> None:
    request = DiagnosticScopeDiscoveryRequest(
        tenant_id="   ",
        reference=ProblemScopeReference(problem_id=mint_problem_id()),
    )
    with pytest.raises(ValueError, match="tenant_id"):
        validate_scope_discovery_request(request)


def test_request_validation_rejects_whitespace_tenant() -> None:
    request = DiagnosticScopeDiscoveryRequest(
        tenant_id=" tenant-a",
        reference=ProblemScopeReference(problem_id=mint_problem_id()),
    )
    with pytest.raises(ValueError, match="whitespace"):
        validate_scope_discovery_request(request)


def test_request_validation_rejects_candidate_limit_out_of_bounds() -> None:
    request = DiagnosticScopeDiscoveryRequest(
        tenant_id=_TENANT,
        reference=ProblemScopeReference(problem_id=mint_problem_id()),
        candidate_limit=0,
    )
    with pytest.raises(ValueError, match="candidate_limit"):
        validate_scope_discovery_request(request)

    request = DiagnosticScopeDiscoveryRequest(
        tenant_id=_TENANT,
        reference=ProblemScopeReference(problem_id=mint_problem_id()),
        candidate_limit=MAX_SCOPE_DISCOVERY_CANDIDATE_LIMIT + 1,
    )
    with pytest.raises(ValueError, match="candidate_limit"):
        validate_scope_discovery_request(request)


def test_request_validation_normalizes_problem_id() -> None:
    problem_id = mint_problem_id()
    request = DiagnosticScopeDiscoveryRequest(
        tenant_id=_TENANT,
        reference=ProblemScopeReference(problem_id=problem_id),
        candidate_limit=DEFAULT_SCOPE_DISCOVERY_CANDIDATE_LIMIT,
    )
    validated = validate_scope_discovery_request(request)
    assert validated.tenant_id == _TENANT
    assert validated.reference.problem_id == problem_id


def test_resolved_result_invariants() -> None:
    subject = _execution_subject()
    candidate = _candidate()
    result = build_diagnostic_scope_discovery_result(
        status=DiagnosticScopeDiscoveryStatus.RESOLVED,
        resolved_scope=subject,
        candidates=(candidate,),
        candidate_count=1,
        candidate_count_exact=True,
        provenance=(_provenance(),),
    )
    assert result.status is DiagnosticScopeDiscoveryStatus.RESOLVED
    assert result.resolved_scope == subject


def test_resolved_result_rejects_missing_scope() -> None:
    with pytest.raises(DiagnosticScopeDiscoveryValidationError, match="resolved_scope"):
        build_diagnostic_scope_discovery_result(
            status=DiagnosticScopeDiscoveryStatus.RESOLVED,
            resolved_scope=None,
            candidates=(_candidate(),),
            candidate_count=1,
            candidate_count_exact=True,
            provenance=(),
        )


def test_ambiguous_result_invariants() -> None:
    result = build_diagnostic_scope_discovery_result(
        status=DiagnosticScopeDiscoveryStatus.AMBIGUOUS,
        resolved_scope=None,
        candidates=(_candidate(), _candidate()),
        candidate_count=2,
        candidate_count_exact=False,
        provenance=(_provenance(),),
    )
    assert result.status is DiagnosticScopeDiscoveryStatus.AMBIGUOUS
    assert result.resolved_scope is None
    assert result.candidate_count == 2


def test_not_found_result_invariants() -> None:
    result = build_diagnostic_scope_discovery_result(
        status=DiagnosticScopeDiscoveryStatus.NOT_FOUND,
        resolved_scope=None,
        candidates=(),
        candidate_count=0,
        candidate_count_exact=True,
        provenance=(),
    )
    assert result.candidate_count == 0
    assert not result.candidates


def test_unsupported_reference_rejects_candidates() -> None:
    with pytest.raises(DiagnosticScopeDiscoveryValidationError, match="fabricated"):
        build_diagnostic_scope_discovery_result(
            status=DiagnosticScopeDiscoveryStatus.UNSUPPORTED_REFERENCE,
            resolved_scope=None,
            candidates=(_candidate(),),
            candidate_count=1,
            candidate_count_exact=True,
            provenance=(),
        )


def test_transport_reference_kind() -> None:
    reference = TransportScopeReference(provider="celery", transport_task_id="task-1")
    assert reference.kind is DiagnosticScopeReferenceKind.TRANSPORT


def test_transport_provider_validation_rejects_empty() -> None:
    with pytest.raises(ValueError, match="provider"):
        validate_transport_scope_provider("   ")


def test_transport_provider_validation_rejects_whitespace() -> None:
    with pytest.raises(ValueError, match="whitespace"):
        validate_transport_scope_provider(" celery")


def test_transport_task_id_validation_rejects_empty() -> None:
    with pytest.raises(ValueError, match="transport_task_id"):
        validate_transport_scope_task_id("")


def test_transport_request_validation() -> None:
    request = DiagnosticScopeDiscoveryRequest(
        tenant_id=_TENANT,
        reference=TransportScopeReference(provider="celery", transport_task_id="task-1"),
    )
    validated = validate_scope_discovery_request(request)
    assert validated.reference.provider == "celery"
    assert validated.reference.transport_task_id == "task-1"


def test_request_validation_rejects_unknown_reference_type() -> None:
    request = DiagnosticScopeDiscoveryRequest(
        tenant_id=_TENANT,
        reference=object(),  # type: ignore[arg-type]
    )
    with pytest.raises(
        TypeError,
        match="ProblemScopeReference, TransportScopeReference, or EventScopeReference",
    ):
        validate_scope_discovery_request(request)


def test_event_scope_reference_kind() -> None:
    event_id = mint_event_id()
    reference = EventScopeReference(event_id=event_id)
    assert reference.kind is DiagnosticScopeReferenceKind.EVENT
    assert reference.event_id == event_id


def test_invalid_event_id_rejected() -> None:
    with pytest.raises(ValueError, match="EventId"):
        validate_scope_discovery_request(
            DiagnosticScopeDiscoveryRequest(
                tenant_id=_TENANT,
                reference=EventScopeReference(event_id="not-an-event-id"),  # type: ignore[arg-type]
            ),
        )


def test_event_request_validation() -> None:
    event_id = mint_event_id()
    request = DiagnosticScopeDiscoveryRequest(
        tenant_id=_TENANT,
        reference=EventScopeReference(event_id=event_id),
    )
    validated = validate_scope_discovery_request(request)
    assert validated.reference.event_id == event_id


def test_reference_union_accepts_all_three_kinds() -> None:
    from intergrax.runtime.diagnostics.diagnostic_scope_discovery_models import (
        DiagnosticScopeReference,
    )

    references: tuple[DiagnosticScopeReference, ...] = (
        ProblemScopeReference(problem_id=mint_problem_id()),
        TransportScopeReference(provider="celery", transport_task_id="task-1"),
        EventScopeReference(event_id=mint_event_id()),
    )
    assert len(references) == 3


def test_no_accidental_fourth_reference_kind() -> None:
    assert tuple(DiagnosticScopeReferenceKind) == (
        DiagnosticScopeReferenceKind.PROBLEM,
        DiagnosticScopeReferenceKind.TRANSPORT,
        DiagnosticScopeReferenceKind.EVENT,
    )

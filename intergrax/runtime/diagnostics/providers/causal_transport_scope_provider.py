# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Transport-task-backed diagnostic execution scope discovery via causal evidence (DG-002 Slice 2)."""

from __future__ import annotations

from intergrax.runtime.diagnostics.diagnostic_scope_discovery_models import (
    DiagnosticExecutionScopeCandidate,
    DiagnosticScopeDiscoveryResult,
    DiagnosticScopeDiscoveryStatus,
    DiagnosticScopeReferenceKind,
    DiagnosticScopeResolutionProvenance,
    TransportScopeReference,
    build_diagnostic_scope_discovery_result,
    validate_scope_discovery_candidate_limit,
    validate_scope_discovery_tenant_id,
    validate_transport_scope_provider,
    validate_transport_scope_task_id,
)
from intergrax.runtime.diagnostics.diagnostic_scope_discovery_provider import (
    DiagnosticScopeProviderIntegrityError,
    DiagnosticScopeProviderResult,
    DiagnosticScopeProviderUnavailableError,
    validate_scope_provider_result,
)
from intergrax.runtime.diagnostics.diagnostic_subject import (
    ExecutionDiagnosticSubjectRef,
    validate_execution_diagnostic_subject_ref,
)
from intergrax.runtime.observability.causal_evidence import (
    CausalRelationKind,
    PlatformCausalEvidence,
)
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
    CausalEvidencePersistenceIntegrityError,
)

CAUSAL_TRANSPORT_SCOPE_PROVIDER_ID = "causal_transport_scope"

_ACCEPTED_RELATION_KIND = CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION


class CausalTransportScopeProvider:
    """Resolve execution diagnostic scope from tenant-scoped transport task identity."""

    def __init__(
        self,
        *,
        causal_evidence_persistence: CausalEvidencePersistence,
    ) -> None:
        self._causal_evidence_persistence = causal_evidence_persistence

    @property
    def provider_id(self) -> str:
        return CAUSAL_TRANSPORT_SCOPE_PROVIDER_ID

    @property
    def supported_reference_kind(self) -> DiagnosticScopeReferenceKind:
        return DiagnosticScopeReferenceKind.TRANSPORT

    def discover(
        self,
        *,
        tenant_id: str,
        reference: TransportScopeReference,
        candidate_limit: int,
    ) -> DiagnosticScopeProviderResult:
        tenant_id = validate_scope_discovery_tenant_id(tenant_id)
        candidate_limit = validate_scope_discovery_candidate_limit(candidate_limit)
        provider = validate_transport_scope_provider(reference.provider)
        transport_task_id = validate_transport_scope_task_id(reference.transport_task_id)
        normalized_reference = TransportScopeReference(
            provider=provider,
            transport_task_id=transport_task_id,
        )
        request_provenance = _transport_request_provenance(reference=normalized_reference)

        evidence_records = _list_transport_evidence(
            self._causal_evidence_persistence,
            tenant_id=tenant_id,
            reference=normalized_reference,
        )
        if not evidence_records:
            return _provider_result_from_public(
                build_diagnostic_scope_discovery_result(
                    status=DiagnosticScopeDiscoveryStatus.NOT_FOUND,
                    resolved_scope=None,
                    candidates=(),
                    candidate_count=0,
                    candidate_count_exact=True,
                    provenance=(request_provenance,),
                ),
            )

        execution_scopes = _collect_execution_scopes(
            evidence_records,
            tenant_id=tenant_id,
            reference=normalized_reference,
        )
        return validate_scope_provider_result(
            _classify_execution_scopes(
                execution_scopes,
                candidate_limit=candidate_limit,
                request_provenance=request_provenance,
            ),
        )


def _list_transport_evidence(
    causal_evidence_persistence: CausalEvidencePersistence,
    *,
    tenant_id: str,
    reference: TransportScopeReference,
) -> tuple[PlatformCausalEvidence, ...]:
    try:
        return causal_evidence_persistence.list_for_transport_task(
            tenant_id=tenant_id,
            provider=reference.provider,
            transport_task_id=reference.transport_task_id,
        )
    except CausalEvidencePersistenceIntegrityError as exc:
        raise DiagnosticScopeProviderIntegrityError(str(exc)) from exc
    except (ConnectionError, TimeoutError, OSError) as exc:
        raise DiagnosticScopeProviderUnavailableError(str(exc)) from exc


def _transport_request_provenance(
    *,
    reference: TransportScopeReference,
) -> DiagnosticScopeResolutionProvenance:
    return DiagnosticScopeResolutionProvenance(
        provider_id=CAUSAL_TRANSPORT_SCOPE_PROVIDER_ID,
        reference_kind=DiagnosticScopeReferenceKind.TRANSPORT,
        canonical_record_ref=(
            f"transport:{reference.provider}:{reference.transport_task_id}"
        ),
    )


def _evidence_provenance(
    evidence: PlatformCausalEvidence,
) -> DiagnosticScopeResolutionProvenance:
    return DiagnosticScopeResolutionProvenance(
        provider_id=CAUSAL_TRANSPORT_SCOPE_PROVIDER_ID,
        reference_kind=DiagnosticScopeReferenceKind.TRANSPORT,
        canonical_record_ref=f"causal_evidence:{evidence.evidence_id}",
    )


def _provider_result_from_public(
    result: DiagnosticScopeDiscoveryResult,
) -> DiagnosticScopeProviderResult:
    return DiagnosticScopeProviderResult(
        status=result.status,
        resolved_scope=result.resolved_scope,
        candidates=result.candidates,
        candidate_count=result.candidate_count,
        candidate_count_exact=result.candidate_count_exact,
        provenance=result.provenance,
        limitations=result.limitations,
    )


def _collect_execution_scopes(
    evidence_records: tuple[PlatformCausalEvidence, ...],
    *,
    tenant_id: str,
    reference: TransportScopeReference,
) -> dict[tuple[str, str], DiagnosticExecutionScopeCandidate]:
    execution_scopes: dict[tuple[str, str], DiagnosticExecutionScopeCandidate] = {}
    for evidence in evidence_records:
        _validate_evidence_integrity(
            evidence,
            tenant_id=tenant_id,
            reference=reference,
        )
        subject_ref = _execution_subject_from_evidence(evidence, tenant_id=tenant_id)
        identity = (str(subject_ref.task_id), str(subject_ref.run_id))
        if identity in execution_scopes:
            continue
        execution_scopes[identity] = DiagnosticExecutionScopeCandidate(
            subject_ref=subject_ref,
            provenance=_evidence_provenance(evidence),
        )
    return execution_scopes


def _validate_evidence_integrity(
    evidence: PlatformCausalEvidence,
    *,
    tenant_id: str,
    reference: TransportScopeReference,
) -> None:
    if evidence.tenant_id != tenant_id:
        raise DiagnosticScopeProviderIntegrityError(
            "causal evidence tenant_id does not match discovery request tenant",
        )
    if evidence.source.tenant_id != tenant_id:
        raise DiagnosticScopeProviderIntegrityError(
            "causal evidence source tenant_id does not match discovery request tenant",
        )
    if evidence.target.tenant_id != tenant_id:
        raise DiagnosticScopeProviderIntegrityError(
            "causal evidence target tenant_id does not match discovery request tenant",
        )
    if evidence.source.provider != reference.provider:
        raise DiagnosticScopeProviderIntegrityError(
            "causal evidence source provider does not match transport reference provider",
        )
    if evidence.source.task_id != reference.transport_task_id:
        raise DiagnosticScopeProviderIntegrityError(
            "causal evidence source task_id does not match transport reference transport_task_id",
        )
    if evidence.relation_kind is not _ACCEPTED_RELATION_KIND:
        raise DiagnosticScopeProviderIntegrityError(
            "causal evidence relation_kind is not transport-task-triggered execution",
        )


def _execution_subject_from_evidence(
    evidence: PlatformCausalEvidence,
    *,
    tenant_id: str,
) -> ExecutionDiagnosticSubjectRef:
    try:
        subject_ref = ExecutionDiagnosticSubjectRef(
            tenant_id=tenant_id,
            task_id=evidence.target.task_id,
            run_id=evidence.target.run_id,
        )
        return validate_execution_diagnostic_subject_ref(subject_ref)
    except (TypeError, ValueError) as exc:
        raise DiagnosticScopeProviderIntegrityError(
            "causal evidence target cannot be validated as execution diagnostic scope",
        ) from exc


def _classify_execution_scopes(
    execution_scopes: dict[tuple[str, str], DiagnosticExecutionScopeCandidate],
    *,
    candidate_limit: int,
    request_provenance: DiagnosticScopeResolutionProvenance,
) -> DiagnosticScopeProviderResult:
    ordered_candidates = _ordered_execution_candidates(execution_scopes)
    distinct_count = len(ordered_candidates)

    if distinct_count == 1:
        resolved_scope = ordered_candidates[0].subject_ref
        return _provider_result_from_public(
            build_diagnostic_scope_discovery_result(
                status=DiagnosticScopeDiscoveryStatus.RESOLVED,
                resolved_scope=resolved_scope,
                candidates=(ordered_candidates[0],),
                candidate_count=1,
                candidate_count_exact=True,
                provenance=(request_provenance,),
            ),
        )

    return _provider_result_from_public(
        build_diagnostic_scope_discovery_result(
            status=DiagnosticScopeDiscoveryStatus.AMBIGUOUS,
            resolved_scope=None,
            candidates=tuple(ordered_candidates[:candidate_limit]),
            candidate_count=distinct_count,
            candidate_count_exact=True,
            provenance=(request_provenance,),
        ),
    )


def _ordered_execution_candidates(
    execution_scopes: dict[tuple[str, str], DiagnosticExecutionScopeCandidate],
) -> list[DiagnosticExecutionScopeCandidate]:
    return sorted(
        execution_scopes.values(),
        key=lambda candidate: (
            str(candidate.subject_ref.task_id),
            str(candidate.subject_ref.run_id),
        ),
    )

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
    CausalEvidencePage,
    CausalEvidencePersistence,
    CausalEvidencePersistenceIntegrityError,
)

CAUSAL_TRANSPORT_SCOPE_PROVIDER_ID = "causal_transport_scope"

_CAUSAL_EVIDENCE_PAGE_SIZE = 100
_MAX_EXAMINED_CAUSAL_EVIDENCE = 1000
_ACCEPTED_RELATION_KIND = CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION
_TRUNCATION_LIMITATION = (
    "causal evidence examination reached hard bound; additional matching "
    "causal evidence remains and execution scope uniqueness cannot be proven"
)
_AMBIGUOUS_TRUNCATION_LIMITATION = (
    "candidate set may be incomplete due to causal-evidence examination budget"
)


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

        scan = _scan_transport_execution_scopes(
            self._causal_evidence_persistence,
            tenant_id=tenant_id,
            reference=normalized_reference,
        )
        return validate_scope_provider_result(
            _classify_execution_scopes(
                scan,
                candidate_limit=candidate_limit,
                request_provenance=request_provenance,
            ),
        )


class _TransportScopeScan:
    def __init__(self) -> None:
        self.execution_scopes: dict[tuple[str, str], DiagnosticExecutionScopeCandidate] = {}
        self.examined_count = 0
        self.saw_evidence = False
        self.scan_complete = False


def _page_transport_evidence(
    causal_evidence_persistence: CausalEvidencePersistence,
    *,
    tenant_id: str,
    reference: TransportScopeReference,
    limit: int,
    cursor: str | None,
) -> CausalEvidencePage:
    try:
        return causal_evidence_persistence.page_for_transport_task(
            tenant_id=tenant_id,
            provider=reference.provider,
            transport_task_id=reference.transport_task_id,
            limit=limit,
            cursor=cursor,
        )
    except CausalEvidencePersistenceIntegrityError as exc:
        raise DiagnosticScopeProviderIntegrityError(str(exc)) from exc
    except (ConnectionError, TimeoutError, OSError) as exc:
        raise DiagnosticScopeProviderUnavailableError(str(exc)) from exc


def _scan_transport_execution_scopes(
    causal_evidence_persistence: CausalEvidencePersistence,
    *,
    tenant_id: str,
    reference: TransportScopeReference,
) -> _TransportScopeScan:
    scan = _TransportScopeScan()
    cursor: str | None = None
    seen_cursors: set[str | None] = {None}

    while scan.examined_count < _MAX_EXAMINED_CAUSAL_EVIDENCE:
        remaining = _MAX_EXAMINED_CAUSAL_EVIDENCE - scan.examined_count
        page_limit = min(_CAUSAL_EVIDENCE_PAGE_SIZE, remaining)
        page = _page_transport_evidence(
            causal_evidence_persistence,
            tenant_id=tenant_id,
            reference=reference,
            limit=page_limit,
            cursor=cursor,
        )

        if not page.items:
            if page.next_cursor is None:
                scan.scan_complete = True
                break
            raise DiagnosticScopeProviderIntegrityError(
                "causal evidence page returned empty items with continuation cursor",
            )

        if page.next_cursor is not None:
            if page.next_cursor == cursor:
                raise DiagnosticScopeProviderIntegrityError(
                    "causal evidence paging cursor did not advance",
                )
            if page.next_cursor in seen_cursors:
                raise DiagnosticScopeProviderIntegrityError(
                    "causal evidence paging cursor cycle detected",
                )

        for evidence in page.items:
            scan.examined_count += 1
            scan.saw_evidence = True
            _validate_evidence_integrity(
                evidence,
                tenant_id=tenant_id,
                reference=reference,
            )
            subject_ref = _execution_subject_from_evidence(evidence, tenant_id=tenant_id)
            identity = (str(subject_ref.task_id), str(subject_ref.run_id))
            if identity in scan.execution_scopes:
                continue
            scan.execution_scopes[identity] = DiagnosticExecutionScopeCandidate(
                subject_ref=subject_ref,
                provenance=_evidence_provenance(evidence),
            )

        if page.next_cursor is None:
            scan.scan_complete = True
            break

        if scan.examined_count >= _MAX_EXAMINED_CAUSAL_EVIDENCE:
            break

        seen_cursors.add(page.next_cursor)
        cursor = page.next_cursor

    return scan


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
    scan: _TransportScopeScan,
    *,
    candidate_limit: int,
    request_provenance: DiagnosticScopeResolutionProvenance,
) -> DiagnosticScopeProviderResult:
    ordered_candidates = _ordered_execution_candidates(scan.execution_scopes)
    distinct_count = len(ordered_candidates)
    limitations: list[str] = []

    if not scan.saw_evidence and scan.scan_complete:
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

    if not scan.scan_complete and distinct_count <= 1:
        return _provider_result_from_public(
            build_diagnostic_scope_discovery_result(
                status=DiagnosticScopeDiscoveryStatus.INSUFFICIENT_EVIDENCE,
                resolved_scope=None,
                candidates=tuple(ordered_candidates[:candidate_limit]),
                candidate_count=distinct_count,
                candidate_count_exact=False,
                provenance=(request_provenance,),
                limitations=(_TRUNCATION_LIMITATION,),
            ),
        )

    if not scan.scan_complete and distinct_count >= 2:
        limitations.append(_AMBIGUOUS_TRUNCATION_LIMITATION)
        return _provider_result_from_public(
            build_diagnostic_scope_discovery_result(
                status=DiagnosticScopeDiscoveryStatus.AMBIGUOUS,
                resolved_scope=None,
                candidates=tuple(ordered_candidates[:candidate_limit]),
                candidate_count=distinct_count,
                candidate_count_exact=False,
                provenance=(request_provenance,),
                limitations=tuple(limitations),
            ),
        )

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
            limitations=tuple(limitations),
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

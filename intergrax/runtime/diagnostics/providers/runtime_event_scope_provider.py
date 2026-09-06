# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""EventId-backed diagnostic execution scope discovery via canonical RuntimeEvent (DG-002 Slice 3)."""

from __future__ import annotations

from intergrax.contracts.execution_identity import EventId, validate_event_id
from intergrax.runtime.diagnostics.diagnostic_scope_discovery_models import (
    DiagnosticExecutionScopeCandidate,
    DiagnosticScopeDiscoveryResult,
    DiagnosticScopeDiscoveryStatus,
    DiagnosticScopeReferenceKind,
    DiagnosticScopeResolutionProvenance,
    EventScopeReference,
    build_diagnostic_scope_discovery_result,
    validate_scope_discovery_candidate_limit,
    validate_scope_discovery_tenant_id,
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
from intergrax.runtime.events.execution_position import PositionedRuntimeEvent
from intergrax.runtime.events.persistence_contract import (
    RuntimeEventPersistence,
    RuntimeEventPersistenceIntegrityError,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent

RUNTIME_EVENT_SCOPE_PROVIDER_ID = "runtime_event_scope"


class RuntimeEventScopeProvider:
    """Resolve execution diagnostic scope from tenant-scoped EventId."""

    def __init__(
        self,
        *,
        runtime_event_persistence: RuntimeEventPersistence,
    ) -> None:
        self._runtime_event_persistence = runtime_event_persistence

    @property
    def provider_id(self) -> str:
        return RUNTIME_EVENT_SCOPE_PROVIDER_ID

    @property
    def supported_reference_kind(self) -> DiagnosticScopeReferenceKind:
        return DiagnosticScopeReferenceKind.EVENT

    def discover(
        self,
        *,
        tenant_id: str,
        reference: EventScopeReference,
        candidate_limit: int,
    ) -> DiagnosticScopeProviderResult:
        tenant_id = validate_scope_discovery_tenant_id(tenant_id)
        validate_scope_discovery_candidate_limit(candidate_limit)
        event_id = validate_event_id(reference.event_id)
        provenance = _event_provenance(event_id=event_id)

        positioned = _get_by_event_id(
            self._runtime_event_persistence,
            tenant_id=tenant_id,
            event_id=event_id,
        )
        if positioned is None:
            return _provider_result_from_public(
                build_diagnostic_scope_discovery_result(
                    status=DiagnosticScopeDiscoveryStatus.NOT_FOUND,
                    resolved_scope=None,
                    candidates=(),
                    candidate_count=0,
                    candidate_count_exact=True,
                    provenance=(provenance,),
                ),
            )

        _validate_positioned_event_identity(positioned, event_id=event_id)
        subject_ref = _execution_subject_from_event(
            positioned.event,
            tenant_id=tenant_id,
        )
        candidate = DiagnosticExecutionScopeCandidate(
            subject_ref=subject_ref,
            provenance=provenance,
        )
        return validate_scope_provider_result(
            _provider_result_from_public(
                build_diagnostic_scope_discovery_result(
                    status=DiagnosticScopeDiscoveryStatus.RESOLVED,
                    resolved_scope=subject_ref,
                    candidates=(candidate,),
                    candidate_count=1,
                    candidate_count_exact=True,
                    provenance=(provenance,),
                ),
            ),
        )


def _get_by_event_id(
    runtime_event_persistence: RuntimeEventPersistence,
    *,
    tenant_id: str,
    event_id: EventId,
) -> PositionedRuntimeEvent | None:
    try:
        return runtime_event_persistence.get_by_event_id(
            tenant_id=tenant_id,
            event_id=event_id,
        )
    except RuntimeEventPersistenceIntegrityError as exc:
        raise DiagnosticScopeProviderIntegrityError(str(exc)) from exc
    except (ConnectionError, TimeoutError, OSError) as exc:
        raise DiagnosticScopeProviderUnavailableError(str(exc)) from exc


def _validate_positioned_event_identity(
    positioned: PositionedRuntimeEvent,
    *,
    event_id: EventId,
) -> None:
    if positioned.event_id != event_id:
        raise DiagnosticScopeProviderIntegrityError(
            "persisted RuntimeEvent event_id does not match discovery request event_id",
        )


def _execution_subject_from_event(
    event: RuntimeEvent,
    *,
    tenant_id: str,
) -> ExecutionDiagnosticSubjectRef:
    try:
        subject_ref = ExecutionDiagnosticSubjectRef(
            tenant_id=tenant_id,
            task_id=event.task_id,
            run_id=event.run_id,
        )
        return validate_execution_diagnostic_subject_ref(subject_ref)
    except (TypeError, ValueError) as exc:
        raise DiagnosticScopeProviderIntegrityError(
            "persisted RuntimeEvent cannot be validated as execution diagnostic scope",
        ) from exc


def _event_provenance(
    *,
    event_id: EventId,
) -> DiagnosticScopeResolutionProvenance:
    return DiagnosticScopeResolutionProvenance(
        provider_id=RUNTIME_EVENT_SCOPE_PROVIDER_ID,
        reference_kind=DiagnosticScopeReferenceKind.EVENT,
        canonical_record_ref=f"runtime_event:{event_id}",
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

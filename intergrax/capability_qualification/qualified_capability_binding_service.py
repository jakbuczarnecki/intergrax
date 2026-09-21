# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Qualified capability binding coordination (UCA-6C)."""

from __future__ import annotations

from datetime import UTC, datetime

from intergrax.capability_qualification.qualified_capability_binding_registry import (
    QualifiedCapabilityBindingProviderRegistry,
)
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityBindingOutcome,
    QualifiedCapabilityBindingProvider,
    QualifiedCapabilityBindingReasonCode,
    QualifiedCapabilityBindingRequest,
    QualifiedCapabilityBindingResult,
)


class QualifiedCapabilityBindingService:
    """Dispatch qualified-subject binding to plugin providers — no AW routing."""

    def __init__(
        self,
        providers: tuple[QualifiedCapabilityBindingProvider, ...],
    ) -> None:
        self._registry = QualifiedCapabilityBindingProviderRegistry(providers)

    def bind(
        self,
        request: QualifiedCapabilityBindingRequest,
    ) -> QualifiedCapabilityBindingResult:
        started_at = datetime.now(tz=UTC)
        try:
            provider = self._registry.resolve(request)
        except Exception as exc:
            return _terminal(
                request=request,
                outcome=QualifiedCapabilityBindingOutcome.FAILED,
                reason_code=QualifiedCapabilityBindingReasonCode.INTERNAL_ERROR,
                started_at=started_at,
                reason_detail=str(exc),
            )
        if provider is None:
            return _terminal(
                request=request,
                outcome=QualifiedCapabilityBindingOutcome.NO_PROVIDER,
                reason_code=QualifiedCapabilityBindingReasonCode.PROVIDER_UNAVAILABLE,
                started_at=started_at,
            )
        result = provider.bind(request)
        if result.binding_operation_id != request.binding_operation_id:
            completed_at = datetime.now(tz=UTC)
            return QualifiedCapabilityBindingResult(
                binding_operation_id=request.binding_operation_id,
                outcome=QualifiedCapabilityBindingOutcome.CONFLICT,
                reason_code=QualifiedCapabilityBindingReasonCode.INTEGRITY_CONFLICT,
                started_at=started_at,
                completed_at=completed_at,
                reason_detail="binding_operation_id mismatch",
            )
        if (
            result.outcome is QualifiedCapabilityBindingOutcome.BOUND
            and result.execution_target is not None
            and result.execution_target.qualified_subject_reference
            != request.qualified_subject.qualified_subject_reference
        ):
            completed_at = datetime.now(tz=UTC)
            return QualifiedCapabilityBindingResult(
                binding_operation_id=request.binding_operation_id,
                outcome=QualifiedCapabilityBindingOutcome.CONFLICT,
                reason_code=QualifiedCapabilityBindingReasonCode.SUBJECT_MISMATCH,
                started_at=started_at,
                completed_at=completed_at,
                reason_detail="qualified subject identity changed during binding",
            )
        return result


def _terminal(
    *,
    request: QualifiedCapabilityBindingRequest,
    outcome: QualifiedCapabilityBindingOutcome,
    reason_code: QualifiedCapabilityBindingReasonCode,
    started_at: datetime,
    reason_detail: str = "",
) -> QualifiedCapabilityBindingResult:
    completed_at = datetime.now(tz=UTC)
    return QualifiedCapabilityBindingResult(
        binding_operation_id=request.binding_operation_id,
        outcome=outcome,
        reason_code=reason_code,
        started_at=started_at,
        completed_at=completed_at,
        reason_detail=reason_detail,
    )


__all__ = ["QualifiedCapabilityBindingService"]

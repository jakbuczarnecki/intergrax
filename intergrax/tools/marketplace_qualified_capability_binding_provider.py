# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Marketplace Tool qualified capability binding provider (S24-GAP-02-P2)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityBindingOutcome,
    QualifiedCapabilityBindingReasonCode,
    QualifiedCapabilityBindingRequest,
    QualifiedCapabilityBindingResult,
    QualifiedCapabilityExecutionTarget,
)
from intergrax.contracts.capability_qualification.qualified_subject import (
    QualifiedCapabilitySubjectKind,
)
from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)
from intergrax.contracts.marketplace.handoff_traceability import (
    CapabilityHandoffConsumerTarget,
)
from intergrax.contracts.tools.marketplace_qualified_capability import (
    MarketplaceQualifiedToolStageRepository,
    MarketplaceQualifiedToolStageUnavailableError,
)
from intergrax.contracts.tools.marketplace_qualified_tool_stage_context import (
    MarketplaceQualifiedToolStageContextNotFoundError,
    MarketplaceQualifiedToolStageContextResolver,
    MarketplaceQualifiedToolStageContextResolverConflictError,
    MarketplaceQualifiedToolStageContextResolverIntegrityError,
    MarketplaceQualifiedToolStageContextResolverNotSupportedError,
    MarketplaceQualifiedToolStageContextResolverUnavailableError,
)

MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID: Final = (
    "marketplace.tool.qualified_binding.v1"
)
_MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID: Final = "marketplace.gap_acquisition.v1"
_EXECUTION_TARGET_PREFIX: Final = "marketplace-qualified-tool:v1:"


def execution_target_reference_for_marketplace_qualified_tool(handoff_id: str) -> str:
    from intergrax.contracts.capability_catalog._validation import require_non_empty_text

    normalized = require_non_empty_text(handoff_id, label="handoff_id")
    return f"{_EXECUTION_TARGET_PREFIX}{normalized}"


class MarketplaceToolQualifiedCapabilityBindingProvider:
    """Bind qualified Marketplace Tool handoffs to opaque execution targets — no activation."""

    def __init__(
        self,
        *,
        stage_repository: MarketplaceQualifiedToolStageRepository,
        context_resolver: MarketplaceQualifiedToolStageContextResolver,
    ) -> None:
        self._stage_repository = stage_repository
        self._context_resolver = context_resolver

    @property
    def provider_id(self) -> str:
        return MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID

    def supports(self, request: QualifiedCapabilityBindingRequest) -> bool:
        subject = request.qualified_subject
        if subject.subject_kind is not QualifiedCapabilitySubjectKind.DOMAIN_HANDOFF_REFERENCE:
            return False
        qualification = request.qualification_result
        if qualification.outcome is not CapabilityQualificationOutcome.QUALIFIED:
            return False
        if qualification.strategy_id != _MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID:
            return False
        evidence = qualification.evidence
        if evidence is None or not evidence.domain_handoff_reference:
            return False
        return evidence.domain_handoff_reference == subject.subject_reference

    def bind(
        self,
        request: QualifiedCapabilityBindingRequest,
    ) -> QualifiedCapabilityBindingResult:
        started_at = datetime.now(tz=UTC)
        qualification = request.qualification_result
        if qualification.outcome is not CapabilityQualificationOutcome.QUALIFIED:
            return _terminal(
                request=request,
                outcome=QualifiedCapabilityBindingOutcome.NOT_SUPPORTED,
                reason_code=QualifiedCapabilityBindingReasonCode.SUBJECT_NOT_SUPPORTED,
                started_at=started_at,
            )

        subject_ref = request.qualified_subject.subject_reference
        evidence = qualification.evidence
        if evidence is None or evidence.domain_handoff_reference != subject_ref:
            return _terminal(
                request=request,
                outcome=QualifiedCapabilityBindingOutcome.CONFLICT,
                reason_code=QualifiedCapabilityBindingReasonCode.SUBJECT_MISMATCH,
                started_at=started_at,
            )

        try:
            ctx = self._context_resolver.resolve_for_qualification(
                acquisition_request_id=qualification.acquisition_request_id,
                domain_handoff_reference=subject_ref,
                strategy_id=qualification.strategy_id,
            )
        except MarketplaceQualifiedToolStageContextResolverNotSupportedError:
            return _terminal(
                request=request,
                outcome=QualifiedCapabilityBindingOutcome.NOT_SUPPORTED,
                reason_code=QualifiedCapabilityBindingReasonCode.SUBJECT_NOT_SUPPORTED,
                started_at=started_at,
            )
        except MarketplaceQualifiedToolStageContextNotFoundError:
            return _terminal(
                request=request,
                outcome=QualifiedCapabilityBindingOutcome.UNAVAILABLE,
                reason_code=QualifiedCapabilityBindingReasonCode.PROVIDER_UNAVAILABLE,
                started_at=started_at,
            )
        except MarketplaceQualifiedToolStageContextResolverUnavailableError:
            return _terminal(
                request=request,
                outcome=QualifiedCapabilityBindingOutcome.UNAVAILABLE,
                reason_code=QualifiedCapabilityBindingReasonCode.PROVIDER_UNAVAILABLE,
                started_at=started_at,
            )
        except (
            MarketplaceQualifiedToolStageContextResolverConflictError,
            MarketplaceQualifiedToolStageContextResolverIntegrityError,
        ):
            return _terminal(
                request=request,
                outcome=QualifiedCapabilityBindingOutcome.CONFLICT,
                reason_code=QualifiedCapabilityBindingReasonCode.INTEGRITY_CONFLICT,
                started_at=started_at,
            )

        if ctx.tenant_id != request.tenant_id:
            return _terminal(
                request=request,
                outcome=QualifiedCapabilityBindingOutcome.CONFLICT,
                reason_code=QualifiedCapabilityBindingReasonCode.SUBJECT_MISMATCH,
                started_at=started_at,
                reason_detail="tenant_id mismatch",
            )

        try:
            stage = self._stage_repository.get(
                tenant_id=ctx.tenant_id,
                handoff_id=ctx.handoff_id,
            )
        except MarketplaceQualifiedToolStageUnavailableError:
            return _terminal(
                request=request,
                outcome=QualifiedCapabilityBindingOutcome.UNAVAILABLE,
                reason_code=QualifiedCapabilityBindingReasonCode.PROVIDER_UNAVAILABLE,
                started_at=started_at,
            )

        if stage is None:
            return _terminal(
                request=request,
                outcome=QualifiedCapabilityBindingOutcome.UNAVAILABLE,
                reason_code=QualifiedCapabilityBindingReasonCode.PROVIDER_UNAVAILABLE,
                started_at=started_at,
                reason_detail="staged tool release missing",
            )

        if stage.selected_release.discovery.kind is not CapabilityKind.TOOL:
            return _terminal(
                request=request,
                outcome=QualifiedCapabilityBindingOutcome.CONFLICT,
                reason_code=QualifiedCapabilityBindingReasonCode.INTEGRITY_CONFLICT,
                started_at=started_at,
            )

        if stage.consumer_target is not CapabilityHandoffConsumerTarget.TOOL_DOMAIN:
            return _terminal(
                request=request,
                outcome=QualifiedCapabilityBindingOutcome.CONFLICT,
                reason_code=QualifiedCapabilityBindingReasonCode.INTEGRITY_CONFLICT,
                started_at=started_at,
            )

        completed_at = datetime.now(tz=UTC)
        target = QualifiedCapabilityExecutionTarget(
            execution_target_reference=execution_target_reference_for_marketplace_qualified_tool(
                ctx.handoff_id,
            ),
            binding_provider_id=self.provider_id,
            qualified_subject_reference=request.qualified_subject.qualified_subject_reference,
        )
        return QualifiedCapabilityBindingResult(
            binding_operation_id=request.binding_operation_id,
            outcome=QualifiedCapabilityBindingOutcome.BOUND,
            reason_code=QualifiedCapabilityBindingReasonCode.NONE,
            provider_id=self.provider_id,
            execution_target=target,
            started_at=started_at,
            completed_at=completed_at,
        )


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
        provider_id=MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID,
        started_at=started_at,
        completed_at=completed_at,
        reason_detail=reason_detail,
    )


__all__ = [
    "MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID",
    "MarketplaceToolQualifiedCapabilityBindingProvider",
    "execution_target_reference_for_marketplace_qualified_tool",
]

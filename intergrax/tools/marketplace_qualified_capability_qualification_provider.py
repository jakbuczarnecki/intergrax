# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Marketplace Tool capability qualification provider (S24-GAP-02-P2)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

from intergrax.contracts.capability_acquisition.acquisition_outcome import (
    CapabilityAcquisitionOutcome,
)
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_qualification.qualification_evidence import (
    CapabilityQualificationEvidence,
)
from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)
from intergrax.contracts.capability_qualification.qualification_reason_code import (
    CapabilityQualificationReasonCode,
)
from intergrax.contracts.capability_qualification.qualification_request import (
    CapabilityQualificationRequest,
)
from intergrax.contracts.capability_qualification.qualification_result import (
    CapabilityQualificationResult,
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

MARKETPLACE_TOOL_CAPABILITY_QUALIFICATION_PROVIDER_ID: Final = (
    "marketplace.tool.qualification.v1"
)
_MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID: Final = "marketplace.gap_acquisition.v1"


class MarketplaceToolCapabilityQualificationProvider:
    """Qualify Marketplace gap Tool handoffs using staged release evidence only."""

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
        return MARKETPLACE_TOOL_CAPABILITY_QUALIFICATION_PROVIDER_ID

    def supports(self, request: CapabilityQualificationRequest) -> bool:
        if request.strategy_id != _MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID:
            return False
        acquisition = request.acquisition_result
        if acquisition.outcome is not CapabilityAcquisitionOutcome.SUCCEEDED:
            return False
        evidence = acquisition.evidence
        if evidence is None:
            return False
        if not evidence.domain_handoff_reference:
            return False
        if evidence.artifact_reference is not None:
            return False
        if evidence.evidence_ref is not None:
            return False
        return True

    def qualify(
        self,
        request: CapabilityQualificationRequest,
    ) -> CapabilityQualificationResult:
        started_at = datetime.now(tz=UTC)
        if request.strategy_id != _MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID:
            return _terminal(
                request=request,
                outcome=CapabilityQualificationOutcome.NOT_SUPPORTED,
                reason_code=CapabilityQualificationReasonCode.PROVIDER_NOT_SUPPORTED,
                started_at=started_at,
            )

        acquisition = request.acquisition_result
        evidence = acquisition.evidence
        if evidence is None or not evidence.domain_handoff_reference:
            return _terminal(
                request=request,
                outcome=CapabilityQualificationOutcome.REJECTED,
                reason_code=CapabilityQualificationReasonCode.PROVIDER_REJECTED,
                started_at=started_at,
                reason_detail="missing domain handoff evidence",
            )
        domain_handoff_reference = evidence.domain_handoff_reference

        try:
            ctx = self._context_resolver.resolve_for_qualification(
                acquisition_request_id=request.acquisition_request_id,
                domain_handoff_reference=domain_handoff_reference,
                strategy_id=request.strategy_id,
            )
        except MarketplaceQualifiedToolStageContextResolverNotSupportedError:
            return _terminal(
                request=request,
                outcome=CapabilityQualificationOutcome.NOT_SUPPORTED,
                reason_code=CapabilityQualificationReasonCode.PROVIDER_NOT_SUPPORTED,
                started_at=started_at,
            )
        except MarketplaceQualifiedToolStageContextNotFoundError:
            return _terminal(
                request=request,
                outcome=CapabilityQualificationOutcome.UNAVAILABLE,
                reason_code=CapabilityQualificationReasonCode.PROVIDER_UNAVAILABLE,
                started_at=started_at,
                reason_detail="stage context association missing",
            )
        except MarketplaceQualifiedToolStageContextResolverUnavailableError:
            return _terminal(
                request=request,
                outcome=CapabilityQualificationOutcome.UNAVAILABLE,
                reason_code=CapabilityQualificationReasonCode.PROVIDER_UNAVAILABLE,
                started_at=started_at,
            )
        except (
            MarketplaceQualifiedToolStageContextResolverConflictError,
            MarketplaceQualifiedToolStageContextResolverIntegrityError,
        ):
            return _terminal(
                request=request,
                outcome=CapabilityQualificationOutcome.CONFLICT,
                reason_code=CapabilityQualificationReasonCode.EVIDENCE_INCONSISTENT,
                started_at=started_at,
            )

        try:
            stage = self._stage_repository.get(
                tenant_id=ctx.tenant_id,
                handoff_id=ctx.handoff_id,
            )
        except MarketplaceQualifiedToolStageUnavailableError:
            return _terminal(
                request=request,
                outcome=CapabilityQualificationOutcome.UNAVAILABLE,
                reason_code=CapabilityQualificationReasonCode.PROVIDER_UNAVAILABLE,
                started_at=started_at,
            )

        if stage is None:
            return _terminal(
                request=request,
                outcome=CapabilityQualificationOutcome.UNAVAILABLE,
                reason_code=CapabilityQualificationReasonCode.PROVIDER_UNAVAILABLE,
                started_at=started_at,
                reason_detail="staged tool release missing",
            )

        if stage.handoff_id != ctx.handoff_id or stage.tenant_id != ctx.tenant_id:
            return _terminal(
                request=request,
                outcome=CapabilityQualificationOutcome.CONFLICT,
                reason_code=CapabilityQualificationReasonCode.EVIDENCE_INCONSISTENT,
                started_at=started_at,
            )

        if stage.consumer_target is not CapabilityHandoffConsumerTarget.TOOL_DOMAIN:
            return _terminal(
                request=request,
                outcome=CapabilityQualificationOutcome.FAILED,
                reason_code=CapabilityQualificationReasonCode.EVIDENCE_INCONSISTENT,
                started_at=started_at,
            )

        if stage.selected_release.discovery.kind is not CapabilityKind.TOOL:
            return _terminal(
                request=request,
                outcome=CapabilityQualificationOutcome.FAILED,
                reason_code=CapabilityQualificationReasonCode.EVIDENCE_INCONSISTENT,
                started_at=started_at,
            )

        completed_at = datetime.now(tz=UTC)
        return CapabilityQualificationResult(
            qualification_request_id=request.qualification_request_id,
            acquisition_request_id=request.acquisition_request_id,
            gap_id=request.gap_id,
            strategy_id=request.strategy_id,
            provider_id=self.provider_id,
            outcome=CapabilityQualificationOutcome.QUALIFIED,
            reason_code=CapabilityQualificationReasonCode.NONE,
            started_at=started_at,
            completed_at=completed_at,
            evidence=CapabilityQualificationEvidence(
                provider_id=self.provider_id,
                qualification_request_id=request.qualification_request_id,
                acquisition_request_id=request.acquisition_request_id,
                acquisition_strategy_id=request.strategy_id,
                gap_id=request.gap_id,
                domain_handoff_reference=domain_handoff_reference,
            ),
            correlation_id=request.correlation_id,
            causation_id=request.causation_id,
        )


def _terminal(
    *,
    request: CapabilityQualificationRequest,
    outcome: CapabilityQualificationOutcome,
    reason_code: CapabilityQualificationReasonCode,
    started_at: datetime,
    reason_detail: str = "",
) -> CapabilityQualificationResult:
    completed_at = datetime.now(tz=UTC)
    return CapabilityQualificationResult(
        qualification_request_id=request.qualification_request_id,
        acquisition_request_id=request.acquisition_request_id,
        gap_id=request.gap_id,
        strategy_id=request.strategy_id,
        provider_id=MARKETPLACE_TOOL_CAPABILITY_QUALIFICATION_PROVIDER_ID,
        outcome=outcome,
        reason_code=reason_code,
        started_at=started_at,
        completed_at=completed_at,
        reason_detail=reason_detail,
        correlation_id=request.correlation_id,
        causation_id=request.causation_id,
    )


__all__ = [
    "MARKETPLACE_TOOL_CAPABILITY_QUALIFICATION_PROVIDER_ID",
    "MarketplaceToolCapabilityQualificationProvider",
]

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Capability acquisition coordination — dispatch only (UCA-3)."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import ValidationError

from intergrax.capability_acquisition.acquisition_registry import (
    CapabilityAcquisitionStrategyRegistry,
    descriptor_for_strategy,
)
from intergrax.capability_acquisition.default_strategy_selection_policy import (
    DefaultCapabilityAcquisitionStrategySelectionPolicy,
)
from intergrax.contracts.capability_acquisition.acquisition_authorization import (
    CapabilityAcquisitionAuthorizationOutcome,
    CapabilityAcquisitionAuthorizationPort,
    CapabilityAcquisitionAuthorizationResult,
)
from intergrax.contracts.capability_acquisition.acquisition_evidence import (
    CapabilityAcquisitionEvidence,
)
from intergrax.contracts.capability_acquisition.acquisition_outcome import (
    CapabilityAcquisitionOutcome,
)
from intergrax.contracts.capability_acquisition.acquisition_reason_code import (
    CapabilityAcquisitionReasonCode,
)
from intergrax.contracts.capability_acquisition.acquisition_request import (
    CapabilityAcquisitionRequest,
)
from intergrax.contracts.capability_acquisition.acquisition_result import (
    CapabilityAcquisitionResult,
)
from intergrax.contracts.capability_acquisition.acquisition_strategy import (
    CapabilityAcquisitionStrategy,
)
from intergrax.contracts.capability_acquisition.errors import (
    CapabilityAcquisitionIntegrityError,
)
from intergrax.contracts.capability_acquisition.strategy_selection import (
    CapabilityAcquisitionGovernanceContext,
    CapabilityAcquisitionStrategySelection,
    CapabilityAcquisitionStrategySelectionOutcome,
    CapabilityAcquisitionStrategySelectionPolicy,
)


class CapabilityAcquisitionService:
    """Validate request, govern, select strategy, dispatch, validate typed result."""

    def __init__(
        self,
        strategies: tuple[CapabilityAcquisitionStrategy, ...],
        *,
        authorization: CapabilityAcquisitionAuthorizationPort,
        selection_policy: CapabilityAcquisitionStrategySelectionPolicy | None = None,
    ) -> None:
        self._registry = CapabilityAcquisitionStrategyRegistry(strategies)
        self._authorization = authorization
        self._selection_policy = (
            selection_policy or DefaultCapabilityAcquisitionStrategySelectionPolicy()
        )

    def acquire(
        self, request: CapabilityAcquisitionRequest
    ) -> CapabilityAcquisitionResult:
        started_at = datetime.now(tz=UTC)
        try:
            eligible = self._registry.eligible_strategies(request)
        except CapabilityAcquisitionIntegrityError as exc:
            return _terminal_result(
                request=request,
                outcome=CapabilityAcquisitionOutcome.FAILED,
                reason_code=CapabilityAcquisitionReasonCode.STRATEGY_METADATA_INCONSISTENT,
                started_at=started_at,
                reason_detail=str(exc),
            )

        descriptors = tuple(descriptor_for_strategy(strategy) for strategy in eligible)
        try:
            auth = CapabilityAcquisitionAuthorizationResult.model_validate(
                self._authorization.authorize(request, descriptors).model_dump(),
            )
        except ValidationError as exc:
            return _terminal_result(
                request=request,
                outcome=CapabilityAcquisitionOutcome.FAILED,
                reason_code=CapabilityAcquisitionReasonCode.GOVERNANCE_BLOCKED,
                started_at=started_at,
                reason_detail=str(exc),
            )
        if auth.outcome is CapabilityAcquisitionAuthorizationOutcome.BLOCKED:
            return _terminal_result(
                request=request,
                outcome=CapabilityAcquisitionOutcome.BLOCKED,
                reason_code=CapabilityAcquisitionReasonCode.GOVERNANCE_BLOCKED,
                started_at=started_at,
                reason_detail=auth.reason_detail or "acquisition governance blocked",
            )
        if auth.outcome is CapabilityAcquisitionAuthorizationOutcome.REQUIRES_HITL:
            return _terminal_result(
                request=request,
                outcome=CapabilityAcquisitionOutcome.REQUIRES_HITL,
                reason_code=CapabilityAcquisitionReasonCode.HUMAN_APPROVAL_REQUIRED,
                started_at=started_at,
                reason_detail=auth.reason_detail or "human approval required",
            )
        if auth.outcome is not CapabilityAcquisitionAuthorizationOutcome.PERMITTED:
            return _terminal_result(
                request=request,
                outcome=CapabilityAcquisitionOutcome.FAILED,
                reason_code=CapabilityAcquisitionReasonCode.GOVERNANCE_BLOCKED,
                started_at=started_at,
                reason_detail="unrecognized acquisition authorization outcome",
            )

        governance_context = CapabilityAcquisitionGovernanceContext(
            authorization_decision_id=auth.decision_id or None,
        )
        try:
            raw_selection = self._selection_policy.select(
                request=request,
                candidates=descriptors,
                governance_context=governance_context,
            )
            selection = CapabilityAcquisitionStrategySelection.model_validate(
                raw_selection.model_dump(),
            )
        except ValidationError as exc:
            return _terminal_result(
                request=request,
                outcome=CapabilityAcquisitionOutcome.FAILED,
                reason_code=CapabilityAcquisitionReasonCode.EVIDENCE_INCONSISTENT,
                started_at=started_at,
                reason_detail=str(exc),
            )
        if (
            selection.outcome
            is CapabilityAcquisitionStrategySelectionOutcome.NO_STRATEGY
        ):
            return _terminal_result(
                request=request,
                outcome=CapabilityAcquisitionOutcome.NO_STRATEGY,
                reason_code=selection.reason_code,
                started_at=started_at,
                reason_detail=selection.reason_detail,
            )
        if selection.outcome is CapabilityAcquisitionStrategySelectionOutcome.CONFLICT:
            return _terminal_result(
                request=request,
                outcome=CapabilityAcquisitionOutcome.CONFLICT,
                reason_code=selection.reason_code,
                started_at=started_at,
                reason_detail=selection.reason_detail,
            )
        if selection.outcome is CapabilityAcquisitionStrategySelectionOutcome.BLOCKED:
            return _terminal_result(
                request=request,
                outcome=CapabilityAcquisitionOutcome.BLOCKED,
                reason_code=selection.reason_code,
                started_at=started_at,
                reason_detail=selection.reason_detail,
            )
        if (
            selection.outcome
            is CapabilityAcquisitionStrategySelectionOutcome.REQUIRES_HITL
        ):
            return _terminal_result(
                request=request,
                outcome=CapabilityAcquisitionOutcome.REQUIRES_HITL,
                reason_code=selection.reason_code,
                started_at=started_at,
                reason_detail=selection.reason_detail,
            )

        if (
            selection.outcome
            is not CapabilityAcquisitionStrategySelectionOutcome.SELECTED
        ):
            return _terminal_result(
                request=request,
                outcome=CapabilityAcquisitionOutcome.FAILED,
                reason_code=CapabilityAcquisitionReasonCode.EVIDENCE_INCONSISTENT,
                started_at=started_at,
                reason_detail="selection ended without a selected strategy",
            )

        strategy_id = selection.strategy_id
        if strategy_id is None:
            return _terminal_result(
                request=request,
                outcome=CapabilityAcquisitionOutcome.FAILED,
                reason_code=CapabilityAcquisitionReasonCode.EVIDENCE_INCONSISTENT,
                started_at=started_at,
                reason_detail="SELECTED outcome missing strategy_id",
            )

        strategy = self._registry.get(strategy_id)
        if strategy is None or strategy not in eligible:
            return _terminal_result(
                request=request,
                outcome=CapabilityAcquisitionOutcome.FAILED,
                reason_code=CapabilityAcquisitionReasonCode.EVIDENCE_INCONSISTENT,
                started_at=started_at,
                reason_detail="selection referenced unknown or ineligible strategy",
            )

        try:
            result = strategy.acquire(request)
        except ValidationError as exc:
            return _terminal_result(
                request=request,
                outcome=CapabilityAcquisitionOutcome.FAILED,
                reason_code=CapabilityAcquisitionReasonCode.EVIDENCE_INCONSISTENT,
                started_at=started_at,
                strategy_id=strategy.strategy_id,
                reason_detail=str(exc),
            )
        except CapabilityAcquisitionIntegrityError as exc:
            return _terminal_result(
                request=request,
                outcome=CapabilityAcquisitionOutcome.FAILED,
                reason_code=CapabilityAcquisitionReasonCode.EVIDENCE_INCONSISTENT,
                started_at=started_at,
                strategy_id=strategy.strategy_id,
                reason_detail=str(exc),
            )

        _assert_result_matches_request(
            request,
            result,
            strategy_id=strategy.strategy_id,
        )
        return result


def _assert_result_matches_request(
    request: CapabilityAcquisitionRequest,
    result: CapabilityAcquisitionResult,
    *,
    strategy_id: str,
) -> None:
    if result.request_id != request.request_id:
        raise CapabilityAcquisitionIntegrityError("result request_id mismatch")
    if result.gap_id != request.capability_gap.gap_id:
        raise CapabilityAcquisitionIntegrityError("result gap_id mismatch")
    if result.strategy_id != strategy_id:
        raise CapabilityAcquisitionIntegrityError("result strategy_id mismatch")
    if result.correlation_id != request.correlation_id:
        raise CapabilityAcquisitionIntegrityError("result correlation_id mismatch")
    if result.causation_id != request.causation_id:
        raise CapabilityAcquisitionIntegrityError("result causation_id mismatch")


def _terminal_result(
    *,
    request: CapabilityAcquisitionRequest,
    outcome: CapabilityAcquisitionOutcome,
    reason_code: CapabilityAcquisitionReasonCode,
    started_at: datetime,
    strategy_id: str | None = None,
    reason_detail: str = "",
    evidence: CapabilityAcquisitionEvidence | None = None,
) -> CapabilityAcquisitionResult:
    completed_at = datetime.now(tz=UTC)
    return CapabilityAcquisitionResult(
        request_id=request.request_id,
        gap_id=request.capability_gap.gap_id,
        strategy_id=strategy_id,
        outcome=outcome,
        reason_code=reason_code,
        started_at=started_at,
        completed_at=completed_at,
        evidence=evidence,
        reason_detail=reason_detail,
        correlation_id=request.correlation_id,
        causation_id=request.causation_id,
    )


__all__ = ["CapabilityAcquisitionService"]

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Capability qualification coordination — verification dispatch only (UCA-4)."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import ValidationError

from intergrax.capability_qualification.default_lifecycle_policy import (
    DefaultCapabilityQualificationLifecyclePolicy,
)
from intergrax.capability_qualification.default_provider_selection_policy import (
    DefaultCapabilityQualificationProviderSelectionPolicy,
)
from intergrax.capability_qualification.qualification_registry import (
    CapabilityQualificationProviderRegistry,
    descriptor_for_provider,
)
from intergrax.contracts.capability_qualification.audit_record import (
    build_qualification_audit_record,
)
from intergrax.contracts.capability_qualification.errors import (
    CapabilityQualificationIntegrityError,
)
from intergrax.contracts.capability_qualification.lifecycle_decision import (
    CapabilityQualificationLifecycleDecision,
    CapabilityQualificationLifecyclePolicy,
)
from intergrax.contracts.capability_qualification.provider import (
    CapabilityQualificationProvider,
)
from intergrax.contracts.capability_qualification.provider_selection import (
    CapabilityQualificationProviderSelection,
    CapabilityQualificationProviderSelectionOutcome,
    CapabilityQualificationProviderSelectionPolicy,
)
from intergrax.contracts.capability_qualification.qualification_decision import (
    CapabilityQualificationDecision,
)
from intergrax.contracts.capability_qualification.qualification_evidence import (
    CapabilityQualificationEvidence,
)
from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)
from intergrax.contracts.capability_qualification.qualification_reason_code import (
    CapabilityQualificationReasonCode,
)
from intergrax.contracts.capability_qualification.qualification_integrity import (
    validate_qualification_subject_binding,
)
from intergrax.contracts.capability_qualification.qualification_request import (
    CapabilityQualificationRequest,
)
from intergrax.contracts.capability_qualification.qualification_result import (
    CapabilityQualificationResult,
)


class CapabilityQualificationService:
    """Validate request, select provider, dispatch, validate result, lifecycle decide."""

    def __init__(
        self,
        providers: tuple[CapabilityQualificationProvider, ...],
        *,
        selection_policy: CapabilityQualificationProviderSelectionPolicy | None = None,
        lifecycle_policy: CapabilityQualificationLifecyclePolicy | None = None,
    ) -> None:
        self._registry = CapabilityQualificationProviderRegistry(providers)
        self._selection_policy = (
            selection_policy or DefaultCapabilityQualificationProviderSelectionPolicy()
        )
        self._lifecycle_policy = (
            lifecycle_policy or DefaultCapabilityQualificationLifecyclePolicy()
        )

    def qualify(
        self, request: CapabilityQualificationRequest
    ) -> CapabilityQualificationDecision:
        started_at = datetime.now(tz=UTC)
        eligible = self._registry.eligible_providers(request)
        descriptors = tuple(descriptor_for_provider(provider) for provider in eligible)
        try:
            raw_selection = self._selection_policy.select(
                request=request,
                candidates=descriptors,
            )
            selection = CapabilityQualificationProviderSelection.model_validate(
                raw_selection.model_dump(),
            )
        except ValidationError as exc:
            qualification_result = _terminal_result(
                request=request,
                outcome=CapabilityQualificationOutcome.FAILED,
                reason_code=CapabilityQualificationReasonCode.EVIDENCE_INCONSISTENT,
                started_at=started_at,
                reason_detail=str(exc),
            )
            return _build_decision(
                request=request,
                qualification_result=qualification_result,
                lifecycle_policy=self._lifecycle_policy,
            )

        if (
            selection.outcome
            is CapabilityQualificationProviderSelectionOutcome.NO_PROVIDER
        ):
            qualification_result = _terminal_result(
                request=request,
                outcome=CapabilityQualificationOutcome.NO_PROVIDER,
                reason_code=selection.reason_code,
                started_at=started_at,
                reason_detail=selection.reason_detail,
            )
            return _build_decision(
                request=request,
                qualification_result=qualification_result,
                lifecycle_policy=self._lifecycle_policy,
            )
        if (
            selection.outcome
            is CapabilityQualificationProviderSelectionOutcome.CONFLICT
        ):
            qualification_result = _terminal_result(
                request=request,
                outcome=CapabilityQualificationOutcome.CONFLICT,
                reason_code=selection.reason_code,
                started_at=started_at,
                reason_detail=selection.reason_detail,
            )
            return _build_decision(
                request=request,
                qualification_result=qualification_result,
                lifecycle_policy=self._lifecycle_policy,
            )
        if (
            selection.outcome
            is not CapabilityQualificationProviderSelectionOutcome.SELECTED
        ):
            qualification_result = _terminal_result(
                request=request,
                outcome=CapabilityQualificationOutcome.FAILED,
                reason_code=CapabilityQualificationReasonCode.EVIDENCE_INCONSISTENT,
                started_at=started_at,
                reason_detail="selection ended without a selected provider",
            )
            return _build_decision(
                request=request,
                qualification_result=qualification_result,
                lifecycle_policy=self._lifecycle_policy,
            )

        provider_id = selection.provider_id
        if provider_id is None:
            qualification_result = _terminal_result(
                request=request,
                outcome=CapabilityQualificationOutcome.FAILED,
                reason_code=CapabilityQualificationReasonCode.EVIDENCE_INCONSISTENT,
                started_at=started_at,
                reason_detail="SELECTED outcome missing provider_id",
            )
            return _build_decision(
                request=request,
                qualification_result=qualification_result,
                lifecycle_policy=self._lifecycle_policy,
            )

        provider = self._registry.get(provider_id)
        if provider is None or provider not in eligible:
            qualification_result = _terminal_result(
                request=request,
                outcome=CapabilityQualificationOutcome.FAILED,
                reason_code=CapabilityQualificationReasonCode.EVIDENCE_INCONSISTENT,
                started_at=started_at,
                reason_detail="selection referenced unknown or ineligible provider",
            )
            return _build_decision(
                request=request,
                qualification_result=qualification_result,
                lifecycle_policy=self._lifecycle_policy,
            )

        try:
            result = provider.qualify(request)
        except ValidationError as exc:
            qualification_result = _terminal_result(
                request=request,
                outcome=CapabilityQualificationOutcome.FAILED,
                reason_code=CapabilityQualificationReasonCode.EVIDENCE_INCONSISTENT,
                started_at=started_at,
                provider_id=provider.provider_id,
                reason_detail=str(exc),
            )
            return _build_decision(
                request=request,
                qualification_result=qualification_result,
                lifecycle_policy=self._lifecycle_policy,
            )
        except CapabilityQualificationIntegrityError as exc:
            qualification_result = _terminal_result(
                request=request,
                outcome=CapabilityQualificationOutcome.FAILED,
                reason_code=CapabilityQualificationReasonCode.EVIDENCE_INCONSISTENT,
                started_at=started_at,
                provider_id=provider.provider_id,
                reason_detail=str(exc),
            )
            return _build_decision(
                request=request,
                qualification_result=qualification_result,
                lifecycle_policy=self._lifecycle_policy,
            )

        _assert_result_matches_request(
            request,
            result,
            provider_id=provider.provider_id,
        )
        return _build_decision(
            request=request,
            qualification_result=result,
            lifecycle_policy=self._lifecycle_policy,
        )


def _assert_result_matches_request(
    request: CapabilityQualificationRequest,
    result: CapabilityQualificationResult,
    *,
    provider_id: str,
) -> None:
    if result.qualification_request_id != request.qualification_request_id:
        raise CapabilityQualificationIntegrityError(
            "result qualification_request_id mismatch",
        )
    if result.acquisition_request_id != request.acquisition_request_id:
        raise CapabilityQualificationIntegrityError(
            "result acquisition_request_id mismatch",
        )
    if result.gap_id != request.gap_id:
        raise CapabilityQualificationIntegrityError("result gap_id mismatch")
    if result.strategy_id != request.strategy_id:
        raise CapabilityQualificationIntegrityError("result strategy_id mismatch")
    if result.provider_id != provider_id:
        raise CapabilityQualificationIntegrityError("result provider_id mismatch")
    if result.correlation_id != request.correlation_id:
        raise CapabilityQualificationIntegrityError("result correlation_id mismatch")
    if result.causation_id != request.causation_id:
        raise CapabilityQualificationIntegrityError("result causation_id mismatch")
    if result.evidence is not None:
        acquisition_evidence = request.acquisition_result.evidence
        if acquisition_evidence is None:
            raise CapabilityQualificationIntegrityError(
                "qualification evidence requires acquisition evidence subject",
            )
        try:
            validate_qualification_subject_binding(
                acquisition_evidence,
                result.evidence,
            )
        except ValueError as exc:
            raise CapabilityQualificationIntegrityError(str(exc)) from exc


def _terminal_result(
    *,
    request: CapabilityQualificationRequest,
    outcome: CapabilityQualificationOutcome,
    reason_code: CapabilityQualificationReasonCode,
    started_at: datetime,
    provider_id: str | None = None,
    reason_detail: str = "",
    evidence: CapabilityQualificationEvidence | None = None,
) -> CapabilityQualificationResult:
    completed_at = datetime.now(tz=UTC)
    return CapabilityQualificationResult(
        qualification_request_id=request.qualification_request_id,
        acquisition_request_id=request.acquisition_request_id,
        gap_id=request.gap_id,
        strategy_id=request.strategy_id,
        provider_id=provider_id,
        outcome=outcome,
        reason_code=reason_code,
        started_at=started_at,
        completed_at=completed_at,
        evidence=evidence,
        reason_detail=reason_detail,
        correlation_id=request.correlation_id,
        causation_id=request.causation_id,
    )


def _build_decision(
    *,
    request: CapabilityQualificationRequest,
    qualification_result: CapabilityQualificationResult,
    lifecycle_policy: CapabilityQualificationLifecyclePolicy,
) -> CapabilityQualificationDecision:
    lifecycle_decision = CapabilityQualificationLifecycleDecision.model_validate(
        lifecycle_policy.decide(
            request=request,
            qualification_result=qualification_result,
        ).model_dump(),
    )
    audit_record = build_qualification_audit_record(
        qualification_request_id=request.qualification_request_id,
        acquisition_request_id=request.acquisition_request_id,
        acquisition_strategy_id=request.strategy_id,
        gap_id=request.gap_id,
        qualification_provider_id=qualification_result.provider_id,
        qualification_outcome=qualification_result.outcome,
        lifecycle_decision=lifecycle_decision,
        correlation_id=request.correlation_id,
        causation_id=request.causation_id,
    )
    return CapabilityQualificationDecision(
        qualification_result=qualification_result,
        lifecycle_decision=lifecycle_decision,
        audit_record=audit_record,
    )


__all__ = ["CapabilityQualificationService"]

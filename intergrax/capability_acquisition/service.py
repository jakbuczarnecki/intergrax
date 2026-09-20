# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Capability realization coordination — dispatch only (UCA-2)."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import ValidationError

from intergrax.capability_acquisition.registry import (
    CapabilityRealizationProviderRegistry,
)
from intergrax.contracts.capability_acquisition.errors import (
    CapabilityRealizationIntegrityError,
)
from intergrax.contracts.capability_acquisition.evidence import (
    CapabilityRealizationEvidence,
)
from intergrax.contracts.capability_acquisition.outcome import (
    CapabilityRealizationOutcome,
)
from intergrax.contracts.capability_acquisition.provider import (
    CapabilityRealizationProvider,
)
from intergrax.contracts.capability_acquisition.reason_code import (
    CapabilityRealizationReasonCode,
)
from intergrax.contracts.capability_acquisition.request import (
    CapabilityRealizationRequest,
)
from intergrax.contracts.capability_acquisition.result import (
    CapabilityRealizationResult,
)


class CapabilityRealizationService:
    """Validate request, resolve provider, dispatch, return typed result."""

    def __init__(
        self,
        providers: tuple[CapabilityRealizationProvider, ...],
    ) -> None:
        self._registry = CapabilityRealizationProviderRegistry(providers)

    def realize(
        self, request: CapabilityRealizationRequest
    ) -> CapabilityRealizationResult:
        started_at = datetime.now(tz=UTC)
        eligible = self._registry.eligible_providers(request)
        if not eligible:
            return _terminal_result(
                request=request,
                outcome=CapabilityRealizationOutcome.NOT_SUPPORTED,
                reason_code=CapabilityRealizationReasonCode.NO_PROVIDER,
                started_at=started_at,
                reason_detail="no realization provider for capability kind",
            )
        if len(eligible) > 1:
            provider_ids = ", ".join(sorted(p.provider_id for p in eligible))
            return _terminal_result(
                request=request,
                outcome=CapabilityRealizationOutcome.CONFLICT,
                reason_code=CapabilityRealizationReasonCode.AMBIGUOUS_PROVIDER,
                started_at=started_at,
                reason_detail=f"ambiguous providers: {provider_ids}",
            )

        assert len(eligible) == 1
        provider = eligible[0]
        try:
            result = provider.realize(request)
        except ValidationError as exc:
            return _terminal_result(
                request=request,
                outcome=CapabilityRealizationOutcome.FAILED,
                reason_code=CapabilityRealizationReasonCode.EVIDENCE_INCONSISTENT,
                started_at=started_at,
                provider_id=provider.provider_id,
                reason_detail=str(exc),
            )
        except CapabilityRealizationIntegrityError as exc:
            return _terminal_result(
                request=request,
                outcome=CapabilityRealizationOutcome.FAILED,
                reason_code=CapabilityRealizationReasonCode.EVIDENCE_INCONSISTENT,
                started_at=started_at,
                provider_id=provider.provider_id,
                reason_detail=str(exc),
            )

        _assert_result_matches_request(
            request, result, provider_id=provider.provider_id
        )
        return result


def _assert_result_matches_request(
    request: CapabilityRealizationRequest,
    result: CapabilityRealizationResult,
    *,
    provider_id: str,
) -> None:
    if result.request_id != request.request_id:
        raise CapabilityRealizationIntegrityError("result request_id mismatch")
    if result.realization_need_id != request.realization_need.realization_need_id:
        raise CapabilityRealizationIntegrityError("result realization_need_id mismatch")
    if (
        result.capability_identity.sort_key
        != request.realization_need.capability_identity.sort_key
    ):
        raise CapabilityRealizationIntegrityError("result capability_identity mismatch")
    if result.provider_id is not None and result.provider_id != provider_id:
        raise CapabilityRealizationIntegrityError("result provider_id mismatch")


def _terminal_result(
    *,
    request: CapabilityRealizationRequest,
    outcome: CapabilityRealizationOutcome,
    reason_code: CapabilityRealizationReasonCode,
    started_at: datetime,
    provider_id: str | None = None,
    reason_detail: str = "",
    evidence: CapabilityRealizationEvidence | None = None,
) -> CapabilityRealizationResult:
    completed_at = datetime.now(tz=UTC)
    return CapabilityRealizationResult(
        request_id=request.request_id,
        realization_need_id=request.realization_need.realization_need_id,
        provider_id=provider_id,
        outcome=outcome,
        reason_code=reason_code,
        capability_identity=request.realization_need.capability_identity,
        started_at=started_at,
        completed_at=completed_at,
        evidence=evidence,
        reason_detail=reason_detail,
    )


__all__ = ["CapabilityRealizationService"]

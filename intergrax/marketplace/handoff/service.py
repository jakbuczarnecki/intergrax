# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Marketplace lifecycle handoff orchestration — validate, route, delegate (ME-RB4)."""

from __future__ import annotations

from intergrax.contracts.marketplace.lifecycle_handoff_outcome import (
    MarketplaceLifecycleHandoffOutcome,
    MarketplaceLifecycleHandoffReasonCode,
    MarketplaceLifecycleHandoffStatus,
)
from intergrax.contracts.marketplace.lifecycle_handoff_request import (
    MarketplaceLifecycleHandoffRequest,
    selection_identity_key,
)
from intergrax.marketplace.handoff.errors import (
    MarketplaceLifecycleHandlerError,
    MarketplaceLifecycleHandoffValidationError,
)
from intergrax.marketplace.handoff.resolver import LifecycleHandoffResolver


class MarketplaceLifecycleHandoffService:
    """Routes typed handoff requests to registered handlers — no lifecycle mutation."""

    def __init__(self, resolver: LifecycleHandoffResolver) -> None:
        self._resolver = resolver

    def handoff(
        self,
        request: MarketplaceLifecycleHandoffRequest,
    ) -> MarketplaceLifecycleHandoffOutcome:
        kind = request.selection.capability.identity.kind
        handler = self._resolver.handler_for(kind)
        if handler is None:
            return MarketplaceLifecycleHandoffOutcome(
                request_id=request.request_id,
                status=MarketplaceLifecycleHandoffStatus.REJECTED,
                domain_authority_id="marketplace_handoff",
                reason_code=MarketplaceLifecycleHandoffReasonCode.HANDLER_MISSING,
                reason_detail=f"no handler registered for capability kind {kind.value}",
            )

        try:
            self._validate_identity_alignment(request)
        except MarketplaceLifecycleHandoffValidationError as exc:
            return MarketplaceLifecycleHandoffOutcome(
                request_id=request.request_id,
                status=MarketplaceLifecycleHandoffStatus.REJECTED,
                domain_authority_id=handler.domain_authority_id,
                reason_code=MarketplaceLifecycleHandoffReasonCode.IDENTITY_MISMATCH,
                reason_detail=str(exc),
            )

        try:
            return handler.handoff(request)
        except MarketplaceLifecycleHandlerError as exc:
            return MarketplaceLifecycleHandoffOutcome(
                request_id=request.request_id,
                status=MarketplaceLifecycleHandoffStatus.REJECTED,
                domain_authority_id=handler.domain_authority_id,
                reason_code=MarketplaceLifecycleHandoffReasonCode.HANDLER_FAILED,
                reason_detail=str(exc),
            )

    @staticmethod
    def _validate_identity_alignment(request: MarketplaceLifecycleHandoffRequest) -> None:
        selected = selection_identity_key(request.selection)
        payload_key = request.domain_payload.identity_key()
        if selected != payload_key:
            raise MarketplaceLifecycleHandoffValidationError(
                "selection capability identity does not match domain payload identity",
            )


__all__ = ["MarketplaceLifecycleHandoffService"]

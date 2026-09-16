# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Marketplace lifecycle handoff orchestration — validate, route, delegate (ME-RB4)."""

from __future__ import annotations

from intergrax.contracts.marketplace.lifecycle_handoff_outcome import (
    MarketplaceLifecycleHandoffOutcome,
    MarketplaceLifecycleHandoffReasonCode,
    MarketplaceLifecycleHandoffStatus,
)
from intergrax.contracts.capability_catalog.release_identity import CapabilityReleaseIdentity
from intergrax.contracts.marketplace.lifecycle_handoff_request import (
    MarketplaceLifecycleHandoffRequest,
    selection_identity_key,
)
from intergrax.marketplace.handoff.errors import (
    MarketplaceLifecycleHandlerError,
    MarketplaceLifecycleHandoffValidationError,
)
from intergrax.contracts.marketplace.diagnostics import (
    MarketplaceDiagnosticEvent,
    MarketplaceDiagnosticEventKind,
    MarketplaceDiagnosticOutcome,
    MarketplaceDiagnosticObserver,
    MarketplaceObserverFailurePolicy,
    MarketplacePipelineStage,
)
from intergrax.contracts.marketplace.diagnostics import MarketplaceObservationContext
from intergrax.marketplace.diagnostics import emit_to_observer
from intergrax.marketplace.handoff.resolver import LifecycleHandoffResolver


class MarketplaceLifecycleHandoffService:
    """Routes typed handoff requests to registered handlers — no lifecycle mutation."""

    def __init__(
        self,
        resolver: LifecycleHandoffResolver,
        *,
        diagnostic_observer: MarketplaceDiagnosticObserver | None = None,
        observer_failure_policy: MarketplaceObserverFailurePolicy = (
            MarketplaceObserverFailurePolicy.BEST_EFFORT
        ),
    ) -> None:
        self._resolver = resolver
        self._diagnostic_observer = diagnostic_observer
        self._observer_failure_policy = observer_failure_policy

    def handoff(
        self,
        request: MarketplaceLifecycleHandoffRequest,
    ) -> MarketplaceLifecycleHandoffOutcome:
        kind = request.selection.capability.identity.kind
        handler = self._resolver.handler_for(kind)
        correlation = MarketplaceObservationContext(
            discovery_correlation_id=request.correlation_id or request.request_id,
            query_correlation_id=request.request_id,
        )
        selected_release = request.selection.selected_release
        if handler is None:
            outcome = MarketplaceLifecycleHandoffOutcome(
                request_id=request.request_id,
                status=MarketplaceLifecycleHandoffStatus.REJECTED,
                domain_authority_id="marketplace_handoff",
                reason_code=MarketplaceLifecycleHandoffReasonCode.HANDLER_MISSING,
                reason_detail=f"no handler registered for capability kind {kind.value}",
            )
            self._emit_handoff_diagnostic(correlation, selected_release, outcome)
            return outcome

        try:
            self._validate_identity_alignment(request)
        except MarketplaceLifecycleHandoffValidationError as exc:
            outcome = MarketplaceLifecycleHandoffOutcome(
                request_id=request.request_id,
                status=MarketplaceLifecycleHandoffStatus.REJECTED,
                domain_authority_id=handler.domain_authority_id,
                reason_code=MarketplaceLifecycleHandoffReasonCode.IDENTITY_MISMATCH,
                reason_detail=str(exc),
            )
            self._emit_handoff_diagnostic(correlation, selected_release, outcome)
            return outcome

        try:
            outcome = handler.handoff(request)
            self._emit_handoff_diagnostic(correlation, selected_release, outcome)
            return outcome
        except MarketplaceLifecycleHandlerError as exc:
            outcome = MarketplaceLifecycleHandoffOutcome(
                request_id=request.request_id,
                status=MarketplaceLifecycleHandoffStatus.REJECTED,
                domain_authority_id=handler.domain_authority_id,
                reason_code=MarketplaceLifecycleHandoffReasonCode.HANDLER_FAILED,
                reason_detail=str(exc),
            )
            self._emit_handoff_diagnostic(correlation, selected_release, outcome)
            return outcome

    def _emit_handoff_diagnostic(
        self,
        correlation: MarketplaceObservationContext,
        selected_release: CapabilityReleaseIdentity,
        outcome: MarketplaceLifecycleHandoffOutcome,
    ) -> None:
        diagnostic_outcome = (
            MarketplaceDiagnosticOutcome.REJECTED
            if outcome.status is MarketplaceLifecycleHandoffStatus.REJECTED
            else MarketplaceDiagnosticOutcome.SUCCESS
        )
        emit_to_observer(
            self._diagnostic_observer,
            MarketplaceDiagnosticEvent(
                stage=MarketplacePipelineStage.HANDOFF,
                event_kind=MarketplaceDiagnosticEventKind.COMPLETED,
                correlation=correlation,
                selected_release=selected_release,
                handoff_domain_authority_id=outcome.domain_authority_id,
                handoff_reason_code=(
                    outcome.reason_code.value if outcome.reason_code is not None else None
                ),
                handoff_status=outcome.status.value,
                outcome=diagnostic_outcome,
            ),
            failure_policy=self._observer_failure_policy,
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

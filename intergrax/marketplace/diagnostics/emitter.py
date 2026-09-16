# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.contracts.marketplace.diagnostics import (
    MarketplaceDiagnosticEvent,
    MarketplaceDiagnosticObserver,
    MarketplaceObserverEmitError,
    MarketplaceObserverFailurePolicy,
)
from intergrax.marketplace.diagnostics.session import MarketplacePipelineObservationSession


def emit_marketplace_diagnostic(
    session: MarketplacePipelineObservationSession | None,
    event: MarketplaceDiagnosticEvent,
) -> None:
    """Emit one diagnostic event according to the session observer failure policy."""
    if session is None or session.observer is None:
        return
    try:
        session.observer.emit(event)
    except Exception as exc:
        if session.failure_policy is MarketplaceObserverFailurePolicy.STRICT:
            raise MarketplaceObserverEmitError(
                f"marketplace diagnostic observer {session.observer.observer_id!r} failed",
            ) from exc


def emit_to_observer(
    observer: MarketplaceDiagnosticObserver | None,
    event: MarketplaceDiagnosticEvent,
    *,
    failure_policy: MarketplaceObserverFailurePolicy = (
        MarketplaceObserverFailurePolicy.BEST_EFFORT
    ),
) -> None:
    if observer is None:
        return
    session = MarketplacePipelineObservationSession(
        correlation=event.correlation,
        observer=observer,
        failure_policy=failure_policy,
    )
    emit_marketplace_diagnostic(session, event)


__all__ = ["emit_marketplace_diagnostic", "emit_to_observer"]

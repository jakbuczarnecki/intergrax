# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.contracts.marketplace.diagnostics import (
    NOOP_MARKETPLACE_DIAGNOSTIC_OBSERVER_ID,
    MarketplaceDiagnosticEvent,
)


class NoOpMarketplaceDiagnosticObserver:
    """Deterministic zero-side-effect observer for default composition."""

    @property
    def observer_id(self) -> str:
        return NOOP_MARKETPLACE_DIAGNOSTIC_OBSERVER_ID

    def emit(self, event: MarketplaceDiagnosticEvent) -> None:
        return None


__all__ = ["NoOpMarketplaceDiagnosticObserver"]

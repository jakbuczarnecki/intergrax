# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.marketplace.diagnostics import (
    MarketplaceDiagnosticObserver,
    MarketplaceObservationContext,
    MarketplaceObserverFailurePolicy,
)


@dataclass(frozen=True, slots=True)
class MarketplacePipelineObservationSession:
    """Optional observation bundle threaded through marketplace orchestration."""

    correlation: MarketplaceObservationContext
    observer: MarketplaceDiagnosticObserver | None = None
    failure_policy: MarketplaceObserverFailurePolicy = (
        MarketplaceObserverFailurePolicy.BEST_EFFORT
    )

    @classmethod
    def for_discovery(
        cls,
        discovery_correlation_id: str,
        *,
        query_correlation_id: str | None = None,
        observer: MarketplaceDiagnosticObserver | None = None,
        failure_policy: MarketplaceObserverFailurePolicy = (
            MarketplaceObserverFailurePolicy.BEST_EFFORT
        ),
    ) -> MarketplacePipelineObservationSession:
        return cls(
            correlation=MarketplaceObservationContext(
                discovery_correlation_id=require_non_empty_text(
                    discovery_correlation_id,
                    label="discovery_correlation_id",
                ),
                query_correlation_id=query_correlation_id,
            ),
            observer=observer,
            failure_policy=failure_policy,
        )

__all__ = ["MarketplacePipelineObservationSession"]

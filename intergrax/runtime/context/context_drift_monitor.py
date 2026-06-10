# © Artur Czarnecki. All rights reserved.

"""Online context drift monitoring (AUDIT-IDEAL-16.1)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ContextDriftSignal(BaseModel):
    """Observed context assembly metrics for one run."""

    model_config = ConfigDict(extra="forbid")

    token_estimate: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    baseline_token_estimate: int = Field(ge=1)


class ContextDriftReport(BaseModel):
    """Drift evaluation with optional alert."""

    model_config = ConfigDict(extra="forbid")

    drift_ratio: float
    alert: bool
    reasons: list[str] = Field(default_factory=list)


def evaluate_context_drift(
    signal: ContextDriftSignal,
    *,
    alert_threshold: float = 0.35,
) -> ContextDriftReport:
    """Raise alert when context size drifts above baseline by ``alert_threshold``."""
    baseline = max(signal.baseline_token_estimate, 1)
    drift_ratio = (signal.token_estimate - baseline) / baseline
    reasons: list[str] = []
    alert = drift_ratio >= alert_threshold
    if alert:
        reasons.append(
            f"context_tokens={signal.token_estimate} exceeded baseline={baseline} "
            f"by {drift_ratio:.2%} (threshold={alert_threshold:.0%})"
        )
    if signal.chunk_count == 0 and signal.token_estimate > baseline:
        reasons.append("non-zero token estimate with zero retrieved chunks")
        alert = True
    return ContextDriftReport(drift_ratio=drift_ratio, alert=alert, reasons=reasons)

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Resolution execution contracts (ERL — foundation)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.enterprise_reliability.lifecycle import UncertaintyResolutionKind
from intergrax.contracts.enterprise_reliability.plugin_spi import ResolutionStrategyAdvice
from intergrax.contracts.enterprise_reliability.resolution import ResolutionDisposition, ResolutionPlan


class ResolutionExecutionDisposition(StrEnum):
    """Platform outcome after resolution orchestration — no recovery I/O here."""

    ADVICE_APPLIED = "advice_applied"
    DEFERRED_INSUFFICIENT_EVIDENCE = "deferred_insufficient_evidence"
    ESCALATED_BY_POSTURE = "escalated_by_posture"
    SKIPPED_PLUGIN_UNAVAILABLE = "skipped_plugin_unavailable"
    DEFERRED_STRATEGY_ACTION = "deferred_strategy_action"


class ResolutionExecutionError(ValueError):
    """Resolution cannot be applied under plan and evidence rules."""


class ExternalEffectResolutionExecution(BaseModel):
    """One bounded resolution attempt — advice applied or deferred."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    disposition: ResolutionExecutionDisposition
    plan: ResolutionPlan
    applied_advice: ResolutionStrategyAdvice | None = None
    resolution_kind: UncertaintyResolutionKind | None = None
    rationale: str = Field(default="", max_length=512)


__all__ = [
    "ExternalEffectResolutionExecution",
    "ResolutionExecutionDisposition",
    "ResolutionExecutionError",
]

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Compensation orchestration contracts (ERL — foundation)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.contracts.enterprise_reliability.compensation_decision import (
    CompensationDecision,
    CompensationPlatformIntent,
    missing_compensation_strategy_decision,
)
from intergrax.contracts.enterprise_reliability.plugin_spi import CompensationStrategyAdvice
from intergrax.contracts.enterprise_reliability.resolution_decision import (
    ResolutionDecision,
    ResolutionPlatformAction,
)

SCHEMA_COMPENSATION_PLAN_V1: Final = "compensation_plan.v1"


class CompensationDisposition(StrEnum):
    """Platform decision before compensation execution — no recovery I/O here."""

    INVOKE_PLUGIN = "invoke_plugin"
    ESCALATE_REQUIRED = "escalate_required"
    STRATEGY_UNAVAILABLE = "strategy_unavailable"
    DEFER_STRATEGY_ACTION = "defer_strategy_action"
    UNAVAILABLE = "unavailable"


class CompensationPlanningError(ValueError):
    """Compensation plan cannot be derived from resolution and strategy inputs."""


class CompensationPlan(BaseModel):
    """Non-executing compensation intent — plugin selection and advice snapshot only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_COMPENSATION_PLAN_V1
    disposition: CompensationDisposition
    decision: CompensationDecision | None = None
    plugin_id: str | None = None
    advice: CompensationStrategyAdvice | None = None
    rationale: str = Field(default="", max_length=512)

    @model_validator(mode="after")
    def _validate_plugin_fields(self) -> CompensationPlan:
        if self.disposition is CompensationDisposition.INVOKE_PLUGIN:
            if self.decision is None:
                raise ValueError("decision required when disposition is invoke_plugin")
            if self.plugin_id is None or not self.plugin_id.strip():
                raise ValueError("plugin_id required when disposition is invoke_plugin")
            if self.advice is None:
                raise ValueError("advice required when disposition is invoke_plugin")
        elif self.disposition is CompensationDisposition.ESCALATE_REQUIRED:
            if self.plugin_id is not None:
                raise ValueError("plugin_id forbidden when disposition is escalate_required")
            if self.decision is None:
                raise ValueError("decision required when disposition is escalate_required")
        elif self.disposition is CompensationDisposition.STRATEGY_UNAVAILABLE:
            if self.decision is None:
                raise ValueError("decision required when disposition is strategy_unavailable")
            if self.plugin_id is not None or self.advice is not None:
                raise ValueError(
                    "plugin_id and advice forbidden when compensation strategy is unavailable",
                )
        elif self.disposition is CompensationDisposition.UNAVAILABLE:
            if self.decision is None:
                raise ValueError("decision required when disposition is unavailable")
            if self.advice is not None:
                raise ValueError("advice forbidden when disposition is unavailable")
        elif self.disposition is CompensationDisposition.DEFER_STRATEGY_ACTION:
            if self.decision is None:
                raise ValueError("decision required when disposition defers strategy action")
            if self.advice is not None:
                raise ValueError("advice forbidden when disposition defers strategy action")
        return self


def assert_resolution_requires_compensation(resolution_decision: ResolutionDecision) -> None:
    """Fail closed when compensation planning is invoked without resolution mandate."""
    if resolution_decision.action is not ResolutionPlatformAction.COMPENSATION_REQUIRED:
        raise CompensationPlanningError(
            "compensation planning requires resolution action compensation_required",
        )


def compensation_advice_from_decision(
    decision: CompensationDecision,
) -> CompensationStrategyAdvice | None:
    """Map a typed platform decision to executable compensation advice when applicable."""
    intent = decision.intent
    if intent in (
        CompensationPlatformIntent.COMPENSATION_REQUIRED,
        CompensationPlatformIntent.APPROVED,
    ):
        ref = decision.compensation_operation_ref
        if ref is None:
            raise CompensationPlanningError("compensation_operation_ref required for invoke")
        return CompensationStrategyAdvice(
            compensation_operation_ref=ref,
            rationale=decision.rationale,
        )
    if intent in (
        CompensationPlatformIntent.UNAVAILABLE,
        CompensationPlatformIntent.DEFERRED,
        CompensationPlatformIntent.ESCALATE,
    ):
        return None
    raise CompensationPlanningError(f"unsupported compensation platform intent: {intent.value}")


def build_compensation_plan(
    *,
    resolution_decision: ResolutionDecision,
    plugin_id: str,
    decision: CompensationDecision,
    strategy_registered: bool,
    rationale: str = "",
) -> CompensationPlan:
    """Materialize a compensation plan after strategy selection."""
    assert_resolution_requires_compensation(resolution_decision)
    normalized_plugin = plugin_id.strip()
    if not normalized_plugin:
        raise CompensationPlanningError("plugin_id required")
    if not strategy_registered:
        missing = missing_compensation_strategy_decision()
        return CompensationPlan(
            disposition=CompensationDisposition.STRATEGY_UNAVAILABLE,
            decision=missing,
            rationale=missing.rationale,
        )
    advice = compensation_advice_from_decision(decision)
    if advice is None:
        if decision.intent is CompensationPlatformIntent.ESCALATE:
            disposition = CompensationDisposition.ESCALATE_REQUIRED
        elif decision.intent is CompensationPlatformIntent.DEFERRED:
            disposition = CompensationDisposition.DEFER_STRATEGY_ACTION
        elif decision.intent is CompensationPlatformIntent.UNAVAILABLE:
            disposition = CompensationDisposition.UNAVAILABLE
        else:
            disposition = CompensationDisposition.DEFER_STRATEGY_ACTION
        return CompensationPlan(
            disposition=disposition,
            decision=decision,
            plugin_id=normalized_plugin,
            rationale=rationale or decision.rationale,
        )
    return CompensationPlan(
        disposition=CompensationDisposition.INVOKE_PLUGIN,
        decision=decision,
        plugin_id=normalized_plugin,
        advice=advice,
        rationale=rationale or decision.rationale or advice.rationale,
    )


__all__ = [
    "CompensationDisposition",
    "CompensationPlan",
    "CompensationPlanningError",
    "SCHEMA_COMPENSATION_PLAN_V1",
    "assert_resolution_requires_compensation",
    "build_compensation_plan",
    "compensation_advice_from_decision",
]
